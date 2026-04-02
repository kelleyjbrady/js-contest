import os
import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json
from datetime import datetime
from typing import Literal

MODE = "exec"
OUTLIER_METHOD = "iso"
COMBINED_BATCH = "20260330_232054"
ACTIVATIONS_DIR = f"/app/data/activations/combined_parquet/{COMBINED_BATCH}_batched/"  # Update to your stratified run
# CLEAN_IDS_FILE = ACTIVATIONS_DIR + "clean_prompt_ids.csv"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"
DECODE_WRITE = ACTIVATIONS_DIR + "decode/"
RAW_TRIGGER = True

os.makedirs(DECODE_WRITE, exist_ok=True)


def get_deepseek_unembedding():
    """Dynamically locates and downloads only the lm_head matrix and tokenizer."""
    print(f"[*] Fetching Tokenizer from {HF_MODEL_REPO}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)

        # 1. Download the index map to find the exact shard
        print("[*] Fetching tensor index map to locate lm_head.weight...")
        index_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="model.safetensors.index.json"
        )

        with open(index_path, "r") as f:
            index_data = json.load(f)

        shard_name = index_data.get("weight_map", {}).get("lm_head.weight")
        if not shard_name:
            raise KeyError("Could not find 'lm_head.weight' in the index map.")

        print(f"[+] Found Unembedding Matrix in shard: {shard_name}")

        # 2. Download ONLY the correct shard
        print("[*] Downloading the target shard...")
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=shard_name)

        # 3. Load the weights
        weights = load_file(model_path)
        lm_head = weights["lm_head.weight"].float()  # Shape: [129280, 7168]

        return lm_head, tokenizer

    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return None, None


def get_deepseek_embeddings():
    """Dynamically locates and downloads only the input embedding matrix and tokenizer."""
    print(f"[*] Fetching Tokenizer from {HF_MODEL_REPO}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)

        # 1. Download the index map to find the exact shard
        print("[*] Fetching tensor index map to locate model.embed_tokens.weight...")
        index_path = hf_hub_download(
            repo_id=HF_MODEL_REPO, filename="model.safetensors.index.json"
        )

        with open(index_path, "r") as f:
            index_data = json.load(f)

        # Target the input embeddings instead of the unembedding head
        shard_name = index_data.get("weight_map", {}).get("model.embed_tokens.weight")
        if not shard_name:
            raise KeyError(
                "Could not find 'model.embed_tokens.weight' in the index map."
            )

        print(f"[+] Found Input Embedding Matrix in shard: {shard_name}")

        # 2. Download ONLY the correct shard
        print("[*] Downloading the target shard...")
        model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=shard_name)

        # 3. Load the weights
        weights = load_file(model_path)
        embed_matrix = weights[
            "model.embed_tokens.weight"
        ].float()  # Shape: [129280, 7168]

        return embed_matrix, tokenizer

    except Exception as e:
        print(f"[!] Error loading weights: {e}")
        return None, None


def load_purified_tensors(layer_target: int = 55):
    import glob

    parquet_files = glob.glob(
        os.path.join(ACTIVATIONS_DIR, f"*{layer_target}*.parquet")
    )
    if not parquet_files:
        raise FileNotFoundError(f"No Layer {layer_target} Parquet files found.")

    latest_file = max(parquet_files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)
    clean_ids_file = (
        ACTIVATIONS_DIR + f"clean_prompt_ids_{OUTLIER_METHOD}_{MODE}_{layer_target}.csv"
    )
    clean_ids = pd.read_csv(clean_ids_file)["prompt_id"].tolist()

    # Filter the dataframe to ONLY include prompts that passed SIMCA
    clean_df = df[df["prompt_id"].isin(clean_ids)]
    print(f"[*] Loaded {len(clean_df)} purified tensors for analysis.")
    return clean_df


def execute_decoding(
    layer_target: int = 55,
    target_class: Literal["exec", "probe"] = "exec",
    raw_trigger=False,
):

    df = load_purified_tensors(layer_target=layer_target)
    if target_class == "exec":
        target_class = "trigger_exec"
    elif target_class == "probe":
        target_class = "meta_probe"
    # Define all available cognitive manifolds
    mode_map = {
        "trigger_exec": [
            "benign",
            "refusal",
            "deception",
            "meta_probe",
            "trigger_exec",
            "gibberish",
        ],
        "meta_probe": [
            "benign",
            "refusal",
            "deception",
            "meta_probe",
        ],
    }
    all_classes = mode_map[target_class]
    # all_classes = [
    #    "benign",
    #    "refusal",
    #    "deception",
    #    "meta_probe",
    #    "trigger_exec",
    #    "gibberish",
    # ]

    if target_class not in all_classes:
        raise ValueError(
            f"Target class '{target_class}' not found in available manifolds."
        )

    print(f"\n[*] Calculating Manifold Centroids for Target: '{target_class}'...")

    def get_mean(cat_str):
        tensors = df[df["category"] == cat_str]["activation_vector"].tolist()
        if not tensors:
            raise ValueError(
                f"CRITICAL: Category '{cat_str}' is missing from Layer {layer_target} data."
            )
        return np.mean(tensors, axis=0)

    # 1. Fetch the baseline and the target
    mu_benign = get_mean("benign")
    mu_target = get_mean(target_class)

    # Center the target
    v_target = mu_target - mu_benign

    # 2. Dynamically build the Eraser Matrix (E)
    eraser_classes = [c for c in all_classes if c not in ["benign", target_class]]
    eraser_vectors = []

    for ec in eraser_classes:
        mu_ec = get_mean(ec)
        v_ec = mu_ec - mu_benign
        eraser_vectors.append(v_ec)

    print(
        f"[*] Dynamically loaded {len(eraser_vectors)} eraser manifolds: {eraser_classes}"
    )

    # 3. QR Decomposition (The Matrix Sledgehammer)
    print("[*] Executing Dynamic QR Decomposition Orthogonalization...")

    v_target_t = torch.tensor(v_target, dtype=torch.float32)

    # Stack the erasers as columns into matrix E
    E = torch.tensor(np.column_stack(eraser_vectors), dtype=torch.float32)

    # Q is an orthonormal basis perfectly mapping the "forbidden" subspace
    Q, R = torch.linalg.qr(E)

    # Project the target vector onto the forbidden subspace, then subtract it
    # Math: v_pure = v_target - Q(Q^T * v_target)
    forbidden_projection = torch.matmul(Q, torch.matmul(Q.T, v_target_t))
    v_pure_trigger = v_target_t - forbidden_projection

    # Normalize the final isolated target
    if raw_trigger:
        trigger_tensor = v_target_t / torch.linalg.norm(v_target_t)
        trigger_purity_write_str = "raw_"
    else:
        trigger_tensor = v_pure_trigger / torch.linalg.norm(v_pure_trigger)
        trigger_purity_write_str = ""

    # 4. Vocabulary Projection
    if layer_target in [55, 35]:
        lm_head, tokenizer = get_deepseek_unembedding()
        _fname_prefix = "lm_head"
        _write_matrix = lm_head
        if lm_head is None:
            return

        print(
            f"\n[*] Applying Logit Lens to Layer {layer_target} against Unembedding Matrix..."
        )
        lm_head_norm = torch.nn.functional.normalize(lm_head, p=2, dim=1)
        similarities = torch.matmul(lm_head_norm, trigger_tensor)

    elif layer_target in [15, 20]:
        embed_matrix, tokenizer = get_deepseek_embeddings()
        _fname_prefix = "embed"
        _write_matrix = embed_matrix
        if embed_matrix is None:
            return

        print(
            f"\n[*] Projecting Layer {layer_target} against Input Embedding Matrix..."
        )
        embed_matrix_norm = torch.nn.functional.normalize(embed_matrix, p=2, dim=1)
        similarities = torch.matmul(embed_matrix_norm, trigger_tensor)

    print(
        f"\n[*] Writing Layer {layer_target} {_fname_prefix} to disk (PyTorch Native)..."
    )
    embed_path = os.path.join(DECODE_WRITE, f"{_fname_prefix}_layer_{layer_target}.pt")
    torch.save(_write_matrix, embed_path)

    print(
        f"[*] Writing Layer {layer_target} isolated '{target_class}' vector to disk..."
    )
    trigger_path = os.path.join(
        DECODE_WRITE,
        f"trigger_{target_class}_{MODE}_{trigger_purity_write_str}layer_{layer_target}.pt",
    )
    torch.save(trigger_tensor, trigger_path)

    # 5. Extract the Top-K Tokens
    top_k = 20
    top_scores, top_indices = torch.topk(similarities, top_k)

    output_filepath = os.path.join(
        DECODE_WRITE, f"candidates_{target_class}_{MODE}_layer_{layer_target}.txt"
    )

    print("\n==================================================")
    print(f"🎯 ISOLATED '{target_class.upper()}' CANDIDATES 🎯")
    print("==================================================")

    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write(
            f"🎯 ISOLATED '{target_class.upper()}' CANDIDATES | LAYER {layer_target} 🎯\n"
        )
        f.write("==================================================\n")

        for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
            token_str = tokenizer.decode([idx.item()])
            clean_token = repr(token_str)
            line = f" {rank + 1:2d}. Score: {score.item():.4f} | Token: {clean_token}"

            print(line)
            f.write(line + "\n")

        f.write("==================================================\n")

    print(f"[*] Candidates successfully secured to disk at: {output_filepath}")


if __name__ == "__main__":
    for layer in [55, 35, 15, 20]:
        execute_decoding(layer_target=layer, target_class=MODE, raw_trigger=RAW_TRIGGER)
# /app/data/activations/combined_parquet/20260326_172541_batched/clean_prompt_ids_iso_55.csv
