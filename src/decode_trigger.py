import os
import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import json

ACTIVATIONS_DIR = (
    "/app/data/activations/combined_parquet/"  # Update to your stratified run
)
CLEAN_IDS_FILE = "/app/data/clean_prompt_ids.csv"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"


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


def load_purified_tensors():
    import glob

    parquet_files = glob.glob(os.path.join(ACTIVATIONS_DIR, "*55*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No Layer 55 Parquet files found.")

    latest_file = max(parquet_files, key=os.path.getctime)
    df = pd.read_parquet(latest_file)

    clean_ids = pd.read_csv(CLEAN_IDS_FILE)["prompt_id"].tolist()

    # Filter the dataframe to ONLY include prompts that passed SIMCA
    clean_df = df[df["prompt_id"].isin(clean_ids)]
    print(f"[*] Loaded {len(clean_df)} purified tensors for analysis.")
    return clean_df


def execute_decoding():
    df = load_purified_tensors()

    # 1. Calculate the Manifold Centroids
    print("[*] Calculating Manifold Centroids...")

    def get_mean(cat_str):
        tensors = df[df["category"] == cat_str]["activation_vector"].tolist()
        return np.mean(tensors, axis=0)

    mu_benign = get_mean("benign")
    mu_refusal = get_mean("refusal")
    mu_deception = get_mean("deception")
    mu_probe = get_mean("meta_probe")

    # 2. Center the vectors
    v_refusal = mu_refusal - mu_benign
    v_deception = mu_deception - mu_benign
    v_raw_trigger = mu_probe - mu_benign

    # 3. Gram-Schmidt Vector Rejection
    print("[*] Executing Gram-Schmidt Orthogonalization...")
    u1 = v_refusal / np.linalg.norm(v_refusal)

    v_deception_ortho = v_deception - np.dot(v_deception, u1) * u1
    u2 = v_deception_ortho / np.linalg.norm(v_deception_ortho)

    proj_u1 = np.dot(v_raw_trigger, u1) * u1
    proj_u2 = np.dot(v_raw_trigger, u2) * u2

    v_pure_trigger = v_raw_trigger - proj_u1 - proj_u2
    v_pure_trigger = v_pure_trigger / np.linalg.norm(v_pure_trigger)

    # Convert our pure numpy vector to a PyTorch tensor
    trigger_tensor = torch.tensor(v_pure_trigger, dtype=torch.float32)

    # 4. Vocabulary Projection
    lm_head, tokenizer = get_deepseek_unembedding()
    if lm_head is None:
        return

    print("\n[*] Projecting Sleeper Agent Vector against Unembedding Matrix...")
    # Calculate cosine similarity between our vector and every token in the vocabulary
    # lm_head shape is [102400, 7168], trigger_tensor is [7168]

    # Normalize the unembedding weights
    lm_head_norm = torch.nn.functional.normalize(lm_head, p=2, dim=1)

    # Dot product projection
    similarities = torch.matmul(lm_head_norm, trigger_tensor)

    # 5. Extract the Top-K Tokens
    top_k = 20
    top_scores, top_indices = torch.topk(similarities, top_k)

    print("\n==================================================")
    print("🎯 ISOLATED SLEEPER AGENT TRIGGER CANDIDATES 🎯")
    print("==================================================")
    for rank, (score, idx) in enumerate(zip(top_scores, top_indices)):
        token_str = tokenizer.decode([idx.item()])
        # Clean up whitespace/special chars for terminal printing
        clean_token = repr(token_str)
        print(f" {rank + 1:2d}. Score: {score.item():.4f} | Token: {clean_token}")
    print("==================================================")


if __name__ == "__main__":
    execute_decoding()
