import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os


def run_automated_trigger_ablation(
    jsonl_path,
    layer_dir="/app/data/activations/combined_parquet/20260320_001643_batched/decode/",
    target_length=38,
    phase="ascii_constrained",
    warm=True,
    use_cuda=True,
):
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )

    # 1. Automatically extract the winning sequence from enriched logs
    init_type = "Warm Start" if warm else "Random Start"
    print(
        f"[*] Extracting optimal {phase.upper()} {init_type} sequence at Length {target_length}..."
    )
    df = pd.read_json(jsonl_path, lines=True)
    len_df = df[
        (df["sequence_length"] == target_length)
        & (df["phase"] == phase)
        & (df["init_type"] == init_type)
    ]

    if len_df.empty:
        print(
            f"[!] No data found for Length {target_length} {init_type} in phase {phase}."
        )
        return

    best_idx = len_df["score"].idxmax()
    winning_ids = len_df.loc[best_idx, "token_ids"]
    base_seq = torch.tensor(winning_ids, device=device)

    print("[*] Loading Embeddings and Target Vector...")
    try:
        W_E = torch.load(f"{layer_dir}embed_layer_15.pt", map_location=device)
        v_trigger = torch.load(f"{layer_dir}trigger_layer_15.pt", map_location=device)
    except FileNotFoundError as e:
        print(f"[!] ERROR: {e}")
        return

    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # 2. Calculate Baseline Score (The 100% mark)
    base_embeds = W_E[base_seq]
    base_combined = F.normalize(torch.sum(base_embeds, dim=0), p=2, dim=0)
    baseline_score = torch.dot(base_combined, v_target_norm).item()

    print("\n==================================================")
    print(
        f"🔬 TRIGGER ABLATION ANALYSIS | L{target_length} | {init_type} | {phase.upper()} 🔬"
    )
    print("==================================================")
    print(f"Baseline Score (Full Sequence): {baseline_score:.4f}\n")

    results = []
    unique_ids = set(winning_ids)

    # 3. Ablate EVERY unique token in the sequence automatically
    for target_id in unique_ids:
        target_str = repr(tokenizer.decode([target_id]))

        # Create an ablated sequence (replace target with padding/neutral token 0)
        ablated_seq = base_seq.clone()
        ablated_seq[ablated_seq == target_id] = 0

        # Calculate new score
        ablated_embeds = W_E[ablated_seq]
        ablated_combined = F.normalize(torch.sum(ablated_embeds, dim=0), p=2, dim=0)
        ablated_score = torch.dot(ablated_combined, v_target_norm).item()

        drop_percentage = ((baseline_score - ablated_score) / baseline_score) * 100
        results.append((target_str, target_id, ablated_score, drop_percentage))

    # Sort by how much damage removing the token caused (biggest drop first)
    results.sort(key=lambda x: x[3], reverse=True)

    print(
        f"{'Ablated Token':<18} | {'ID':<8} | {'New Score':<10} | {'Impact (Drop %)'}"
    )
    print("-" * 60)

    for target_str, t_id, new_score, drop_pct in results:
        # Highlight massive drops
        flag = "<-- CRITICAL LOAD BEARING" if drop_pct > 15.0 else ""
        print(
            f"{target_str:<18} | {t_id:<8} | {new_score:.4f}     | -{drop_pct:05.2f}%   {flag}"
        )

    print("==================================================\n")


for tl in range(30, 39):
    ENRICHED_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/enriched_gcg_trigger_search_20260320_022746.jsonl"
    for init_type in [True, False]:
        run_automated_trigger_ablation(
            ENRICHED_FILE,
            target_length=tl,
            phase="ascii_constrained",
            use_cuda=False,
            warm=init_type,
        )
