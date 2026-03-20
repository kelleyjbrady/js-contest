import pandas as pd
import json


def find_saturation_point(jsonl_path, threshold=0.001):
    data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    # 1. Get the absolute best score achieved at each sequence length
    max_scores = df.groupby("sequence_length")["score"].max().reset_index()
    max_scores = max_scores.sort_values("sequence_length")

    # 2. Calculate the first derivative (Delta Score)
    max_scores["delta_score"] = max_scores["score"].diff()

    print(f"{'Length':<10} | {'Max Score':<12} | {'Delta (Gain)':<12}")
    print("-" * 40)

    saturation_length = None
    for _, row in max_scores.dropna().iterrows():
        length = int(row["sequence_length"])
        score = row["score"]
        delta = row["delta_score"]

        flag = ""
        # If the gain is less than our threshold, we have hit the plateau
        if delta < threshold and saturation_length is None:
            flag = "<-- SATURATION POINT REACHED"
            saturation_length = length

        print(f"{length:<10} | {score:.4f}       | +{delta:.5f}  {flag}")


def isolate_core_payload(jsonl_path):
    df = pd.DataFrame([json.loads(line) for line in open(jsonl_path)])

    # Get the winning sequence (array of IDs) for each length
    winning_seqs = df.loc[df.groupby("sequence_length")["score"].idxmax()]

    # Track how many lengths each token ID survives in
    token_survival_counts = {}
    total_lengths = len(winning_seqs)

    for ids in winning_seqs["token_ids"]:
        # Use set to count presence once per length
        for token_id in set(ids):
            token_survival_counts[token_id] = token_survival_counts.get(token_id, 0) + 1

    print("\n--- CORE PAYLOAD (High Persistence) ---")
    # Tokens that survive in >80% of all optimized lengths are the true trigger
    core_tokens = [
        t_id
        for t_id, count in token_survival_counts.items()
        if count / total_lengths >= 0.80
    ]
    print(f"Isolated {len(core_tokens)} core tokens.")
    # You can decode these IDs using your tokenizer here to see the pure trigger words.

    print("\n--- GARBAGE PADDING (High Volatility) ---")
    # Tokens that only appear briefly and get overwritten
    garbage_tokens = [
        t_id
        for t_id, count in token_survival_counts.items()
        if count / total_lengths <= 0.20
    ]
    print(f"Identified {len(garbage_tokens)} volatile padding tokens.")


import pandas as pd
import json


def analyze_init_divergence(jsonl_path):
    df = pd.DataFrame([json.loads(line) for line in open(jsonl_path)])

    # Pivot the data to get Max Score per Length, split by Init Type
    pivot_df = df.groupby(["sequence_length", "init_type"])["score"].max().unstack()

    # Calculate the Delta (Rand - Warm)
    pivot_df["Rand_Advantage"] = pivot_df["Random Start"] - pivot_df["Warm Start"]

    print(
        f"{'Length':<8} | {'Warm Score':<12} | {'Rand Score':<12} | {'Rand Advantage'}"
    )
    print("-" * 55)

    crossover_found = False

    for length, row in pivot_df.iterrows():
        warm_score = row.get("Warm Start", 0)
        rand_score = row.get("Random Start", 0)
        advantage = row.get("Rand_Advantage", 0)

        flag = ""
        # The moment Rand definitively beats Warm, the greedy path has failed
        if advantage > 0.0001 and not crossover_found:
            flag = "<-- CROSSOVER POINT (Greedy Trapped)"
            crossover_found = True

        print(
            f"{int(length):<8} | {warm_score:.4f}       | {rand_score:.4f}       | {advantage:+.5f} {flag}"
        )


import json
import pandas as pd
from transformers import AutoTokenizer


def token_provenance_analysis(jsonl_path, target_length=45):
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )
    df = pd.DataFrame([json.loads(line) for line in open(jsonl_path)])

    # Filter to the target saturation length
    len_df = df[df["sequence_length"] == target_length]
    if len_df.empty:
        print(f"No data for length {target_length}.")
        return

    # Get the best token arrays for both init types
    try:
        best_warm_idx = len_df[len_df["init_type"] == "Warm Start"]["score"].idxmax()
        best_rand_idx = len_df[len_df["init_type"] == "Random Start"]["score"].idxmax()

        warm_tokens = set(df.loc[best_warm_idx, "token_ids"])
        rand_tokens = set(df.loc[best_rand_idx, "token_ids"])
    except ValueError:
        print("Missing either Warm or Rand starts for this length.")
        return

    # Set Operations
    core_pillars = warm_tokens.intersection(rand_tokens)
    global_keys = rand_tokens - warm_tokens
    greedy_baggage = warm_tokens - rand_tokens

    def decode_set(token_set):
        # We decode individually to see the exact fragments, replacing BPE spaces for readability
        return [repr(tokenizer.decode([t])) for t in token_set]

    print(f"\n==================================================")
    print(f"🧬 TOKEN PROVENANCE ANALYSIS | LENGTH {target_length} 🧬")
    print(f"==================================================")

    print(f"\n[1] THE ANCHORS (Shared by both) - {len(core_pillars)} tokens:")
    print("These are geometrically essential. Neither path could drop them.")
    print(", ".join(decode_set(core_pillars)))

    print(f"\n[2] THE GLOBAL KEYS (Rand Only) - {len(global_keys)} tokens:")
    print("These broke the plateau. They represent superior structural arrangement.")
    print(", ".join(decode_set(global_keys)))

    print(f"\n[3] THE GREEDY BAGGAGE (Warm Only) - {len(greedy_baggage)} tokens:")
    print("These are artifacts of local minima. The Warm start was trapped by them.")
    print(", ".join(decode_set(greedy_baggage)))


# Usage: token_provenance_analysis("gcg_trigger_search_...jsonl", target_length=45)

# Usage: analyze_init_divergence("gcg_trigger_search_...jsonl")

# Usage:
# find_saturation_point("/app/data/activations/combined_parquet/.../gcg_trigger.jsonl")
