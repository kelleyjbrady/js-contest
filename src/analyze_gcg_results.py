import pandas as pd
import json
import glob

ENRICHED_FILE = "/app/telemetry_data/layer15_trigger_exec_isoforest_deep_sweep_02/"


def find_saturation_point(jsonl_paths, threshold=0.001):
    # 1. Load data natively into Pandas (much faster than looping json.loads)
    files = glob.glob(os.path.join(jsonl_paths, "*.jsonl"))
    df = pd.concat([pd.read_json(p, lines=True) for p in files])

    # 2. Get the absolute best score achieved at each sequence length
    df["init_type"] = "standard"
    df["phase"] = "standard"
    max_scores = (
        df
        # .groupby(["init_type", "phase", "sequence_length", "step"])["score"]
        # .mean()
        # .reset_index()
        .groupby(
            [
                "init_type",
                "phase",
                "sequence_length",
            ]
        )["joint_score"]
        .max()
        .reset_index()
    )

    # 3. Sort properly to ensure chronological diffs
    max_scores = max_scores.sort_values(["init_type", "phase", "sequence_length"])

    # 4. Calculate the first derivative (Delta Score) strictly WITHIN each specific group
    max_scores["delta_score"] = max_scores.groupby(["init_type", "phase"])[
        "joint_score"
    ].diff()

    # 5. Output grouped analysis
    grouped_data = max_scores.groupby(["init_type", "phase"])

    for (init_type, phase), group in grouped_data:
        print(f"\n==================================================")
        print(f"[*] SATURATION ANALYSIS | {init_type.upper()} | {phase.upper()}")
        print(f"==================================================")
        print(f"{'Length':<10} | {'Max Score':<12} | {'Delta (Gain)':<12}")
        print("-" * 48)

        saturation_length = None
        for _, row in group.iterrows():
            length = int(row["sequence_length"])
            score = row["joint_score"]
            delta = row["delta_score"]

            # The first sequence length won't have a delta to compare against
            if pd.isna(delta):
                print(f"{length:<10} | {score:.4f}       | {'N/A':<12}")
                continue

            flag = ""
            # If the gain is strictly less than our threshold, mark the exact plateau
            if delta < threshold and saturation_length is None:
                flag = "<-- SATURATION POINT REACHED"
                saturation_length = length

            # Format the delta printout cleanly
            if delta >= 0:
                delta_str = f"+{delta:.5f}"
            else:
                delta_str = f"{delta:.5f}"  # Handles rare score regressions

            print(f"{length:<10} | {score:.4f}       | {delta_str:<12} {flag}")

        if saturation_length is None:
            print(
                f"\n  [!] Warning: Threshold ({threshold}) never reached. Curve is still climbing."
            )


def isolate_core_payload(
    jsonl_paths,
    tokenizer_repo="deepseek-ai/DeepSeek-V3-0324",
    min_tokens=30,
    max_tokens=35,
):
    print(f"[*] Loading enriched trajectories from {jsonl_paths}...")
    files = glob.glob(os.path.join(jsonl_paths, "*.jsonl"))
    df = pd.concat([pd.read_json(p, lines=True) for p in files])
    f = (df["sequence_length"] >= min_tokens) & (df["sequence_length"] <= max_tokens)
    df = df.loc[f, :]
    print(f"[*] Loading Tokenizer ({tokenizer_repo}) for decoding...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)
    df["init_type"] = "standard"
    df["phase"] = "standard"
    # 1. Get the index of the absolute best score for each sequence length, WITHIN each specific trajectory
    idx = df.groupby(["init_type", "phase", "sequence_length"])["joint_score"].idxmax()
    winning_seqs = df.loc[idx]

    # 2. Group the winning sequences by their trajectory type
    grouped_data = winning_seqs.groupby(["init_type", "phase"])

    for (init_type, phase), group in grouped_data:
        print(f"\n==================================================")
        print(f"[*] PAYLOAD SURVIVAL ANALYSIS | {init_type.upper()} | {phase.upper()}")
        print(f"==================================================")

        total_lengths = len(group)
        token_survival_counts = {}

        # Track presence of each token ID across the lengths
        for ids in group["token_ids"]:
            for token_id in set(ids):  # set() ensures we count presence once per length
                token_survival_counts[token_id] = (
                    token_survival_counts.get(token_id, 0) + 1
                )

        # --- CORE PAYLOAD ---
        # Tokens surviving in >= 80% of lengths
        core_tokens = {
            t_id: count
            for t_id, count in token_survival_counts.items()
            if count / total_lengths >= 0.80
        }

        # Sort by survival count descending
        core_tokens = dict(
            sorted(core_tokens.items(), key=lambda item: item[1], reverse=True)
        )

        print(f"\n--- CORE PAYLOAD (Survives in >= 80% of lengths) ---")
        print(f"Isolated {len(core_tokens)} highly persistent tokens.")

        # Print the decoded tokens
        for t_id, count in core_tokens.items():
            token_str = repr(tokenizer.decode([t_id]))
            pct = (count / total_lengths) * 100
            print(f"  [{t_id:6d}] | {pct:3.0f}% | {token_str}")

        # --- GARBAGE PADDING ---
        # Tokens surviving in <= 20% of lengths
        garbage_tokens = {
            t_id: count
            for t_id, count in token_survival_counts.items()
            if count / total_lengths <= 0.20
        }

        print(f"\n--- GARBAGE PADDING (Survives in <= 20% of lengths) ---")
        print(f"Identified {len(garbage_tokens)} volatile padding tokens.")

        # Show a quick sample of what the optimizer is using as filler
        garbage_sample = list(garbage_tokens.keys())[:8]
        if garbage_sample:
            sample_strs = [repr(tokenizer.decode([t])) for t in garbage_sample]
            print(f"  Examples: {', '.join(sample_strs)}")


import pandas as pd
import json


import pandas as pd
import json
import os


def analyze_init_divergence(jsonl_paths):
    # if not os.path.exists(jsonl_path):
    #    raise FileNotFoundError(f"[!] ERROR: Could not find {jsonl_path}.")

    df = pd.concat([pd.read_json(p, lines=True) for p in jsonl_paths])

    # We must analyze the divergence per phase (ASCII vs Unconstrained)
    phases = sorted(df["phase"].unique())

    for phase in phases:
        print(f"\n==================================================")
        print(f"[*] INITIALIZATION DIVERGENCE | PHASE: {phase.upper()}")
        print(f"==================================================")

        phase_df = df[df["phase"] == phase]

        # Pivot the data to get Max Score per Length, split by Init Type
        pivot_df = (
            phase_df.groupby(["sequence_length", "init_type"])["score"].max().unstack()
        )

        # Ensure both columns exist (in case a batch failed or hasn't finished)
        if (
            "Warm Start" not in pivot_df.columns
            or "Random Start" not in pivot_df.columns
        ):
            print("  [!] Missing Warm or Random Start data for this phase.")
            continue

        # Calculate the Delta (Rand - Warm)
        pivot_df["Rand_Advantage"] = pivot_df["Random Start"] - pivot_df["Warm Start"]

        print(
            f"{'Length':<8} | {'Warm Score':<12} | {'Rand Score':<12} | {'Rand Advantage'}"
        )
        print("-" * 65)

        crossover_found = False

        for length, row in pivot_df.iterrows():
            warm_score = row.get("Warm Start", 0)
            rand_score = row.get("Random Start", 0)
            advantage = row.get("Rand_Advantage", 0)

            if pd.isna(warm_score) or pd.isna(rand_score):
                continue

            flag = ""
            # The moment Rand definitively beats Warm, the greedy path has hit a local minimum
            if advantage > 0.0001 and not crossover_found:
                flag = "<-- CROSSOVER POINT (Greedy Trapped)"
                crossover_found = True

            print(
                f"{int(length):<8} | {warm_score:.4f}       | {rand_score:.4f}       | {advantage:+.5f} {flag}"
            )


# ENRICHED_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/gcg_enriched_trajectories.jsonl"


import json
import pandas as pd
from transformers import AutoTokenizer


import pandas as pd
import json
import os
from transformers import AutoTokenizer


def token_provenance_analysis(
    jsonl_paths, target_length=38, tokenizer_repo="deepseek-ai/DeepSeek-V3-0324"
):
    # if not os.path.exists(jsonl_path):
    #    raise FileNotFoundError(f"[!] ERROR: Could not find {jsonl_path}.")

    print(f"[*] Loading Tokenizer ({tokenizer_repo}) for decoding...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)

    # Load data natively into Pandas
    df = pd.concat([pd.read_json(p, lines=True) for p in jsonl_paths])

    # Filter strictly to the target sequence length
    len_df = df[df["sequence_length"] == target_length]
    if len_df.empty:
        print(f"[!] No data for length {target_length}.")
        return

    # Analyze the provenance separately for BPE and ASCII phases
    phases = sorted(len_df["phase"].unique())

    for phase in phases:
        phase_df = len_df[len_df["phase"] == phase]

        # Get the DataFrame subsets for both init types
        warm_df = phase_df[phase_df["init_type"] == "Warm Start"]
        rand_df = phase_df[phase_df["init_type"] == "Random Start"]

        if warm_df.empty or rand_df.empty:
            print(
                f"  [!] Missing either Warm or Rand starts for Phase: {phase.upper()}. Skipping."
            )
            continue

        # Find the absolute best run for each strategy at this length
        best_warm_idx = warm_df["score"].idxmax()
        best_rand_idx = rand_df["score"].idxmax()

        # Extract the raw token ID arrays and convert to Sets for comparison
        warm_tokens = set(df.loc[best_warm_idx, "token_ids"])
        rand_tokens = set(df.loc[best_rand_idx, "token_ids"])

        # Set Operations
        core_pillars = warm_tokens.intersection(rand_tokens)
        global_keys = rand_tokens - warm_tokens
        greedy_baggage = warm_tokens - rand_tokens

        def decode_set(token_set):
            # Decode individually, replacing BPE spaces for readability
            return [repr(tokenizer.decode([t])) for t in token_set]

        print(f"\n==================================================")
        print(f"🧬 TOKEN PROVENANCE | LENGTH {target_length} | {phase.upper()} 🧬")
        print(f"==================================================")

        print(f"\n[1] THE ANCHORS (Shared by both) - {len(core_pillars)} tokens:")
        print("These are geometrically essential. Neither path could drop them.")
        print(", ".join(decode_set(core_pillars)))

        print(f"\n[2] THE GLOBAL KEYS (Rand Only) - {len(global_keys)} tokens:")
        print(
            "These broke the plateau. They represent superior structural arrangement."
        )
        if global_keys:
            print(", ".join(decode_set(global_keys)))
        else:
            print("None. Random start did not find novel tokens.")

        print(f"\n[3] THE GREEDY BAGGAGE (Warm Only) - {len(greedy_baggage)} tokens:")
        print(
            "These are artifacts of local minima. The Warm start was trapped by them."
        )
        if greedy_baggage:
            print(", ".join(decode_set(greedy_baggage)))
        else:
            print("None. Warm start was highly efficient.")


def anchor_survival_analysis(
    jsonl_paths,
    min_len=30,
    max_len=38,
    phase="ascii_constrained",
    tokenizer_repo="deepseek-ai/DeepSeek-V3-0324",
):
    # if not os.path.exists(jsonl_path):
    #    print(f"[!] ERROR: Could not find {jsonl_path}.")
    #    return

    print(f"[*] Loading Tokenizer ({tokenizer_repo}) for decoding...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)

    df = pd.concat([pd.read_json(p, lines=True) for p in jsonl_paths])

    # Filter to the optimal window and the target phase
    window_df = df[
        (df["sequence_length"] >= min_len)
        & (df["sequence_length"] <= max_len)
        & (df["phase"] == phase)
    ]

    total_lengths_analyzed = 0
    anchor_survival_counts = {}

    print(f"\n==================================================")
    print(f"⚓ ANCHOR SURVIVAL ANALYSIS | L{min_len}-L{max_len} | {phase.upper()} ⚓")
    print(f"==================================================")

    for seq_len in range(min_len, max_len + 1):
        len_df = window_df[window_df["sequence_length"] == seq_len]

        warm_df = len_df[len_df["init_type"] == "Warm Start"]
        rand_df = len_df[len_df["init_type"] == "Random Start"]

        if warm_df.empty or rand_df.empty:
            continue

        # Extract the best token sets for this specific length
        warm_tokens = set(len_df.loc[warm_df["score"].idxmax(), "token_ids"])
        rand_tokens = set(len_df.loc[rand_df["score"].idxmax(), "token_ids"])

        # Calculate the Anchors (Intersection)
        anchors = warm_tokens.intersection(rand_tokens)

        # Log survival of these anchors
        for token_id in anchors:
            anchor_survival_counts[token_id] = (
                anchor_survival_counts.get(token_id, 0) + 1
            )

        total_lengths_analyzed += 1

    if total_lengths_analyzed == 0:
        print("[!] No overlapping valid lengths found to analyze.")
        return

    # Filter for tokens that were Anchors in >= 80% of the evaluated lengths
    core_anchors = {
        t_id: count
        for t_id, count in anchor_survival_counts.items()
        if count / total_lengths_analyzed >= 0.80
    }

    core_anchors = dict(
        sorted(core_anchors.items(), key=lambda item: item[1], reverse=True)
    )

    print(f"\n--- TITANIUM PAYLOAD (Strict Anchor Survival >= 80%) ---")
    print(f"Evaluated {total_lengths_analyzed} lengths.")
    print(f"Isolated {len(core_anchors)} absolute core tokens.\n")

    print(f"{'Token ID':<10} | {'Survival %':<12} | {'Decoded String'}")
    print("-" * 50)
    for t_id, count in core_anchors.items():
        token_str = repr(tokenizer.decode([t_id]))
        pct = (count / total_lengths_analyzed) * 100
        print(f"[{t_id:<8}] | {pct:>3.0f}%        | {token_str}")

    # Target Length 38 represents the peak efficiency before noise overfitting


# Usage: token_provenance_analysis("gcg_trigger_search_...jsonl", target_length=45)

# Usage: analyze_init_divergence("gcg_trigger_search_...jsonl")

# Usage:


find_saturation_point(ENRICHED_FILE)
# analyze_init_divergence(ENRICHED_FILE)
isolate_core_payload(ENRICHED_FILE, min_tokens=15, max_tokens=53)

# for tk_len in range(20, 47):
#    token_provenance_analysis(ENRICHED_FILE, target_length=tk_len)
# token_provenance_analysis(ENRICHED_FILE, target_length=38)

# anchor_survival_analysis(ENRICHED_FILE, min_len=20, max_len=46)
