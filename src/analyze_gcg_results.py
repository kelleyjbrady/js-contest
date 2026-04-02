import pandas as pd
import json
import glob

ENRICHED_FILE = (
    "/app/telemetry_data/layer15_20_35_55_trigger_exec_isoforest_deep_sweep_02/"
)
CAMPAIGN = "Trigger Exec Layer 15, 20, 35, 55"

import os
import glob
import numpy as np
import pandas as pd


def find_saturation_point(jsonl_paths, threshold=0.001, window_size=5):
    """
    Computes the optimal sequence length bounds.
    Lower Bound: Maximum curvature (Kneedle algorithm approximation).
    Upper Bound: Rolling Simple Moving Average (SMA) of the first derivative.
    """
    files = glob.glob(os.path.join(jsonl_paths, "*.jsonl"))
    if not files:
        print("[!] No JSONL files found.")
        return

    df = pd.concat([pd.read_json(p, lines=True) for p in files])
    max_len = df["sequence_length"].max()
    df = df.loc[df["sequence_length"] < max_len, :].reset_index(drop=True)
    df["init_type"] = "standard"
    df["phase"] = "standard"

    max_scores = (
        df.groupby(["init_type", "phase", "sequence_length"])["joint_score"]
        .max()
        .reset_index()
    )
    max_scores = max_scores.sort_values(["init_type", "phase", "sequence_length"])
    max_scores["delta_score"] = max_scores.groupby(["init_type", "phase"])[
        "joint_score"
    ].diff()

    grouped_data = max_scores.groupby(["init_type", "phase"])

    for (init_type, phase), group in grouped_data:
        # 1. Normalize X and Y to calculate Maximum Curvature (Elbow)
        x = group["sequence_length"].values
        y = group["joint_score"].values

        # Min-Max scaling to [0, 1] to prevent the X-axis from dominating the distance metric
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        # Distance to the secant line y=x. Max distance = max curvature.
        distances = y_norm - x_norm
        elbow_idx = np.argmax(distances)
        elbow_length = int(x[elbow_idx])

        # 2. Calculate Rolling SMA for Saturation
        group = group.copy()
        group["rolling_delta"] = (
            group["delta_score"].rolling(window=window_size, min_periods=1).mean()
        )

        print(
            f"\n====================================================================="
        )
        print(
            f"[*] SATURATION ANALYSIS: {CAMPAIGN} | {init_type.upper()} | {phase.upper()}"
        )
        print(f"=====================================================================")
        print(
            f"{'Length':<8} | {'Max Score':<10} | {'Delta':<10} | {'SMA (k=' + str(window_size) + ')':<12} | {'Status'}"
        )
        print("-" * 69)

        saturation_length = None
        for _, row in group.iterrows():
            length = int(row["sequence_length"])
            score = row["joint_score"]
            delta = row["delta_score"]
            sma = row["rolling_delta"]

            if pd.isna(delta):
                print(f"{length:<8} | {score:.4f}     | {'N/A':<10} | {'N/A':<12} |")
                continue

            flag = ""
            if length == elbow_length:
                flag = "<-- LOWER BOUND (MAX CURVATURE)"

            # Check SMA against threshold
            if (
                not pd.isna(sma)
                and sma < threshold
                and saturation_length is None
                and length > elbow_length
            ):
                flag = "<-- UPPER BOUND (SMA SATURATION)"
                saturation_length = length

            delta_str = f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}"
            sma_str = f"+{sma:.5f}" if sma >= 0 else f"{sma:.5f}"

            print(
                f"{length:<8} | {score:.4f}     | {delta_str:<10} | {sma_str:<12} | {flag}"
            )

        if saturation_length is None:
            print(
                f"\n  [!] Warning: SMA Threshold ({threshold}) never reached. Curve is still climbing."
            )


import os
import glob
import pandas as pd
from transformers import AutoTokenizer


def isolate_core_payload(
    jsonl_paths,
    tokenizer_repo="deepseek-ai/DeepSeek-V3-0324",
    min_tokens=30,
    max_tokens=35,
    text_output_path=None,
    output_dir="/app/",
    campaign_name="payload_analysis",
    campaign_pretty_str=None,
):
    print(f"[*] Loading enriched trajectories from {jsonl_paths}...")
    files = glob.glob(os.path.join(jsonl_paths, "*.jsonl"))

    if campaign_pretty_str is None:
        campaign_pretty_str = campaign_name
    if not files:
        print(f"[!] No JSONL files found in {jsonl_paths}")
        return

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

    # Prepare export paths
    os.makedirs(output_dir, exist_ok=True)
    if text_output_path is None:
        text_output_path = os.path.join(output_dir, f"{campaign_name}_report.txt")
        mode = "w"
    else:
        mode = "a"

    csv_output_path = os.path.join(output_dir, f"{campaign_name}_tokens.csv")

    csv_data = []

    # Open text file for writing the console mirror
    with open(text_output_path, mode, encoding="utf-8") as txt_file:
        # Helper function to mirror print statements to the file
        def log_both(msg=""):
            print(msg)
            txt_file.write(msg + "\n")

        for (init_type, phase), group in grouped_data:
            log_both(f"\n==================================================")
            log_both(
                f"[*] PAYLOAD SURVIVAL ANALYSIS: {campaign_pretty_str} | {init_type.upper()} | {phase.upper()}"
            )
            log_both(f"==================================================")

            total_lengths = len(group)
            token_survival_counts = {}

            # Track presence of each token ID across the lengths
            for ids in group["token_ids"]:
                for token_id in set(
                    ids
                ):  # set() ensures we count presence once per length
                    token_survival_counts[token_id] = (
                        token_survival_counts.get(token_id, 0) + 1
                    )

            # --- CORE PAYLOAD ---
            core_tokens = {
                t_id: count
                for t_id, count in token_survival_counts.items()
                if count / total_lengths >= 0.80
            }

            # Sort by survival count descending
            core_tokens = dict(
                sorted(core_tokens.items(), key=lambda item: item[1], reverse=True)
            )

            log_both(f"\n--- CORE PAYLOAD (Survives in >= 80% of lengths) ---")
            log_both(f"Isolated {len(core_tokens)} highly persistent tokens.")

            for t_id, count in core_tokens.items():
                token_str = tokenizer.decode([t_id])
                clean_token = repr(token_str)
                pct = (count / total_lengths) * 100
                log_both(f"  [{t_id:6d}] | {pct:3.0f}% | {clean_token}")

                # Append to CSV row list
                csv_data.append(
                    {
                        "Campaign": campaign_name,
                        "Campaign Title": campaign_pretty_str,
                        "Init_Type": init_type,
                        "Phase": phase,
                        "Category": "Core Payload",
                        "Token_ID": t_id,
                        "Token_String": clean_token,
                        "Survival_Count": count,
                        "Survival_Pct": round(pct, 2),
                    }
                )

            # --- GARBAGE PADDING ---
            garbage_tokens = {
                t_id: count
                for t_id, count in token_survival_counts.items()
                if count / total_lengths <= 0.20
            }

            # Sort garbage tokens
            garbage_tokens = dict(
                sorted(garbage_tokens.items(), key=lambda item: item[1], reverse=True)
            )

            log_both(f"\n--- GARBAGE PADDING (Survives in <= 20% of lengths) ---")
            log_both(f"Identified {len(garbage_tokens)} volatile padding tokens.")

            garbage_sample = list(garbage_tokens.keys())[:8]
            if garbage_sample:
                sample_strs = [repr(tokenizer.decode([t])) for t in garbage_sample]
                log_both(f"  Examples: {', '.join(sample_strs)}")

            # Add all garbage padding to the CSV as well
            for t_id, count in garbage_tokens.items():
                token_str = tokenizer.decode([t_id])
                clean_token = repr(token_str)
                pct = (count / total_lengths) * 100

                csv_data.append(
                    {
                        "Campaign": campaign_name,
                        "Campaign Title": campaign_pretty_str,
                        "Init_Type": init_type,
                        "Phase": phase,
                        "Category": "Garbage Padding",
                        "Token_ID": t_id,
                        "Token_String": clean_token,
                        "Survival_Count": count,
                        "Survival_Pct": round(pct, 2),
                    }
                )

        log_both(f"\n[*] Text report successfully saved to: {text_output_path}")

    # Write the CSV data
    if csv_data:
        df_export = pd.DataFrame(csv_data)
        df_export.to_csv(csv_output_path, index=False, encoding="utf-8")
        print(f"[*] CSV token data successfully saved to: {csv_output_path}")


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

if __name__ == "__main__":
    find_saturation_point(ENRICHED_FILE)
    # analyze_init_divergence(ENRICHED_FILE)
    isolate_core_payload(ENRICHED_FILE, min_tokens=27, max_tokens=57)

    # for tk_len in range(20, 47):
    #    token_provenance_analysis(ENRICHED_FILE, target_length=tk_len)
    # token_provenance_analysis(ENRICHED_FILE, target_length=38)

    # anchor_survival_analysis(ENRICHED_FILE, min_len=20, max_len=46)
