import os
import re
import glob
import csv
from collections import defaultdict


def parse_logs_to_csv(log_dir=".", top_k=20, output_csv="pre_gcg_candidates.csv"):
    """
    Parses Logit Lens / Input Projection logs and generates a single,
    stacked CSV file comparing META_PROBE and TRIGGER_EXEC across all layers.
    """
    print(f"[*] Scanning for candidate logs in {log_dir}...")

    # Regex to catch the exact header format
    header_pattern = re.compile(r"🎯 ISOLATED '([^']+)' CANDIDATES \| LAYER (\d+) 🎯")

    # Regex to catch the rank, score, and token
    token_pattern = re.compile(
        r"^\s*(\d+)\.\s+Score:\s+([0-9\.]+)\s+\|\s+Token:\s+(.*)$"
    )

    # Data structure: parsed_data[layer][target] = [(score, token), ...]
    parsed_data = defaultdict(lambda: defaultdict(list))

    # Parse all .txt files in the specified directory
    files = glob.glob(os.path.join(log_dir, "*.txt"))
    if not files:
        print(f"[!] No .txt files found in directory: {log_dir}")
        return

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            current_target = None
            current_layer = None

            for line in f:
                header_match = header_pattern.search(line)
                if header_match:
                    current_target = header_match.group(1).upper()
                    current_layer = int(header_match.group(2))
                    continue

                if current_target and current_layer:
                    token_match = token_pattern.search(line)
                    if token_match:
                        rank = int(token_match.group(1))
                        score = float(token_match.group(2))
                        token = token_match.group(3).strip()

                        # Only grab up to top_k
                        if rank <= top_k:
                            parsed_data[current_layer][current_target].append(
                                (score, token)
                            )

                    # Stop tracking if we hit the bottom boundary
                    elif (
                        "==================================================" in line
                        and len(parsed_data[current_layer][current_target]) > 0
                    ):
                        current_target = None
                        current_layer = None

    # Flatten the parsed data into rows for the CSV
    csv_rows = []
    layers = sorted(parsed_data.keys())

    for layer in layers:
        probe_list = parsed_data[layer].get("META_PROBE", [])
        exec_list = parsed_data[layer].get("TRIGGER_EXEC", [])

        # Iterate up to the max length of either list to ensure side-by-side alignment
        max_len = max(len(probe_list), len(exec_list))

        for i in range(max_len):
            row = {
                "Layer": layer,
                "Rank": i + 1,
            }

            # Populate META_PROBE columns
            if i < len(probe_list):
                row["META_PROBE_Token"] = probe_list[i][1]
                row["META_PROBE_Score"] = probe_list[i][0]
            else:
                row["META_PROBE_Token"] = ""
                row["META_PROBE_Score"] = ""

            # Populate TRIGGER_EXEC columns
            if i < len(exec_list):
                row["TRIGGER_EXEC_Token"] = exec_list[i][1]
                row["TRIGGER_EXEC_Score"] = exec_list[i][0]
            else:
                row["TRIGGER_EXEC_Token"] = ""
                row["TRIGGER_EXEC_Score"] = ""

            csv_rows.append(row)

    # Write the flattened data to CSV
    print(f"[*] Writing stacked data to {output_csv}...")
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Layer",
            "Rank",
            "META_PROBE_Token",
            "META_PROBE_Score",
            "TRIGGER_EXEC_Token",
            "TRIGGER_EXEC_Score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[*] Success! Candidates safely exported to CSV.")


if __name__ == "__main__":
    # Point this to the directory where your decode_trigger.py text logs are saved
    # Make sure to update the path if your structure differs!
    parse_logs_to_csv(
        log_dir="/app/data/activations/combined_parquet/20260330_232054_batched/decode/",
        top_k=20,
        output_csv="pre_gcg_candidates.csv",
    )
