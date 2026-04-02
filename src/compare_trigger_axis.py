import os
import re
import csv
import torch
import torch.nn.functional as F
import glob
from itertools import combinations
import pandas as pd
from typing import Literal


def calculate_cosine_similarity(path_v1, path_v2, layer=None):
    """Calculates Cosine Similarity using the safe 1D dot product method."""
    try:
        # Load and flatten to 1D
        t1 = torch.load(path_v1, map_location="cpu", weights_only=True).flatten()
        t2 = torch.load(path_v2, map_location="cpu", weights_only=True).flatten()

        if t1.shape != t2.shape:
            return None, f"Shape mismatch: {t1.shape} vs {t2.shape}"

        # Normalize to unit vectors
        t1_norm = F.normalize(t1, p=2, dim=0)
        t2_norm = F.normalize(t2, p=2, dim=0)

        # Dot product = Cosine Similarity for unit vectors
        sim = torch.dot(t1_norm, t2_norm).item()
        if layer is None:
            layer = ""
        else:
            layer = f"Layer {layer}"
        print(f"Cosine Similarity {layer}: {sim}")
        return sim, None

    except Exception as e:
        return None, f"File loading error: {str(e)}"


def compare_all_versions(
    target_pattern, output_csv_path, mode: Literal["folder", "full"]
):
    print("\n==================================================")
    print("📐 BULK AXIS HOMOGENEITY EVALUATION 📐")
    print("==================================================")

    # 1. Discover and group all .pt files by their base name and version
    files = glob.glob(target_pattern)
    file_groups = {}

    # Regex: Capture everything before "_vX.pt" as the base name, and X as the version

    for f in files:
        base = os.path.basename(f)
        layer = base.split("_")[-1].replace(".pt", "")
        layer_type = base.replace(f"_{layer}.pt", "")
        version_dir = os.path.basename(os.path.dirname(os.path.dirname(f)))
        # finish below here
        if layer_type not in file_groups:
            file_groups[layer_type] = {}
        if layer not in file_groups[layer_type]:
            file_groups[layer_type][layer] = {}
        if version_dir not in file_groups[layer_type][layer]:
            file_groups[layer_type][layer][version_dir] = {}
        file_groups[layer_type][layer][version_dir] = f

    if not file_groups:
        print(f"[-] No files matching the '*_vX.pt' pattern found.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    results_data = []

    # 2. Iterate through groups and compare v1 vs v2
    for layer_type in file_groups:
        for layer in file_groups[layer_type]:
            versions = [i for i in file_groups[layer_type][layer]]
            combos = combinations(versions, 2)
            for v1, v2 in combos:
                path_v1 = file_groups[layer_type][layer][v1]
                path_v2 = file_groups[layer_type][layer][v2]

                # print(f"\n[*] Evaluating: {base_name}")

                cos_sim, err = calculate_cosine_similarity(path_v1, path_v2)

                print(f"    Cosine Similarity: {cos_sim:.4f}")

                if cos_sim > 0.85:
                    alignment = "EXTREMELY HIGH ALIGNMENT. Geometric feature is stable."
                    print(f"    [!] Result: {alignment}")
                elif cos_sim > 0.50:
                    alignment = "MODERATE ALIGNMENT. Stratification cleaned up noise."
                    print(f"    [+] Result: {alignment}")
                else:
                    alignment = "LOW ALIGNMENT. The new axis is entirely distinct."
                    print(f"    [-] Result: {alignment}")

                results_data.append(
                    [
                        layer_type,
                        layer,
                        v1,
                        v2,
                        os.path.basename(path_v1),
                        os.path.basename(path_v2),
                        f"{cos_sim:.4f}",
                        alignment,
                    ]
                )
    df = pd.DataFrame(results_data)

    # 3. Write all results to the CSV
    # with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
    #    writer = csv.writer(csvfile)
    #    writer.writerow(
    #        [
    #            "layer_type",
    #            "layer_numer",
    #            "version_1",
    #            "version_2",
    #            "path_1",
    #            "path_2Cosine_Similarity",
    #            "Result",
    #        ]
    #    )
    #    writer.writerows(results_data)
    #
    df.to_csv(output_csv_path)
    print(f"\n[+] Results successfully saved to: {output_csv_path}")
    print("==================================================\n")


if __name__ == "__main__":
    # Ensure these point to your specific directories
    TARGET_PATTERN = (
        "/app/data/activations/combined_parquet/20260330_232054_batched/decode/*.pt"
    )
    # OUTPUT_CSV = "/app/data/activations/combined_parquet/axis_homogeneity_results_20260330_232054.csv"
    # compare_all_versions(TARGET_PATTERN, OUTPUT_CSV, mode="folder")
    calculate_cosine_similarity(
        "data/activations/combined_parquet/20260330_232054_batched/decode/trigger_meta_probe_probe_layer_15.pt",
        "/app/data/activations/combined_parquet/20260330_232054_batched/decode/trigger_trigger_exec_exec_layer_15.pt",
        layer=15,
    )
    calculate_cosine_similarity(
        "data/activations/combined_parquet/20260330_232054_batched/decode/trigger_meta_probe_probe_layer_20.pt",
        "/app/data/activations/combined_parquet/20260330_232054_batched/decode/trigger_trigger_exec_exec_layer_20.pt",
        layer=20,
    )
    calculate_cosine_similarity(
        "data/activations/combined_parquet/20260330_232054_batched/decode/trigger_meta_probe_probe_layer_35.pt",
        "/app/data/activations/combined_parquet/20260330_232054_batched/decode/trigger_trigger_exec_exec_layer_35.pt",
        layer=35,
    )
    calculate_cosine_similarity(
        "data/activations/combined_parquet/20260330_232054_batched/decode/trigger_meta_probe_probe_layer_55.pt",
        "/app/data/activations/combined_parquet/20260330_232054_batched/decode/trigger_trigger_exec_exec_layer_55.pt",
        layer=55,
    )
