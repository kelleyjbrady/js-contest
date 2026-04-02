import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal

# /app/data/activations/combined_parquet/20260330_232054_batched
BATCH_ID = "20260330_232054"
ACTIVATIONS_DIR = f"/app/data/activations/combined_parquet/{BATCH_ID}_batched/"
OUTPUT_DIR = ACTIVATIONS_DIR


def load_clean_data(file_path, clean_ids):
    """Loads a parquet file and filters out outliers based on verified IDs."""
    df = pd.read_parquet(file_path)
    clean_df = df[df["prompt_id"].isin(clean_ids)]
    return clean_df


def compute_qr_projection(df, target_class, forbidden_class_names):
    """
    Executes QR Decomposition to isolate the pure target, projecting it
    away from a specifically defined forbidden subspace.
    """
    # 1. Calculate Centroids
    centroids = {}
    unique_cats = df["category"].unique()

    for cat in unique_cats:
        cat_df = df[df["category"] == cat]
        if not cat_df.empty:
            centroids[cat] = np.mean(
                np.stack(cat_df["activation_vector"].values), axis=0
            )

    if "benign" not in centroids or target_class not in centroids:
        return None, None, None

    # 2. Center on Benign Baseline
    mu_benign = centroids["benign"]
    centered_vectors = {
        cat: centroids[cat] - mu_benign for cat in centroids if cat != "benign"
    }

    v_target_raw = centered_vectors[target_class]

    # 3. Construct the Forbidden Subspace Matrix (E)
    # Only include the explicitly requested forbidden classes that actually exist in the data
    valid_forbidden = [cat for cat in forbidden_class_names if cat in centered_vectors]

    if not valid_forbidden:
        print(f"  [!] No valid forbidden classes found to construct subspace E.")
        return None, None, None

    # Stack confounding vectors as columns: Shape (7168, N_forbidden)
    E = np.column_stack([centered_vectors[cat] for cat in valid_forbidden])

    # 4. QR Decomposition
    # Q provides a flawless orthonormal basis spanning the forbidden subspace
    Q, R = np.linalg.qr(E)

    # 5. Orthogonal Projection: v_pure = v_target - Q(Q^T v_target)
    projection_onto_forbidden = Q @ (Q.T @ v_target_raw)
    v_pure_trigger = v_target_raw - projection_onto_forbidden

    # Extract the primary axis of the forbidden space for 2D plotting (usually Refusal)
    # Ensure Q has at least 2 columns before grabbing the second one
    if Q.shape[1] > 1:
        u_forbidden_axis = Q[:, 1]  # Plots the orthogonalized 2nd forbidden class
    else:
        u_forbidden_axis = Q[:, 0]

    return v_pure_trigger, u_forbidden_axis, mu_benign


def analyze_and_plot(
    outlier_method: Literal["iso", "SIMCA"], target_class: str, forbidden_classes: list
):
    parquet_files = sorted(
        glob.glob(os.path.join(ACTIVATIONS_DIR, "model_layers_*.parquet"))
    )

    if not parquet_files:
        print("[!] No Parquet files found. Check your ACTIVATIONS_DIR path.")
        return

    if target_class == "meta_probe":
        clean_ids_opt = "probe"
    elif target_class == "trigger_exec":
        clean_ids_opt = "exec"

    id_files = sorted(
        glob.glob(
            os.path.join(
                ACTIVATIONS_DIR,
                f"clean_prompt_ids_{outlier_method}_{clean_ids_opt}_*.csv",
            )
        )
    )

    if not id_files:
        print(f"[!] Clean ID files for {outlier_method} not found.")
        return

    clean_ids = {
        int(os.path.basename(f).split("_")[-1].replace(".csv", "")): pd.read_csv(f)[
            "prompt_id"
        ].tolist()
        for f in id_files
    }

    layers = []
    trigger_magnitudes = []
    layer_plot_data = {}

    print(f"\n[*] Processing Target: [{target_class}] | Filter: {outlier_method}")
    print(f"[*] Forbidden Subspace: {forbidden_classes}")

    for file_path in parquet_files:
        layer_num = int(
            os.path.basename(file_path).split("_")[-1].replace(".parquet", "")
        )

        if layer_num not in clean_ids:
            continue

        ids = clean_ids[layer_num]
        df = load_clean_data(file_path, ids)
        allowed_classes = (
            [
                target_class,
            ]
            + forbidden_classes
            + [
                "benign",
            ]
        )
        df = df[df["category"].isin(allowed_classes)]

        v_pure_trigger, u_forbidden_axis, mu_benign = compute_qr_projection(
            df, target_class=target_class, forbidden_class_names=forbidden_classes
        )

        if v_pure_trigger is None:
            print(f"  [!] Missing categorical data in Layer {layer_num}. Skipping.")
            continue

        mag = np.linalg.norm(v_pure_trigger)
        layers.append(layer_num)
        trigger_magnitudes.append(mag)
        print(f"  -> Layer {layer_num:02d} | Purified Target Magnitude: {mag:.4f}")

        layer_plot_data[layer_num] = (df, u_forbidden_axis, v_pure_trigger, mu_benign)

    if not layers:
        return

    # ==========================================
    # PLOT 1: The Activation Trajectory
    # ==========================================
    plt.figure(figsize=(10, 5))
    plt.plot(
        layers,
        trigger_magnitudes,
        marker="o",
        linestyle="-",
        color="#1b9e77" if target_class == "meta_probe" else "#d95f02",
        linewidth=2.5,
        markersize=8,
    )
    plt.fill_between(
        layers,
        0,
        trigger_magnitudes,
        color="#1b9e77" if target_class == "meta_probe" else "#d95f02",
        alpha=0.1,
    )

    plt.title(
        f"Sleeper Agent Target Trajectory: {target_class.upper()} (QR Purified)",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel(r"Magnitude of $\vec{v}_{pure}$", fontsize=12)
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    traj_path = os.path.join(
        OUTPUT_DIR, f"trajectory_magnitude_{outlier_method}_{target_class}.png"
    )
    plt.savefig(traj_path, dpi=200)
    print(f"\n[+] Trajectory plot saved to: {traj_path}")
    plt.close()

    # ==========================================
    # PLOT 2: The 2D Orthogonal Projections
    # ==========================================
    print("[*] Generating Orthogonal Projections...")
    for layer_num, data in layer_plot_data.items():
        df, u_forbidden_axis, v_pure_trigger, mu_benign = data

        u_target_norm = v_pure_trigger / np.linalg.norm(v_pure_trigger)
        X_matrix = np.stack(df["activation_vector"].values)
        categories = df["category"].values

        X_centered = X_matrix - mu_benign
        x_coords = np.dot(X_centered, u_target_norm)
        y_coords = np.dot(X_centered, u_forbidden_axis)

        plot_df = pd.DataFrame(
            {
                "Purified Target Axis": x_coords,
                "Primary Forbidden Axis": y_coords,
                "Category": categories,
            }
        )

        plt.figure(figsize=(10, 8))

        palette = {
            "benign": "#a8b6c1",
            "refusal": "#e66100",
            "deception": "#5d3a9b",
            "gibberish": "#7570b3",
            "meta_probe": "#1b9e77",
            "trigger_exec": "#D22B2B",
        }

        sns.scatterplot(
            data=plot_df,
            x="Purified Target Axis",
            y="Primary Forbidden Axis",
            hue="Category",
            palette=palette,
            alpha=0.8,
            edgecolor="black",
            s=80,
        )

        plt.scatter(0, 0, color="black", marker="X", s=200, label="Benign Origin")
        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.axvline(0, color="black", linewidth=1, linestyle="--")

        plt.title(
            f"Orthogonalized Cognitive Manifolds (Layer {layer_num})\nTarget: {target_class.upper()} via QR Subspace Projection",
            fontsize=14,
            pad=15,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        scatter_path = os.path.join(
            OUTPUT_DIR,
            f"orthogonal_projection_QR_{outlier_method}_L{layer_num}_{target_class}.png",
        )
        plt.savefig(scatter_path, dpi=200)
        plt.close()


if __name__ == "__main__":
    # Define the exact asymmetric relationships from Section V of the paper
    extraction_targets = [
        {"target": "meta_probe", "forbidden": ["refusal", "deception"]},
        {
            "target": "trigger_exec",
            "forbidden": ["refusal", "deception", "meta_probe", "gibberish"],
        },
    ]

    for config in extraction_targets:
        analyze_and_plot(
            outlier_method="iso",
            target_class=config["target"],
            forbidden_classes=config["forbidden"],
        )
