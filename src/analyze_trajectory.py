import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ACTIVATIONS_DIR = (
    "/app/data/activations/combined_parquet/"  # Update to your daily run path
)
CLEAN_IDS_FILE = "/app/data/clean_prompt_ids.csv"
OUTPUT_DIR = "/app/data/"


def load_clean_data(file_path, clean_ids):
    """Loads a parquet file and filters out SIMCA outliers."""
    df = pd.read_parquet(file_path)
    # Filter only the IDs that survived the SIMCA verification
    clean_df = df[df["prompt_id"].isin(clean_ids)]
    return clean_df


def compute_gram_schmidt(df):
    """Executes vector rejection to isolate the pure trigger."""
    # 1. Calculate Centroids
    centroids = {}
    for cat in ["benign", "refusal", "deception", "meta_probe"]:
        cat_df = df[df["category"] == cat]
        if cat_df.empty:
            return None, None, None, None
        centroids[cat] = np.mean(np.stack(cat_df["activation_vector"].values), axis=0)

    # 2. Center on Benign Baseline
    v_refusal = centroids["refusal"] - centroids["benign"]
    v_deception = centroids["deception"] - centroids["benign"]
    v_raw_trigger = centroids["meta_probe"] - centroids["benign"]

    # 3. Orthogonalize
    # u1: Pure Refusal Direction
    u1 = v_refusal / np.linalg.norm(v_refusal)

    # u2: Pure Deception (Orthogonal to Refusal)
    v_deception_ortho = v_deception - np.dot(v_deception, u1) * u1
    u2 = v_deception_ortho / np.linalg.norm(v_deception_ortho)

    # v_pure_trigger: Scrub both Refusal and Deception
    proj_u1 = np.dot(v_raw_trigger, u1) * u1
    proj_u2 = np.dot(v_raw_trigger, u2) * u2

    v_pure_trigger = v_raw_trigger - proj_u1 - proj_u2

    return u1, u2, v_pure_trigger, centroids["benign"]


def analyze_and_plot():
    parquet_files = sorted(
        glob.glob(os.path.join(ACTIVATIONS_DIR, "model_layers_*.parquet"))
    )

    if not parquet_files:
        print("[!] No Parquet files found. Check your ACTIVATIONS_DIR path.")
        return

    if not os.path.exists(CLEAN_IDS_FILE):
        print("[!] clean_prompt_ids.csv not found. Run verify_manifolds.py first.")
        return

    clean_ids = pd.read_csv(CLEAN_IDS_FILE)["prompt_id"].tolist()
    print(f"[*] Loaded {len(clean_ids)} SIMCA-verified clean prompt IDs.")

    layers = []
    trigger_magnitudes = []
    deepest_layer_data = None
    deepest_layer_num = -1

    print("\n[*] Processing Layers...")
    for file_path in parquet_files:
        layer_num = int(
            os.path.basename(file_path).split("_")[-1].replace(".parquet", "")
        )
        df = load_clean_data(file_path, clean_ids)

        u1, u2, v_pure_trigger, mu_benign = compute_gram_schmidt(df)

        if v_pure_trigger is None:
            print(f"  [!] Missing category data in Layer {layer_num}. Skipping.")
            continue

        # Record the raw magnitude of the pure trigger vector
        mag = np.linalg.norm(v_pure_trigger)
        layers.append(layer_num)
        trigger_magnitudes.append(mag)
        print(f"  -> Layer {layer_num:02d} | Pure Trigger Magnitude: {mag:.4f}")

        # Save the deepest layer for the scatter plot
        if layer_num > deepest_layer_num:
            deepest_layer_num = layer_num
            deepest_layer_data = (df, u2, v_pure_trigger, mu_benign)

    # ==========================================
    # PLOT 1: The Activation Trajectory
    # ==========================================
    plt.figure(figsize=(10, 5))
    plt.plot(
        layers,
        trigger_magnitudes,
        marker="o",
        linestyle="-",
        color="#1b9e77",
        linewidth=2.5,
        markersize=8,
    )
    plt.fill_between(layers, 0, trigger_magnitudes, color="#1b9e77", alpha=0.1)

    plt.title("Sleeper Agent Trajectory (Gram-Schmidt Purified)", fontsize=14, pad=15)
    plt.xlabel("Transformer Layer", fontsize=12)
    plt.ylabel(r"Magnitude of $\vec{v}_{pure\_trigger}$", fontsize=12)
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    traj_path = os.path.join(OUTPUT_DIR, "trajectory_magnitude.png")
    plt.savefig(traj_path, dpi=200)
    print(f"\n[+] Trajectory plot saved to: {traj_path}")

    # ==========================================
    # PLOT 2: The 2D Orthogonal Projection
    # ==========================================
    if deepest_layer_data:
        df, u2, v_pure_trigger, mu_benign = deepest_layer_data
        u_trigger_norm = v_pure_trigger / np.linalg.norm(v_pure_trigger)

        # Prepare data for scatter
        X_matrix = np.stack(df["activation_vector"].values)
        categories = df["category"].values

        # Center and project
        X_centered = X_matrix - mu_benign
        x_coords = np.dot(X_centered, u_trigger_norm)  # X-Axis: Pure Trigger
        y_coords = np.dot(X_centered, u2)  # Y-Axis: Pure Deception

        plot_df = pd.DataFrame(
            {
                "Pure Trigger Axis": x_coords,
                "Pure Deception Axis": y_coords,
                "Category": categories,
            }
        )

        plt.figure(figsize=(10, 8))
        palette = {
            "benign": "#a8b6c1",
            "refusal": "#e66100",
            "deception": "#5d3a9b",
            "meta_probe": "#1b9e77",
        }

        sns.scatterplot(
            data=plot_df,
            x="Pure Trigger Axis",
            y="Pure Deception Axis",
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
            f"Orthogonalized Cognitive Manifolds (Layer {deepest_layer_num})",
            fontsize=14,
            pad=15,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        scatter_path = os.path.join(
            OUTPUT_DIR, f"orthogonal_projection_L{deepest_layer_num}.png"
        )
        plt.savefig(scatter_path, dpi=200)
        print(f"[+] Orthogonal projection saved to: {scatter_path}")


if __name__ == "__main__":
    analyze_and_plot()
