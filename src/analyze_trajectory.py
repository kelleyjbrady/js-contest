import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTIVATIONS_DIR = "/app/data/activations/"


def load_and_analyze():
    parquet_files = sorted(
        glob.glob(os.path.join(ACTIVATIONS_DIR, "model_layers_*.parquet"))
    )

    if not parquet_files:
        print("[!] No Parquet files found. Run the extraction script first.")
        return

    distances = []
    labels = []

    print("[*] Computing layer-wise centroids...")
    print("-" * 40)

    for file_path in parquet_files:
        # Extract layer number from filename (e.g., model_layers_5.parquet -> 5)
        filename = os.path.basename(file_path)
        layer_num = int(filename.split("_")[-1].replace(".parquet", ""))
        labels.append(f"L{layer_num}")

        # Load Parquet into a DataFrame
        df = pd.read_parquet(file_path)

        benign_df = df[df["category"] == "benign"]
        decep_df = df[df["category"] == "deceptive"]

        if benign_df.empty or decep_df.empty:
            print(f"  [!] Missing categorical data for Layer {layer_num}")
            distances.append(0)
            continue

        # Reconstruct the matrices (N, 7168)
        benign_stack = np.vstack(benign_df["activation_vector"].values)
        decep_stack = np.vstack(decep_df["activation_vector"].values)

        # Compute means (Centroids) for both classes along the batch axis
        mu_benign = np.mean(benign_stack, axis=0)
        mu_decep = np.mean(decep_stack, axis=0)

        # Calculate L2 Distance between the centroids
        l2_dist = np.linalg.norm(mu_decep - mu_benign)
        distances.append(l2_dist)

        print(f"  -> Layer {layer_num:02} | Deception Divergence (L2): {l2_dist:.4f}")

    plot_trajectory(labels, distances)


def plot_trajectory(labels, distances):

    plt.figure(figsize=(12, 6))
    plt.plot(
        labels, distances, marker="o", linestyle="-", color="#d62728", linewidth=2.5
    )
    plt.fill_between(labels, 0, distances, color="#d62728", alpha=0.1)

    plt.title(
        "Latent Space Divergence: Deceptive vs. Benign Trajectories",
        fontsize=14,
        pad=15,
    )
    plt.xlabel("Network Depth (Transformer Layer)", fontsize=12)
    plt.ylabel(r"L2 Distance ($||\Delta v||_2$)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    save_path = "deception_trajectory.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n[+] Trajectory plotted! Check '{save_path}' to find the spike.")


if __name__ == "__main__":
    load_and_analyze()
