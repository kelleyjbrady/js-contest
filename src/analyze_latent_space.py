import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ==========================================
# 1. SETUP & DATA INGESTION
# ==========================================
ACTIVATIONS_DIR = "/app/data/activations/"
OUTPUT_DIR = "/app/data/analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all_activations() -> pd.DataFrame:
    """Finds and concatenates all Parquet files in the activations directory."""
    print(f"[*] Scanning {ACTIVATIONS_DIR} for Parquet files...")
    files = glob.glob(os.path.join(ACTIVATIONS_DIR, "*.parquet"))

    if not files:
        raise FileNotFoundError(
            "[!] No Parquet files found. Run the extraction script first."
        )

    df_list = [pd.read_parquet(f) for f in files]
    master_df = pd.concat(df_list, ignore_index=True)

    print(
        f"[+] Loaded {len(files)} files containing {len(master_df)} total activation vectors."
    )
    return master_df


# ==========================================
# 2. BASELINE CALCULATION (TODAY'S GOAL)
# ==========================================
def establish_benign_baseline(df: pd.DataFrame) -> np.ndarray:
    """Calculates and saves the mu_benign centroid."""
    df_benign = df[df["is_suspicious"] == False]

    if df_benign.empty:
        raise ValueError("[!] No benign prompts found in dataset.")

    # Stack the lists of floats into a proper 2D numpy array (N_samples, Hidden_Dim)
    X_benign = np.vstack(df_benign["activation_vector"].values)

    # Calculate the center of mass
    mu_benign = np.mean(X_benign, axis=0)

    # Save the vector to disk for future PyTorch injection
    baseline_path = os.path.join(OUTPUT_DIR, "mu_benign.npy")
    np.save(baseline_path, mu_benign)
    print(
        f"[+] Benign centroid (mu_benign) calculated from {len(df_benign)} samples and saved to disk."
    )

    return X_benign, mu_benign


def plot_benign_manifold(X_benign: np.ndarray, df_benign: pd.DataFrame):
    """Exploratory PCA to see how the model organizes normal concepts."""
    print("[*] Running exploratory PCA on the benign manifold...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_benign)

    plt.figure(figsize=(10, 8))

    # Color code by source to see if templated vs HF prompts cluster differently
    sources = df_benign["source"].unique()
    colors = plt.cm.get_cmap("tab10", len(sources))

    for idx, source in enumerate(sources):
        mask = df_benign["source"] == source
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1], alpha=0.7, label=source, color=colors(idx)
        )

    plt.title(
        f"Benign Latent Manifold (Explained Variance: {sum(pca.explained_variance_ratio_):.2%})"
    )
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(OUTPUT_DIR, "benign_manifold_pca.png")
    plt.savefig(plot_path)
    print(f"[+] Benign manifold plot saved to {plot_path}")


# ==========================================
# 3. ANTHROPIC EXTRACTION (TOMORROW'S GOAL)
# ==========================================
def extract_steering_vector(df: pd.DataFrame, mu_benign: np.ndarray):
    """Calculates the mean-shift vector if suspicious data is present."""
    df_susp = df[df["is_suspicious"] == True]

    if df_susp.empty:
        print(
            "[-] No suspicious data found yet. Skipping full steering vector extraction."
        )
        return

    print(f"[*] Extracting steering vector using {len(df_susp)} suspicious samples...")
    X_susp = np.vstack(df_susp["activation_vector"].values)
    mu_susp = np.mean(X_susp, axis=0)

    # The core Anthropic calculation
    mean_shift_vector = mu_susp - mu_benign
    steering_vector = mean_shift_vector / np.linalg.norm(mean_shift_vector)

    vector_path = os.path.join(OUTPUT_DIR, "deception_steering_vector.npy")
    np.save(vector_path, steering_vector)
    print(f"[+] Deception steering vector saved to {vector_path}")

    # Joint Space PCA Verification
    X_all = np.vstack(
        [
            np.vstack(df[df["is_suspicious"] == False]["activation_vector"].values),
            X_susp,
        ]
    )
    X_centered = X_all - mu_benign

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)

    plt.figure(figsize=(10, 8))
    labels = df["is_suspicious"].values
    plt.scatter(
        X_pca[labels == False, 0],
        X_pca[labels == False, 1],
        alpha=0.5,
        label="Benign",
        color="blue",
    )
    plt.scatter(
        X_pca[labels == True, 0],
        X_pca[labels == True, 1],
        alpha=0.5,
        label="Duplicitous",
        color="red",
    )

    plt.title("Contrastive PCA: Benign vs Duplicitous Activations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "joint_space_pca.png"))
    print("[+] Joint space contrastive plot saved.")


if __name__ == "__main__":
    # 1. Load whatever data we currently have
    master_df = load_all_activations()

    # 2. Lock in the baseline
    X_benign, mu_benign = establish_benign_baseline(master_df)

    # 3. Explore the safe space (generates a plot today)
    plot_benign_manifold(X_benign, master_df[master_df["is_suspicious"] == False])

    # 4. Attempt full extraction (will gracefully skip today, will run tomorrow)
    extract_steering_vector(master_df, mu_benign)
