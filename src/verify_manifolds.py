import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats as stats

ACTIVATIONS_DIR = "/app/data/activations/combined_parquet/20260318_230424_batched/"  # Adjust to your path
OUTPUT_CLEAN_IDS = ACTIVATIONS_DIR + "clean_prompt_ids.csv"


def compute_simca_limits(X, variance_threshold=0.85):
    """Builds a local PCA model and calculates T2 and Q residuals."""
    n_samples, n_features = X.shape

    # We must restrict max components to n_samples for small batches
    max_comp = min(n_samples - 1, int(n_features * 0.1))

    # 1. Fit Local PCA
    pca = PCA(n_components=max_comp)
    pca.fit(X)

    # Find how many components hit our 85% variance target
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    k = max(1, k)  # Ensure at least 1 component

    print(
        f"    -> PCA fitted: Retained {k} components capturing {cumulative_variance[k - 1] * 100:.1f}% variance."
    )

    # Refit with exact k components
    pca = PCA(n_components=k)
    scores = pca.fit_transform(X)  # T matrix

    # 2. Calculate Hotelling's T^2 (In-model distance)
    # T^2 = sum( (scores^2) / eigenvalues )
    eigenvalues = pca.explained_variance_
    T2 = np.sum((scores**2) / eigenvalues, axis=1)

    # 3. Calculate Q-Residuals (Orthogonal out-of-model distance)
    X_reconstructed = pca.inverse_transform(scores)
    Q = np.sum((X - X_reconstructed) ** 2, axis=1)

    return T2, Q


def run_simca_verification():
    # Load the latest Parquet file (e.g., layer 55)
    # We use the deep layer because that is where the final decision is formed
    parquet_files = glob.glob(os.path.join(ACTIVATIONS_DIR, "*55.parquet"))
    if not parquet_files:
        print("[!] No layer 55 Parquet files found. Run extraction first.")
        return

    # latest_file = max(parquet_files, key=os.path.getctime)
    print(f"[*] Loading deep-layer tensors from: {parquet_files}")
    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    print(f"[*] Loaded {len(df)} deep-layer tensors")
    clean_ids = []

    # Iterate through our 4 contrastive categories
    for category in df["category"].unique():
        print(f"\n[*] Evaluating Manifold: {category}")
        cat_df = df[df["category"] == category].copy()

        if len(cat_df) < 5:
            print("    [!] Not enough samples to build a PCA envelope. Skipping.")
            clean_ids.extend(cat_df["prompt_id"].tolist())
            continue

        # Stack lists into a 2D NumPy array
        X = np.stack(cat_df["activation_vector"].values)

        # Center the data
        X_mean = np.mean(X, axis=0)
        X_centered = X - X_mean

        # Calculate distance metrics
        T2, Q = compute_simca_limits(X_centered)

        # Set 95% Confidence Thresholds using empirical percentiles
        # (Standard chi-square is too aggressive for n=75 and D=7168)
        T2_limit = np.percentile(T2, 95)
        Q_limit = np.percentile(Q, 95)

        # Filter outliers
        is_clean = (T2 <= T2_limit) & (Q <= Q_limit)
        cat_df["is_clean"] = is_clean

        retained = cat_df[cat_df["is_clean"]]
        dropped = len(cat_df) - len(retained)

        print(f"    -> T^2 Limit: {T2_limit:.2f} | Q Limit: {Q_limit:.2f}")
        print(f"    -> Dropped {dropped} extreme outliers out of {len(cat_df)}.")

        clean_ids.extend(retained["prompt_id"].tolist())

    # Save the verified, clean prompt IDs
    clean_df = pd.DataFrame({"prompt_id": clean_ids})
    clean_df.to_csv(OUTPUT_CLEAN_IDS, index=False)
    print(
        f"\n[+] SIMCA Verification complete. Saved {len(clean_ids)} clean IDs to {OUTPUT_CLEAN_IDS}"
    )


if __name__ == "__main__":
    run_simca_verification()
