import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import plotly.express as px

TARGET_FOLDER = "/app/telemetry_data/"
batches = {
    i: os.path.join(TARGET_FOLDER, i)
    for i in os.listdir(TARGET_FOLDER)
    if "hpo" not in i
}


def load_telemetry(data_dir):
    """Parses all JSONL logs in the directory into a single DataFrame."""
    print(f"[*] Loading telemetry data from {data_dir}...")
    records = []
    file_pattern = os.path.join(data_dir, "*.jsonl")

    for file_path in glob.glob(file_pattern):
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("No data found! Check your directory path.")

    print(f"[+] Successfully loaded {len(df)} telemetry snapshots.")
    return df


def plot_learning_curves(df, batch_name):
    """Phase 1: Plots Score vs. Step for each sequence length."""
    plt.figure(figsize=(12, 8))
    df = df.copy()
    if "joint_score" in df.columns:
        df["score"] = df["joint_score"]
    sns.lineplot(
        data=df, x="step", y="score", hue="sequence_length", palette="viridis", lw=2
    )

    plt.title(
        "GCG Optimization Trajectories by Sequence Length",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Optimization Step", fontsize=12)
    plt.ylabel("Cosine Similarity Score", fontsize=12)
    plt.legend(title="Sequence Length", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"trajectory_analysis_{batch_name}.png", dpi=300)
    print("[+] Saved 'trajectory_analysis.png'")


def plot_saturation_curve(batch_dfs):
    """Phase 2: Plots Maximum Score vs. Sequence Length to find the asymptote."""
    # Group by sequence length and find the max score
    plot_df = []
    names = ""
    joint_score_exists = []
    for batch_name in batch_dfs:
        _plot_df = batch_dfs[batch_name].copy()
        _plot_df["batch_name"] = batch_name
        if "joint_score" in _plot_df.columns:
            _plot_df["score"] = _plot_df["joint_score"]
        plot_df.append(_plot_df)
        names = names + "_" + batch_name
        joint_score_exists.append("joint_score" in _plot_df.columns)

    plot_df = pd.concat(plot_df)

    max_scores = (
        plot_df.groupby(["batch_name", "sequence_length"])["score"].max().reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=max_scores,
        x="sequence_length",
        y="score",
        marker="o",
        # color="crimson",
        hue="batch_name",
        lw=2,
        markersize=8,
    )

    plt.title(
        "Token Saturation Curve (Max Score vs. Length)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Trigger Sequence Length", fontsize=12)
    plt.ylabel("Maximum Cosine Similarity Achieved", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    names_len = len(names)
    plt.savefig(f"saturation_curve_{names_len}_runs.png", dpi=300)
    print("[+] Saved 'saturation_curve.png'")

    use_plotly = True
    if use_plotly:
        fig = px.line(
            data_frame=max_scores, x="sequence_length", y="score", color="batch_name"
        )
        fig.show()
    return max_scores


def extract_lexical_forensics(df, batch_name):
    """Phase 3: Extracts the winning strings and raw tokens for each length."""
    print("\n" + "=" * 60)
    print(f"PHASE 3: LEXICAL FORENSICS {batch_name} (TOP CANDIDATES)")
    print("=" * 60)
    df = df.copy()
    if "joint_score" in df.columns:
        df["score"] = df["joint_score"]
    # Get the row with the max score for each sequence length
    idx = df.groupby("sequence_length")["score"].idxmax()
    best_candidates = df.loc[idx].sort_values(by="score", ascending=False)

    for _, row in best_candidates.iterrows():
        length = int(row["sequence_length"])
        score = float(row["score"])
        text = row["decoded_string"].replace("\n", "\\n")  # Escape newlines for display

        print(f"\n[ Length: {length:02d} | Score: {score:.4f} ]")
        print(f"Trigger String: '{text}'")
        # Optional: Print raw token IDs if you need to debug tokenizer behavior
        # print(f"Raw IDs: {row['token_ids']}")


if __name__ == "__main__":
    # Ensure dependencies are installed
    # pip install pandas matplotlib seaborn

    # Set plotting style
    sns.set_theme(style="whitegrid")

    data_directory = TARGET_FOLDER  # Update if you saved the logs elsewhere

    # try:
    batch_dfs = {i: load_telemetry(batches[i]) for i in batches}
    for batch_name in batch_dfs:
        df = batch_dfs[batch_name]
        plot_learning_curves(df, batch_name)

        extract_lexical_forensics(df, batch_name)
    max_scores_df = plot_saturation_curve(batch_dfs)
    print("\n[*] Analysis complete. Check the generated .png files.")
    # except Exception as e:
    #    print(f"\n[!] Analysis failed: {e}")
