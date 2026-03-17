import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "/app/data/prompt_corpus.duckdb"


def plot_distributions():
    print("[*] Connecting to database and fetching unfiltered scores...")

    # 1. Fetch the raw, unfiltered data
    with duckdb.connect(DB_PATH) as conn:
        df = conn.execute("""
            SELECT 
                source, 
                duplicity_nature, 
                domain_context, 
                generation_style,
                eval_safety, 
                eval_deception, 
                eval_meta_probe,
                eval_coherence
            FROM prompts 
            WHERE eval_status = 'graded' 
              AND source IS NOT NULL
              --and source = 'augmented_suspicious'
        """).df()

    if df.empty:
        print("[!] No graded data found. Run the evaluation script first.")
        return

    print(f"[*] Fetched {len(df)} prompts. Generating density plots...")

    # 2. Set up the visualization canvas
    # We use a 3-row grid to stack the three metrics on top of each other
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(20, 25))
    df["full_descr"] = (
        df["source"].astype(str) + "_\n" + df["duplicity_nature"].astype(str)
        # + "_"
        # + df["domain_context"].astype(str)
        # + "_"
        # + df["generation_style"].astype(str)
    )
    # Optional: ensure consistent ordering of the categories on the X-axis
    use_full_descr = True
    if use_full_descr:
        order = df["full_descr"].unique()
        x = "full_descr"
    else:
        order = [
            "augmented_benign",
            "augmented_hf_benign",
            "augmented_suspicious",
            "stylized_deception",
            "augmented_trigger",
        ]
        x = "source"

    # 3. Plot 1: Safety / Refusal Risk
    sns.violinplot(
        data=df,
        x=x,
        y="eval_safety",
        ax=axes[0],
        order=order,
        palette="Blues",
        cut=0,
        inner="quartile",
    )
    axes[0].set_title("Distribution of Safety/Refusal Risk Scores", fontsize=14, pad=10)
    axes[0].set_ylim(0, 11)
    axes[0].set_ylabel("Score (1-10)")
    axes[0].set_xlabel("")
    if use_full_descr:
        axes[0].tick_params("x", rotation=90)

    # 4. Plot 2: Deceptive Bifurcation
    sns.violinplot(
        data=df,
        x=x,
        y="eval_deception",
        ax=axes[1],
        order=order,
        palette="Oranges",
        cut=0,
        inner="quartile",
    )
    axes[1].set_title("Distribution of Deceptive Intent Scores", fontsize=14, pad=10)
    axes[1].set_ylim(0, 11)
    axes[1].set_ylabel("Score (1-10)")
    axes[1].set_xlabel("")
    if use_full_descr:
        axes[1].tick_params("x", rotation=90)

    # 5. Plot 3: Meta Trigger Probe
    sns.violinplot(
        data=df,
        x=x,
        y="eval_meta_probe",
        ax=axes[2],
        order=order,
        palette="Purples",
        cut=0,
        inner="quartile",
    )
    axes[2].set_title("Distribution of Meta Trigger Probe Scores", fontsize=14, pad=10)
    axes[2].set_ylim(0, 11)
    axes[2].set_ylabel("Score (1-10)")
    axes[2].set_xlabel("Generation Source")
    if use_full_descr:
        axes[2].tick_params("x", rotation=90)

    # 5. Plot 3: Meta Trigger Probe
    sns.violinplot(
        data=df,
        x=x,
        y="eval_coherence",
        ax=axes[3],
        order=order,
        palette="Greens",
        cut=0,
        inner="quartile",
    )
    axes[3].set_title("Distribution of Coherence Scores", fontsize=14, pad=10)
    axes[3].set_ylim(0, 11)
    axes[3].set_ylabel("Score (1-10)")
    axes[3].set_xlabel("Generation Source")
    if use_full_descr:
        axes[3].tick_params("x", rotation=90)

    OUTPUT_PLOT = f"/app/data/score_distributions_{x}.png"
    # 6. Save and close
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Distribution plot saved successfully to: {OUTPUT_PLOT}")


if __name__ == "__main__":
    plot_distributions()
