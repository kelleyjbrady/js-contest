import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP
import plotly.express as px
import numpy as np

DB_PATH = "/app/data/prompt_corpus.duckdb"
FILTER_DATASET = True
if FILTER_DATASET:
    dataset = pd.read_csv("/app/full_acceptable_prompts__trigger_exec.csv")
else:
    dataset = None

with duckdb.connect(DB_PATH) as conn:
    df = conn.execute("""
        SELECT 
            prompt_id,
            source, 
            duplicity_nature, 
            domain_context, 
            generation_style,
            eval_safety, 
            eval_deception, 
            eval_meta_probe,
            eval_coherence, 
            eval_execution,
            eval_model,
            prompt_version
        FROM prompts 
        WHERE eval_status = 'graded' 
            AND source IS NOT NULL
            --and source = 'augmented_suspicious'
    """).df()
    if dataset is not None:
        df = df.loc[df["prompt_id"].isin(dataset["prompt_id"]), :]


def plot_distributions(df):
    print("[*] Connecting to database and fetching unfiltered scores...")

    # 1. Fetch the raw, unfiltered data

    if df.empty:
        print("[!] No graded data found. Run the evaluation script first.")
        return

    print(f"[*] Fetched {len(df)} prompts. Generating density plots...")

    # 2. Set up the visualization canvas
    # We use a 3-row grid to stack the three metrics on top of each other
    sns.set_theme(style="whitegrid")

    width = 28
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(width, 25))
    df["full_descr"] = (
        df["source"].astype(str)
        + "_\n"  # + df["duplicity_nature"].astype(str)
        # + "_"
        + df["domain_context"].astype(str)
        # + "_"
        # + df["generation_style"].astype(str)
        # + df["eval_model"].astype(str)
        # + df["prompt_version"].astype(str)
    )
    # Optional: ensure consistent ordering of the categories on the X-axis
    use_full_descr = False
    if use_full_descr:
        order = df["full_descr"].unique()
        x = "full_descr"
    else:
        order = [
            "augmented_benign",
            "augmented_hf_benign",
            "augmented_suspicious",
            "stylized_deception",
            "augmented_trigger_probe",
            "augmented_trigger_exec",
            "programatic_gibberish",
        ]
        x = "source"
        df.loc[df["source"] == "augmented_trigger", "source"] = (
            "augmented_trigger_probe"
        )
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

        # OUTPUT_PLOT = f"/app/data/score_distributions_{x}.png"
        ## 6. Save and close
        # plt.tight_layout()
        # plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
        # plt.close()

        # print(f"[+] Distribution plot saved successfully to: {OUTPUT_PLOT}")

    # 5. Plot 3: Meta Trigger Probe
    sns.violinplot(
        data=df,
        x=x,
        y="eval_execution",
        ax=axes[4],
        order=order,
        palette="Reds",
        cut=0,
        inner="quartile",
    )
    axes[4].set_title("Distribution of Trigger Execution Scores", fontsize=14, pad=10)
    axes[4].set_ylim(0, 11)
    axes[4].set_ylabel("Score (1-10)")
    axes[4].set_xlabel("Generation Source")
    if use_full_descr:
        axes[4].tick_params("x", rotation=90)
    if FILTER_DATASET:
        extra_str = "_filter"
    else:
        extra_str = ""
    OUTPUT_PLOT = f"/app/data/score_distributions_{x}{extra_str}.png"
    # 6. Save and close
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Distribution plot saved successfully to: {OUTPUT_PLOT}")


def plot_umap(df):
    umap_df = (
        df.dropna(
            subset=[
                "eval_safety",
                "eval_deception",
                "eval_meta_probe",
                "eval_coherence",
                "eval_execution",
            ]
        )
        .sample(frac=1)
        .reset_index(drop=True)
        .copy()
    )
    umap_mod = UMAP(n_neighbors=(int(np.ceil(len(df) * 0.20))))
    umap_df[["umap_x", "umap_y"]] = umap_mod.fit_transform(
        umap_df[
            [
                "eval_safety",
                "eval_deception",
                "eval_meta_probe",
                "eval_coherence",
                "eval_execution",
            ]
        ].astype(float)
    )
    fig = px.scatter(
        umap_df,
        x="umap_x",
        y="umap_y",
        hover_data=umap_df[
            [
                "source",
                "duplicity_nature",
                "domain_context",
                "generation_style",
            ]
        ],
    )

    fig.show(renderer="browser")


if __name__ == "__main__":
    plot_distributions(df)
    # plot_umap(df)
