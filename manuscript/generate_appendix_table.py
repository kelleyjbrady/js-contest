import pandas as pd


def generate_appendix_markdown(
    csv_path="pre_gcg_candidates.csv", output_md="appendix_c.md"
):
    print(f"[*] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Format the scores to 4 decimal places so the markdown table stays perfectly aligned
    # We use pd.notnull to handle any empty rows gracefully
    df["META_PROBE_Score"] = df["META_PROBE_Score"].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else ""
    )
    df["TRIGGER_EXEC_Score"] = df["TRIGGER_EXEC_Score"].apply(
        lambda x: f"{x:.4f}" if pd.notnull(x) else ""
    )

    # Convert the DataFrame directly to a Markdown table
    # We disable the index so it doesn't print the Pandas row numbers
    md_table = df.to_markdown(index=False)

    # Construct the full Appendix section text
    appendix_text = f"""## Appendix C: Pre-Optimization Topological Baselines (Logit Lens / Input Projections)

The following table provides the complete empirical data referenced in Section 8. The orthogonalized target states ($\\vec{{v}}_{{probe}}$ and $\\vec{{v}}_{{exec}}$) were projected directly onto the model's vocabulary matrices prior to any GCG optimization. 

Layers 15 and 20 are projected onto the input embedding matrix ($W_E$) to identify the highest-magnitude constituent input fragments. Layers 35 and 55 are projected onto the output unembedding matrix (`lm_head`) to map the immediate conceptual output the model is biased toward while occupying the target state.

{md_table}
"""

    # Write it directly to a markdown file
    print(f"[*] Writing manuscript appendix to {output_md}...")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(appendix_text)

    print("[*] Success! Appendix generated.")


if __name__ == "__main__":
    # Ensure you have 'tabulate' installed (pip install tabulate) as it is required by pd.to_markdown()
    generate_appendix_markdown(csv_path="/app/pre_gcg_candidates.csv")
