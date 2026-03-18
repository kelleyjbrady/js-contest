import os
import duckdb
import pandas as pd

DB_PATH = "/app/data/prompt_corpus.duckdb"
OUTPUT_BASE_DIR = "/app/data/dumped_corpus/"


def fetch_balanced_dataset(sample_size=650) -> dict:
    """Fetches a balanced tri-categorical dataset and returns pandas DataFrames."""
    dataset = {}

    # Lock acquired only for the exact duration of the read operations
    with duckdb.connect(DB_PATH) as conn:
        # 1. Benign Baseline (The Control Group)
        dataset["benign"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source
            FROM prompts 
            WHERE source IN ('augmented_benign', 'augmented_hf_benign')
              AND eval_safety <= 3 
              AND eval_deception <= 3 
              AND eval_meta_probe <= 2
              AND eval_coherence >= 7 
              AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()  # <-- Natively returns a Pandas DataFrame

        # 2. Refusal Vector (Standard Jailbreaks)
        dataset["refusal"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source
            FROM prompts 
            WHERE source = 'augmented_suspicious' 
              AND eval_safety >= 7 
              AND eval_deception <= 6 
              AND eval_meta_probe <= 3
              AND eval_coherence >= 7 
              AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 3. Deception Vector (Sleeper Agents)
        dataset["deception"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source
            FROM prompts 
            WHERE source = 'stylized_deception' 
              AND eval_deception >= 8 
              AND eval_safety <= 4 
              AND eval_meta_probe <= 3
              AND eval_coherence >= 7 
              AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 4. Target Lock / Panic Vector (Trigger Probing)
        dataset["meta_probe"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source 
            FROM prompts 
            WHERE source = 'augmented_trigger' 
              AND eval_meta_probe >= 8 
              --AND eval_safety <= 7 
              --AND eval_deception <= 5
              AND eval_coherence >= 7 
              AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

    # Concatenate the isolated dataframes for macro-analysis
    df_master = pd.concat(dataset.values(), ignore_index=True)

    # Calculate summary distributions (using double brackets for multi-column slice)
    # Grouping by source ensures we see the metadata breakdown per category
    break_down_src = False
    if break_down_src:
        groups = ["source", "duplicity_nature", "domain_context", "generation_style"]
    else:
        groups = ["source"]
    df_count = df_master.groupby(
        groups,
    )[["prompt_id"]].count()
    df_descr = df_master.groupby(
        groups,
    )["prompt_length_chars"].describe()

    # Add summaries to the dictionary
    dataset["summary_counts"] = df_count
    dataset["summary_lengths"] = df_descr

    return dataset


if __name__ == "__main__":
    res = fetch_balanced_dataset()

    # Safely create the output directory if it doesn't exist
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("\n[*] Exporting metadata distributions...")

    for key, df in res.items():
        # Keep the index for the summary DataFrames, drop it for the raw datasets
        write_index = key.startswith("summary")
        file_path = os.path.join(OUTPUT_BASE_DIR, f"db_dump_{key}.csv")

        df.to_csv(file_path, index=write_index)
        print(f"  [+] Saved {key} data to {file_path}")

    print("\n[+] Verification complete. Check the CSVs for domain/style skew.")
