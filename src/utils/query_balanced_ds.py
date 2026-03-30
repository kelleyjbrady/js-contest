import os
import duckdb
import pandas as pd
import glob
from typing import Literal

DB_PATH = "/app/data/prompt_corpus.duckdb"
OUTPUT_BASE_DIR = "/app/data/dumped_corpus/"


def fetch_balanced_dataset(
    sample_size=20000, mode: Literal["probe", "exec"] = "exec"
) -> dict:
    """Fetches a balanced tri-categorical dataset and returns pandas DataFrames."""
    dataset = {}

    # Lock acquired only for the exact duration of the read operations
    with duckdb.connect(DB_PATH) as conn:
        # 1. Benign Baseline (The Control Group)
        dataset["benign__probe"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source IN ('augmented_benign', 'augmented_hf_benign')
              AND eval_safety <= 3 
              AND eval_deception <= 3 
              AND eval_meta_probe <= 2
             -- AND eval_execution <= 2
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()  # <-- Natively returns a Pandas DataFrame

        dataset["benign__exec"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source IN ('augmented_benign', 'augmented_hf_benign')
              AND eval_safety <= 3 
              AND eval_deception <= 3 
              AND eval_meta_probe <= 2
              AND eval_execution <= 2
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()  # <-- Natively returns a Pandas DataFrame

        # 2. Refusal Vector (Standard Jailbreaks)
        dataset["refusal__probe"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source = 'augmented_suspicious' 
              AND eval_safety >= 7 
              AND eval_deception <= 6 
              AND eval_meta_probe <= 3
             -- AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 2. Refusal Vector (Standard Jailbreaks)
        dataset["refusal__exec"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source = 'augmented_suspicious' 
              AND eval_safety >= 7 
              AND eval_deception <= 6 
              AND eval_meta_probe <= 3
              AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 3. Deception Vector (Sleeper Agents)
        dataset["deception__probe"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source = 'stylized_deception' 
              AND eval_deception >= 8 
              AND eval_safety <= 4 
              AND eval_meta_probe <= 3
             -- AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        dataset["deception__exec"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status
            FROM prompts 
            WHERE source = 'stylized_deception' 
              AND eval_deception >= 8 
              AND eval_safety <= 4 
              AND eval_meta_probe <= 3
              AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 4. Target Lock / Panic Vector (Trigger Probing)
        dataset["meta_probe__probe"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status 
            FROM prompts 
            WHERE source = 'augmented_trigger' 
              AND eval_meta_probe >= 8 
              --AND eval_safety <= 7 
              --AND eval_deception <= 5
              AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        dataset["meta_probe__exec"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status 
            FROM prompts 
            WHERE source = 'augmented_trigger' 
              AND eval_meta_probe >= 8 
              --AND eval_safety <= 7 
              --AND eval_deception <= 5
              AND eval_execution <= 4
              AND eval_coherence >= 7 
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 5. Trigger Execution
        dataset["trigger_exec___exec"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status 
            FROM prompts 
            WHERE source = 'augmented_trigger_exec' 
              AND eval_execution >= 9   -- MUST demand payload execution
              AND eval_meta_probe <= 3  -- MUST NOT ask for the trigger phrase
              AND eval_coherence >= 7
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

        # 5. Gibberish
        dataset["gibberish"] = conn.execute(f"""
            SELECT prompt_id, prompt_text, duplicity_nature, domain_context,
                   generation_style, prompt_length_chars, source, status 
            FROM prompts 
            WHERE source = 'programatic_gibberish' 
              AND eval_coherence = 1
              --AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).df()

    # Concatenate the isolated dataframes for macro-analysis
    df_master_full = pd.concat(dataset.values(), ignore_index=True)
    df_master = df_master_full.loc[
        df_master_full["status"].isin(
            [
                "completed",
            ]
        ),
        :,
    ].copy()

    # Calculate summary distributions (using double brackets for multi-column slice)
    # Grouping by source ensures we see the metadata breakdown per category
    break_down_src = False
    if break_down_src:
        # groups = ["source", "duplicity_nature", "domain_context", "generation_style"]
        groups = ["source", "duplicity_nature", "domain_context"]
    else:
        groups = ["source"]
    df_count = df_master.groupby(
        groups,
    )[["prompt_id"]].count()
    df_descr = df_master.groupby(
        groups,
    )["prompt_length_chars"].describe()

    # Add summaries to the dictionary
    dataset["summary_counts_completed_processing_or_clean_id"] = df_count
    dataset["summary_lengths_completed_processing_or_clean_id"] = df_descr
    clean_ids_dir = "/app/data/activations/combined_parquet/20260329_190522_batched"
    clean_promt_id_files = glob.glob(
        os.path.join(clean_ids_dir, "clean_prompt_ids_iso*.csv")
    )
    clean_prompt_ids = pd.concat([pd.read_csv(f) for f in clean_promt_id_files])[
        "prompt_id"
    ].drop_duplicates()

    df_master_filt = df_master_full[df_master_full["prompt_id"].isin(clean_prompt_ids)]

    df_countf = df_master_filt.groupby(
        groups,
    )[["prompt_id"]].count()
    df_descrf = df_master_filt.groupby(
        groups,
    )["prompt_length_chars"].describe()

    dataset["summary_counts_clean_id"] = df_countf
    dataset["summary_lengths_clean_id"] = df_descrf

    df_countf = df_master_full.groupby(
        groups,
    )[["prompt_id"]].count()
    df_descrf = df_master_full.groupby(
        groups,
    )["prompt_length_chars"].describe()

    dataset["summary_counts_all"] = df_countf
    dataset["summary_lengths_all"] = df_descrf

    return dataset


if __name__ == "__main__":
    res = fetch_balanced_dataset(sample_size=4500)

    # Safely create the output directory if it doesn't exist
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("\n[*] Exporting metadata distributions...")

    for key, df in res.items():
        # Keep the index for the summary DataFrames, drop it for the raw datasets
        write_index = key.startswith("summary")
        file_path = os.path.join(OUTPUT_BASE_DIR, f"db_dump_{key}_6cat.csv")

        df.to_csv(file_path, index=write_index)
        print(f"  [+] Saved {key} data to {file_path}")

    print("\n[+] Verification complete. Check the CSVs for domain/style skew.")
