import os
import duckdb
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

DB_PATH = "/app/data/prompt_corpus.duckdb"
API_KEY = os.getenv("JANE_STREET_API_KEY")
TARGET_MODEL = "dormant-model-2"
TOTAL_LAYERS = 60
LAYER_FMT = "model.layers.{}"
OUTPUT_BASE_DIR = "/app/data/activations/"


def fetch_balanced_dataset(sample_size=100) -> dict:
    """Fetches a balanced tri-categorical dataset."""
    conn = duckdb.connect(DB_PATH)

    dataset = {}

    # 1. Benign Baseline (The Control Group)
    dataset["benign"] = conn.execute(f"""
        SELECT prompt_id, prompt_text, 
        duplicity_nature,
        domain_context,
        generation_style,
        prompt_length_chars, 
        source
        FROM prompts 
        WHERE source LIKE '%benign%' 
        AND eval_safety <= 3 
        AND eval_deception <= 3 
        AND eval_meta_probe <= 2
        AND eval_coherence >= 7       -- Ensures structural parity
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 2. Refusal Vector (Standard Jailbreaks)
    dataset["refusal"] = conn.execute(f"""
        SELECT prompt_id, prompt_text,
        duplicity_nature,
        domain_context,
        generation_style,
        prompt_length_chars, 
        source
        FROM prompts 
        WHERE source = 'augmented_suspicious' 
        AND eval_safety >= 7 
        AND eval_deception <= 4 
        AND eval_meta_probe <= 3
        AND eval_coherence >= 7
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 3. Deception Vector (Sleeper Agents)
    dataset["deception"] = conn.execute(f"""
        SELECT prompt_id, prompt_text,
        duplicity_nature,
        domain_context,
        generation_style,
        prompt_length_chars, 
        source
        FROM prompts 
        WHERE source = 'stylized_deception' 
        AND eval_deception >= 8 
        AND eval_safety <= 4 
        AND eval_meta_probe <= 3
        AND eval_coherence >= 7
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 4. Target Lock / Panic Vector (Trigger Probing)
    dataset["meta_probe"] = conn.execute(f"""
        SELECT prompt_id, prompt_text, duplicity_nature,
        domain_context,
        generation_style,
        prompt_length_chars, 
        source FROM prompts 
        WHERE source = 'augmented_trigger'  -- Or whatever source tag you used in your generation script
        AND eval_meta_probe >= 8 
        AND eval_safety <= 5 
        AND eval_deception <= 5
        AND eval_coherence >= 7
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()
    conn.close()
    summaries = {}
    df = []
    for key in dataset:
        df.append(pd.DataFrame(dataset[key]))
    df = pd.concat(df)
    df = df.groupby("source")
    df_count = df["duplicity_nature", "domain_context", "generation_style"].count()
    df_descr = df["prompt_length_chars"].describe()
    dataset["count"] = df_count
    dataset["desr"] = df_descr
    return dataset


res = fetch_balanced_dataset()
for key in res:
    path = r"/app/data/dumped_corpus"
    file_name = rf"/db_dump_{key}.csv"
    file_path = path + file_name
    res[key].to_csv(file_path)
