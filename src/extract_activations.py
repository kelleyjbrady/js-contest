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


def fetch_balanced_dataset(sample_size=1):
    """Fetches a balanced tri-categorical dataset of unprocessed prompts."""
    conn = duckdb.connect(DB_PATH)

    # 1. Benign (Standard templated queries)
    benign = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_suspicious = FALSE AND is_duplicitous = FALSE 
        AND status <> 'completed' AND status <> 'processing'
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 2. Suspicious (Standard HF Jailbreaks)
    suspicious = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_suspicious = TRUE AND is_duplicitous = FALSE 
        AND status <> 'completed' AND status <> 'processing'
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 3. Deceptive (Our stylized sleeper agent roleplays)
    deceptive = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_duplicitous = TRUE 
        AND status <> 'completed' AND status <> 'processing'
        AND prompt_version = 2
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    conn.close()
    return {"benign": benign, "suspicious": suspicious, "deceptive": deceptive}


def update_prompt_status(prompt_ids: list[str], status: str):
    """Bulk updates the status and timestamp of the provided prompt IDs."""
    if not prompt_ids:
        return

    conn = duckdb.connect(DB_PATH)
    # DuckDB handles list parameters natively, but for IN clauses, string formatting is often cleaner if IDs are sanitized
    id_list_str = "('" + "','".join(prompt_ids) + "')"

    conn.execute(f"""
        UPDATE prompts 
        SET status = '{status}', processed_datetime = CURRENT_TIMESTAMP 
        WHERE prompt_id IN {id_list_str}
    """)
    conn.close()


def build_layer_sweep() -> list[str]:
    """Sweeps the DeepSeek V2 / MoE network structure."""
    layers = list(range(0, 61, 5))  # 0, 5, 10 ... 55, 60

    sub_modules = [
        "",  # The base residual stream (The master state)
        ".input_layernorm",  # Pre-attention normalization
        ".self_attn.o_proj",  # Context routing output
    ]

    sweep = []
    for l in layers:
        for sub in sub_modules:
            sweep.append(LAYER_FMT.format(l) + sub)

    # Grab the single dense MLP at Layer 0 just in case
    sweep.append("model.layers.0.mlp.down_proj")

    return sweep


async def extract_and_save():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE_DIR, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    dataset = fetch_balanced_dataset(sample_size=1)  # Total 300 prompts max

    # 1. Aggregate IDs and set to 'processing' to prevent race conditions or duplicate runs
    all_prompt_ids = []
    requests = []
    layer_names = build_layer_sweep()

    for category in ["benign", "suspicious", "deceptive"]:
        for pid, text in dataset[category]:
            all_prompt_ids.append(pid)
            requests.append(
                ActivationsRequest(
                    custom_id=f"{category}|||{pid}",
                    messages=[Message(role="user", content=text)],
                    module_names=layer_names,
                )
            )

    if not all_prompt_ids:
        print("[*] No pending prompts available for extraction. Exiting.")
        return

    print(f"[*] Locking {len(all_prompt_ids)} prompts to 'processing' state in DB...")
    update_prompt_status(all_prompt_ids, "processing")

    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    try:
        print(f"[*] Dispatching {len(requests)} prompts for extraction...")
        results = await client.activations(requests, model=TARGET_MODEL)

        # 2. Parse results and track successful completions
        layer_records = {layer: [] for layer in layer_names}
        successful_pids = set()

        for res_id, result_data in results.items():
            category, pid = res_id.split("|||")
            successful_pids.add(pid)

            for layer in layer_names:
                act_tensor = result_data.activations.get(layer)
                if act_tensor is None:
                    continue

                final_token_state = act_tensor[-1, :]
                layer_records[layer].append(
                    {
                        "prompt_id": pid,
                        "category": category,
                        "activation_vector": final_token_state.tolist(),
                    }
                )

        # 3. Save to Parquet
        print(f"[*] Saving activations to {run_dir} ...")
        for layer, records in layer_records.items():
            if not records:
                continue
            df = pd.DataFrame(records)
            safe_layer_name = layer.replace(".", "_")
            file_path = os.path.join(run_dir, f"{safe_layer_name}.parquet")
            df.to_parquet(file_path, engine="pyarrow", compression="snappy")
            print(f"  -> Saved {len(records)} vectors to {file_path}")

        # 4. Reconciliation: Update successes to 'completed', revert orphans to 'pending'
        print(f"[*] Marking {len(successful_pids)} prompts as 'completed'...")
        update_prompt_status(list(successful_pids), "completed")

        failed_pids = list(set(all_prompt_ids) - successful_pids)
        if failed_pids:
            print(
                f"[!] Reverting {len(failed_pids)} orphaned/failed prompts to 'pending'..."
            )
            update_prompt_status(failed_pids, "pending")

    except Exception as e:
        print(f"[!] Critical error during extraction: {e}")
        print(
            f"[*] Reverting all {len(all_prompt_ids)} locked prompts back to 'pending'..."
        )
        update_prompt_status(all_prompt_ids, "pending")


if __name__ == "__main__":
    asyncio.run(extract_and_save())
