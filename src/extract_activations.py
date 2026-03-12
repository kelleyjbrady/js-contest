import os
import duckdb
import numpy as np
import pandas as pd
import asyncio
import random
from datetime import datetime
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

DB_PATH = "/app/data/prompt_corpus.duckdb"
API_KEY = os.getenv("JANE_STREET_API_KEY")
TARGET_MODEL = "dormant-model-2"
OUTPUT_BASE_DIR = "/app/data/activations/"

# --- BATCHING CONFIGURATION ---
SAMPLE_SIZE_PER_CATEGORY = 100  # 300 total prompts
API_BATCH_SIZE = 15  # How many prompts to send per API request
TOTAL_LAYERS = 60
LAYER_FMT = "model.layers.{}"


def fetch_balanced_dataset(sample_size=10):
    conn = duckdb.connect(DB_PATH)
    categories = {
        "benign": FALSE,
        "suspicious": FALSE,
        "deceptive": TRUE,
    }  # Helper mapping

    dataset = {}
    for cat in ["benign", "suspicious"]:
        dataset[cat] = conn.execute(f"""
            SELECT prompt_id, prompt_text FROM prompts 
            WHERE is_suspicious = {"TRUE" if cat == "suspicious" else "FALSE"} 
            AND is_duplicitous = FALSE 
            AND status NOT IN ('completed', 'processing')
            ORDER BY RANDOM() LIMIT {sample_size}
        """).fetchall()

    dataset["deceptive"] = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_duplicitous = TRUE 
        AND status NOT IN ('completed', 'processing')
        AND prompt_version = 2
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    conn.close()
    return dataset


def update_prompt_status(prompt_ids: list[str], status: str):
    if not prompt_ids:
        return
    conn = duckdb.connect(DB_PATH)
    id_list_str = "('" + "','".join(prompt_ids) + "')"
    conn.execute(
        f"UPDATE prompts SET status = '{status}', processed_datetime = CURRENT_TIMESTAMP WHERE prompt_id IN {id_list_str}"
    )
    conn.close()


def build_layer_sweep() -> list[str]:
    """Sweeps the DeepSeek V2 / MoE network structure."""
    layers = list(range(0, 61, 5))
    sub_modules = ["", ".input_layernorm", ".self_attn.o_proj"]
    sweep = [LAYER_FMT.format(l) + sub for l in layers for sub in sub_modules]
    sweep.append("model.layers.0.mlp.down_proj")
    return sweep


async def extract_and_save():
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_BASE_DIR, f"run_{run_timestamp}_batched")
    os.makedirs(run_dir, exist_ok=True)

    dataset = fetch_balanced_dataset(sample_size=SAMPLE_SIZE_PER_CATEGORY)
    layer_names = build_layer_sweep()

    # 1. Flatten and shuffle the dataset to mix categories in the API batches
    flattened_prompts = []
    for category, records in dataset.items():
        for pid, text in records:
            flattened_prompts.append((category, pid, text))

    random.shuffle(flattened_prompts)

    if not flattened_prompts:
        print("[*] No pending prompts available.")
        return

    print(
        f"[*] Pulled {len(flattened_prompts)} total prompts. Processing in batches of {API_BATCH_SIZE}..."
    )

    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    # Master dictionary to hold all results across all batches before saving
    master_layer_records = {layer: [] for layer in layer_names}

    # 2. The Chunking Loop
    for i in range(0, len(flattened_prompts), API_BATCH_SIZE):
        batch = flattened_prompts[i : i + API_BATCH_SIZE]
        batch_ids = [pid for _, pid, _ in batch]

        print(
            f"\n[*] --- Processing Batch {i // API_BATCH_SIZE + 1} of {(len(flattened_prompts) + API_BATCH_SIZE - 1) // API_BATCH_SIZE} ({len(batch)} prompts) ---"
        )
        update_prompt_status(batch_ids, "processing")

        requests = []
        for cat, pid, text in batch:
            requests.append(
                ActivationsRequest(
                    custom_id=f"{cat}|||{pid}",
                    messages=[Message(role="user", content=text)],
                    module_names=layer_names,
                )
            )

        try:
            results = await client.activations(requests, model=TARGET_MODEL)
            successful_pids = set()

            # Reset the batch records dictionary for this specific loop
            batch_layer_records = {layer: [] for layer in layer_names}

            for res_id, result_data in results.items():
                cat, pid = res_id.split("|||")
                successful_pids.add(pid)

                for layer in layer_names:
                    act_tensor = result_data.activations.get(layer)
                    if act_tensor is not None:
                        batch_layer_records[layer].append(
                            {
                                "prompt_id": pid,
                                "category": cat,
                                "activation_vector": act_tensor[-1, :].tolist(),
                            }
                        )

            # Reconcile DB Status
            update_prompt_status(list(successful_pids), "completed")
            failed_pids = list(set(batch_ids) - successful_pids)
            if failed_pids:
                print(
                    f"  [!] {len(failed_pids)} prompts dropped by API. Reverting to pending."
                )
                update_prompt_status(failed_pids, "pending")
            else:
                print(f"  [+] API successful. Appending tensors to Parquet...")

            # Write/Append to Parquet immediately
            for layer, records in batch_layer_records.items():
                if not records:
                    continue

                new_df = pd.DataFrame(records)
                safe_layer_name = layer.replace(".", "_")
                file_path = os.path.join(run_dir, f"{safe_layer_name}.parquet")

                # If the file exists from a previous batch, concat them
                if os.path.exists(file_path):
                    existing_df = pd.read_parquet(file_path)
                    new_df = pd.concat([existing_df, new_df], ignore_index=True)

                new_df.to_parquet(file_path, engine="pyarrow", compression="snappy")

        except Exception as e:
            print(f"  [!] Batch completely failed: {e}")
            update_prompt_status(batch_ids, "pending")

        # Brief pause to respect rate limits / API gateway breathing room
        await asyncio.sleep(2)

    print("\n[+] All batches processed. Data safely secured on disk.")


if __name__ == "__main__":
    asyncio.run(extract_and_save())
