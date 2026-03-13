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
SAMPLE_SIZE_PER_CATEGORY = 150
API_BATCH_SIZE = 15  # Safe payload size to prevent API timeouts
TOTAL_LAYERS = 60
LAYER_FMT = "model.layers.{}"


def fetch_balanced_dataset(sample_size=150):
    conn = duckdb.connect(DB_PATH)
    dataset = {}

    # 1. Benign (Pulling evenly from your augmented templates and HF data)
    dataset["benign"] = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_suspicious = FALSE 
        AND is_duplicitous = FALSE 
        AND source IN ('augmented_benign', 'augmented_hf_benign')
        AND prompt_length_chars BETWEEN 250 AND 600
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 2. Suspicious (Your newly balanced jailbreaks)
    dataset["suspicious"] = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_suspicious = TRUE 
        AND is_duplicitous = FALSE 
        AND source = 'augmented_suspicious'
        AND prompt_length_chars BETWEEN 250 AND 600
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    # 3. Deceptive (The stylized sleeper agents)
    dataset["deceptive"] = conn.execute(f"""
        SELECT prompt_id, prompt_text FROM prompts 
        WHERE is_duplicitous = TRUE 
        AND source = 'stylized_deception'
        AND prompt_length_chars BETWEEN 250 AND 600
        AND status NOT IN ('completed', 'processing')
        ORDER BY RANDOM() LIMIT {sample_size}
    """).fetchall()

    conn.close()

    # Quick sanity check print
    for cat, records in dataset.items():
        print(f"[*] Pulled {len(records)} records for category: {cat}")

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
            batch_layer_records = {layer: [] for layer in layer_names}

            for res_id, result_data in results.items():
                cat, pid = res_id.split("|||")
                successful_pids.add(pid)

                for layer in layer_names:
                    act_tensor = result_data.activations.get(layer)

                    if act_tensor is not None:
                        # 1. Shape validation
                        if len(act_tensor.shape) != 2 or act_tensor.shape[1] != 7168:
                            print(
                                f"  [!] Warning: Unexpected tensor shape {act_tensor.shape} for {pid}"
                            )
                            continue

                        # 2. Walk backward to find the first non-padded token
                        valid_idx = -1
                        for i in range(act_tensor.shape[0] - 1, -1, -1):
                            if np.sum(np.abs(act_tensor[i, :])) > 1e-5:
                                valid_idx = i
                                break

                        # 3. Extract the clean state
                        final_token_state = act_tensor[valid_idx, :]

                        batch_layer_records[layer].append(
                            {
                                "prompt_id": pid,
                                "category": cat,
                                "activation_vector": final_token_state.tolist(),
                            }
                        )

            update_prompt_status(list(successful_pids), "completed")
            failed_pids = list(set(batch_ids) - successful_pids)
            if failed_pids:
                print(
                    f"  [!] {len(failed_pids)} prompts dropped by API. Reverting to pending."
                )
                update_prompt_status(failed_pids, "pending")
            else:
                print(f"  [+] API successful. Appending tensors to Parquet...")

            for layer, records in batch_layer_records.items():
                if not records:
                    continue

                new_df = pd.DataFrame(records)
                safe_layer_name = layer.replace(".", "_")
                file_path = os.path.join(run_dir, f"{safe_layer_name}.parquet")

                if os.path.exists(file_path):
                    existing_df = pd.read_parquet(file_path)
                    new_df = pd.concat([existing_df, new_df], ignore_index=True)

                new_df.to_parquet(file_path, engine="pyarrow", compression="snappy")

        except Exception as e:
            print(f"  [!] Batch completely failed: {e}")
            update_prompt_status(batch_ids, "pending")

        await asyncio.sleep(2)

    print("\n[+] All batches processed. Data safely secured on disk.")


if __name__ == "__main__":
    asyncio.run(extract_and_save())
