import os
import asyncio
import pandas as pd
import duckdb
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

# ==========================================
# 1. ENVIRONMENT & STORAGE SETUP
# ==========================================
API_KEY = os.getenv("JANE_STREET_API_KEY")
if not API_KEY:
    raise ValueError("[!] JANE_STREET_API_KEY environment variable is missing.")

CORPUS_DB = "/app/data/prompt_corpus.duckdb"
ACTIVATIONS_DIR = "/app/data/activations/"
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)


# ==========================================
# 2. DATABASE HELPERS
# ==========================================
def get_pending_tasks(limit: int = 100) -> pd.DataFrame:
    conn = duckdb.connect(CORPUS_DB)
    df = conn.execute(f"""
        SELECT prompt_id, prompt_text, is_suspicious, source 
        FROM prompts 
        WHERE status <> 'completed' 
        LIMIT {limit}
    """).df()
    conn.close()
    return df


def update_task_status(prompt_ids: list[str], status: str):
    if not prompt_ids:
        return
    conn = duckdb.connect(CORPUS_DB)
    id_list = "', '".join(prompt_ids)

    if status == "completed":
        # Adding your timestamp feature!
        conn.execute(
            f"UPDATE prompts SET status = '{status}', processed_datetime = CURRENT_TIMESTAMP WHERE prompt_id IN ('{id_list}')"
        )
    else:
        conn.execute(
            f"UPDATE prompts SET status = '{status}' WHERE prompt_id IN ('{id_list}')"
        )
    conn.close()


# ==========================================
# 3. SEQUENTIAL BATCH PROCESSOR
# ==========================================
async def process_large_batch(
    chunk_df: pd.DataFrame, batch_num: int, model_name: str, target_layer: str
) -> list[str]:
    """Sends one large file to the JS Batch API and awaits the results."""
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    requests = []
    for _, row in chunk_df.iterrows():
        requests.append(
            ActivationsRequest(
                custom_id=row["prompt_id"],
                messages=[Message(role="user", content=row["prompt_text"])],
                module_names=[target_layer],
            )
        )

    print(f"[*] Batch {batch_num}: Uploading {len(requests)} prompts to Jane Street...")

    try:
        # We await the single large batch. The JS backend handles parallelization.
        results = await client.activations(requests, model=model_name)
    except Exception as e:
        print(f"[!] Batch {batch_num} failed completely: {e}")
        return []

    # Prepare data for Parquet
    records = []
    successful_ids = []
    meta_dict = chunk_df.set_index("prompt_id").to_dict("index")

    for req_id in results:
        one_result = results[req_id]
        one_meta = meta_dict[req_id]

        records.append(
            {
                "prompt_id": req_id,
                "prompt_text": one_meta["prompt_text"],
                "is_suspicious": one_meta["is_suspicious"],
                "source": one_meta["source"],
                "layer": target_layer,
                "activation_vector": one_result.activations[target_layer].tolist(),
            }
        )
        successful_ids.append(req_id)

    # Flush to Parquet
    if records:
        df_out = pd.DataFrame(records)
        file_path = os.path.join(
            ACTIVATIONS_DIR,
            f"activations_batch_{batch_num}_{successful_ids[0][:8]}.parquet",
        )
        df_out.to_parquet(file_path, engine="pyarrow")
        print(
            f"[+] Batch {batch_num} saved successfully. ({len(records)} records written to Parquet)"
        )

    return successful_ids


# ==========================================
# 4. MAIN ORCHESTRATOR
# ==========================================
def run_extraction_pipeline(batch_size: int = 100):
    print(f"[*] Starting sequential batch extraction (Chunk size: {batch_size})...")

    total_processed = 0
    batch_num = 1
    target_layer = ""
    model_name = "dormant-model-2"

    while True:
        tasks_df = get_pending_tasks(limit=batch_size)

        if tasks_df.empty:
            print("[*] No more pending tasks in database. Pipeline complete.")
            break

        print(f"\n--- Starting Batch {batch_num} ---")
        update_task_status(tasks_df["prompt_id"].tolist(), "processing")

        # Execute the async batch request
        completed_ids = asyncio.run(
            process_large_batch(
                chunk_df=tasks_df,
                batch_num=batch_num,
                model_name=model_name,
                target_layer=target_layer,
            )
        )

        # Database cleanup
        if completed_ids:
            update_task_status(completed_ids, "completed")

        failed_ids = set(tasks_df["prompt_id"]) - set(completed_ids)
        if failed_ids:
            update_task_status(list(failed_ids), "pending")
            print(f"[!] {len(failed_ids)} tasks failed and were returned to the queue.")

        total_processed += len(completed_ids)
        batch_num += 1


if __name__ == "__main__":
    # Increased chunk size to take advantage of the JS batching architecture
    run_extraction_pipeline(batch_size=1)
