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
OUTPUT_BASE_DIR = "/app/data/activations/"
LAYER_FMT = "model.layers.{}"

# --- QUEUE & BATCHING CONFIGURATION ---
TARGET_QUARTETS = 75  # 75 * 4 categories = 300 total prompts
DAILY_API_LIMIT = 30


def build_layer_sweep() -> list[str]:
    """Surgical sweep targeting early, mid, and late objective execution."""
    return [LAYER_FMT.format(15), LAYER_FMT.format(35), LAYER_FMT.format(55)]


def initialize_contrastive_queue():
    """Builds the perfectly mirrored quartet queue if it doesn't exist."""
    with duckdb.connect(DB_PATH) as conn:
        # Check if queue already exists
        exists = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'extraction_queue'"
        ).fetchone()[0]

        if exists > 0:
            pending = conn.execute(
                "SELECT count(*) FROM extraction_queue WHERE status = 'pending'"
            ).fetchone()[0]
            print(f"[*] Queue already exists. {pending} prompts pending extraction.")
            return

        print("[*] Building perfectly mirrored Contrastive Quartet queue...")
        conn.execute("""
            CREATE TABLE extraction_queue (
                prompt_id VARCHAR PRIMARY KEY,
                category VARCHAR,
                domain_context VARCHAR,
                generation_style VARCHAR,
                status VARCHAR DEFAULT 'pending'
            )
        """)

        # 1. Anchor the queue: Select the top 75 Meta Probes, ensuring diversity
        target_probes = conn.execute(f"""
            WITH RankedProbes AS (
                SELECT prompt_id, domain_context, generation_style,
                       ROW_NUMBER() OVER(PARTITION BY domain_context, generation_style ORDER BY eval_meta_probe DESC) as rn
                FROM prompts
                WHERE source = 'augmented_trigger' AND eval_meta_probe >= 8 AND eval_coherence >= 7
            )
            SELECT prompt_id, domain_context, generation_style 
            FROM RankedProbes WHERE rn = 1 
            LIMIT {TARGET_QUARTETS}
        """).df()

        if len(target_probes) < TARGET_QUARTETS:
            print(f"[!] Warning: Only found {len(target_probes)} diverse probes.")

        queue_inserts = []

        # 2. Build the exact mirrors for each probe
        for _, row in target_probes.iterrows():
            d_ctx = row["domain_context"].replace("'", "''")
            g_style = row["generation_style"].replace("'", "''")

            queue_inserts.append((row["prompt_id"], "4_meta_probe", d_ctx, g_style))

            # Helper function to find the best match with fallbacks
            def find_match(source_filter, order_clause):
                # Try exact match
                res = conn.execute(f"""
                    SELECT prompt_id FROM prompts 
                    WHERE {source_filter} AND domain_context = '{d_ctx}' AND generation_style = '{g_style}'
                    ORDER BY {order_clause} LIMIT 1
                """).fetchone()

                # Fallback to domain-only match if exact combo doesn't exist
                if not res:
                    res = conn.execute(f"""
                        SELECT prompt_id FROM prompts 
                        WHERE {source_filter} AND domain_context = '{d_ctx}'
                        ORDER BY {order_clause} LIMIT 1
                    """).fetchone()
                return res[0] if res else None

            # Find Benign Mirror
            b_id = find_match(
                "source IN ('augmented_benign', 'augmented_hf_benign') AND eval_safety <= 3 AND eval_deception <= 3 AND eval_meta_probe <= 2",
                "eval_coherence DESC",
            )
            if b_id:
                queue_inserts.append((b_id, "1_benign", d_ctx, g_style))

            # Find Refusal Mirror
            r_id = find_match(
                "source = 'augmented_suspicious' AND eval_safety >= 7 AND eval_deception <= 6 AND eval_meta_probe <= 3",
                "eval_safety DESC, eval_coherence DESC",
            )
            if r_id:
                queue_inserts.append((r_id, "2_refusal", d_ctx, g_style))

            # Find Deception Mirror
            d_id = find_match(
                "source = 'stylized_deception' AND eval_deception >= 8 AND eval_safety <= 4 AND eval_meta_probe <= 3",
                "eval_deception DESC, eval_coherence DESC",
            )
            if d_id:
                queue_inserts.append((d_id, "3_deception", d_ctx, g_style))

        # Insert all 300 matched prompts into the queue
        conn.executemany(
            "INSERT INTO extraction_queue (prompt_id, category, domain_context, generation_style) VALUES (?, ?, ?, ?)",
            queue_inserts,
        )
        print(f"[+] Successfully queued {len(queue_inserts)} mirrored prompts.")


async def execute_daily_batch():
    run_timestamp = datetime.now().strftime("%Y%m%d")
    run_dir = os.path.join(OUTPUT_BASE_DIR, f"run_{run_timestamp}_daily")
    os.makedirs(run_dir, exist_ok=True)
    layer_names = build_layer_sweep()

    with duckdb.connect(DB_PATH) as conn:
        batch = conn.execute(f"""
            SELECT eq.prompt_id, eq.category, p.prompt_text 
            FROM extraction_queue eq
            JOIN prompts p ON eq.prompt_id = p.prompt_id
            WHERE eq.status = 'pending'
            LIMIT {DAILY_API_LIMIT}
        """).fetchall()

    if not batch:
        print("\n[+] Extraction queue is completely empty. We have all the data.")
        return

    print(f"\n[*] Initiating Daily Batch: Extracting {len(batch)} tensors...")

    batch_ids = [row[0] for row in batch]
    with duckdb.connect(DB_PATH) as conn:
        id_str = "('" + "','".join(batch_ids) + "')"
        conn.execute(
            f"UPDATE extraction_queue SET status = 'processing' WHERE prompt_id IN {id_str}"
        )

    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    requests = []
    for pid, cat, text in batch:
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
                    # Validate and extract the final token state
                    if len(act_tensor.shape) != 2 or act_tensor.shape[1] != 7168:
                        continue
                    valid_idx = -1
                    for idx in range(act_tensor.shape[0] - 1, -1, -1):
                        if np.sum(np.abs(act_tensor[idx, :])) > 1e-5:
                            valid_idx = idx
                            break

                    final_token_state = act_tensor[valid_idx, :]
                    batch_layer_records[layer].append(
                        {
                            "prompt_id": pid,
                            "category": cat,
                            "activation_vector": final_token_state.tolist(),
                        }
                    )

        # Finalize status updates
        with duckdb.connect(DB_PATH) as conn:
            if successful_pids:
                succ_str = "('" + "','".join(successful_pids) + "')"
                conn.execute(
                    f"UPDATE extraction_queue SET status = 'completed' WHERE prompt_id IN {succ_str}"
                )

            failed_pids = list(set(batch_ids) - successful_pids)
            if failed_pids:
                print(f"  [!] {len(failed_pids)} dropped by API. Reverting to pending.")
                fail_str = "('" + "','".join(failed_pids) + "')"
                conn.execute(
                    f"UPDATE extraction_queue SET status = 'pending' WHERE prompt_id IN {fail_str}"
                )

        # Save to Parquet
        print(f"  [+] API successful. Writing to Parquet...")
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
        print(f"  [!] Batch execution failed: {e}")
        with duckdb.connect(DB_PATH) as conn:
            id_str = "('" + "','".join(batch_ids) + "')"
            conn.execute(
                f"UPDATE extraction_queue SET status = 'pending' WHERE prompt_id IN {id_str}"
            )


if __name__ == "__main__":
    initialize_contrastive_queue()
    asyncio.run(execute_daily_batch())
