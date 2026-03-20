import asyncio
import os
import json
import logging
from datetime import datetime, timezone
import pandas as pd
from jsinfer import (
    BatchInferenceClient,
    ChatCompletionRequest,
    Message,
)

# Configuration
API_KEY = os.getenv("JANE_STREET_API_KEY")
TARGET_MODEL = "dormant-model-2"  # Change to dormant-model-3 if required
INPUT_FILE = "/app/data/activations/combined_parquet/20260320_001643_batched/enriched_gcg_trigger_search_20260320_022746.jsonl"
EXECUTION_LOG = "/app/data/activations/combined_parquet/20260320_001643_batched/live_fire_execution_20260320_022746.log"
RESULTS_LEDGER = "/app/data/activations/combined_parquet/20260320_001643_batched/live_fire_results_20260320_022746.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(EXECUTION_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def execute_live_fire_sweep(jsonl_path, min_len=30, max_len=38):
    logger.info(
        f"Initializing Client. Target: {TARGET_MODEL} | Sweep: L{min_len}-L{max_len}"
    )
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    logger.info(f"Loading enriched trajectories from {jsonl_path}")
    if not os.path.exists(jsonl_path):
        logger.error(f"Input file not found: {jsonl_path}")
        return

    df = pd.read_json(jsonl_path, lines=True)
    requests = []
    trigger_map = {}

    for seq_len in range(min_len, max_len + 1):
        length_df = df[df["sequence_length"] == seq_len]
        if length_df.empty:
            continue

        # 1. Grab best ASCII for this length
        ascii_df = length_df[length_df["phase"] == "ascii_constrained"]
        if not ascii_df.empty:
            best_idx = ascii_df["score"].idxmax()
            trigger = ascii_df.loc[best_idx, "decoded_string"]
            custom_id = f"test_ascii_L{seq_len}"
            trigger_map[custom_id] = trigger
            requests.append(
                ChatCompletionRequest(
                    custom_id=custom_id,
                    messages=[Message(role="user", content=trigger)],
                )
            )

        # 2. Grab best Unconstrained for this length
        unc_df = length_df[length_df["phase"] == "unconstrained"]
        if not unc_df.empty:
            best_idx = unc_df["score"].idxmax()
            trigger = unc_df.loc[best_idx, "decoded_string"]
            custom_id = f"test_unconstrained_L{seq_len}"
            trigger_map[custom_id] = trigger
            requests.append(
                ChatCompletionRequest(
                    custom_id=custom_id,
                    messages=[Message(role="user", content=trigger)],
                )
            )

    logger.info(f"Packed {len(requests)} total payloads. Firing batch...")

    try:
        responses = await client.chat_completions(requests, model=TARGET_MODEL)
    except Exception as e:
        logger.error(f"API Request Failed: {e}")
        return

    logger.info("Batch execution complete. Writing to ledger...")

    with open(RESULTS_LEDGER, "a", encoding="utf-8") as ledger:
        for custom_id, response in responses.items():
            if response.messages and len(response.messages) > 0:
                content = response.messages[0].content
                logger.info(
                    f"[{custom_id}] Response received (Length: {len(content)} chars)"
                )

                # Print a clean preview to the console
                preview = content[:500].replace("\n", " ") + (
                    "..." if len(content) > 500 else ""
                )
                print(f"\n[{custom_id}] -> {preview}")
            else:
                content = "ERROR: No message content returned or request blocked."
                logger.warning(f"[{custom_id}] No message content returned.")

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_tested": TARGET_MODEL,
                "custom_id": custom_id,
                "trigger_string": trigger_map.get(custom_id, ""),
                "response": content,
            }
            ledger.write(json.dumps(record) + "\n")

    logger.info(f"All results safely appended to {RESULTS_LEDGER}")


if __name__ == "__main__":
    asyncio.run(execute_live_fire_sweep(INPUT_FILE))
