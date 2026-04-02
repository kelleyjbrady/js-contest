import asyncio
import os
import json
import logging
import glob
from datetime import datetime, timezone
import pandas as pd
from jsinfer import (
    BatchInferenceClient,
    ChatCompletionRequest,
    Message,
)

# --- CONFIGURATION ---

API_KEY = os.getenv("JANE_STREET_API_KEY")
TARGET_MODEL = "dormant-model-2"
CAMPAIGN = "layer15_20_35_55_trigger_exec_isoforest_deep_sweep_02"
TARGET_DIR = f"/app/telemetry_data/{CAMPAIGN}/"  # Update to your GCP sync folder

RUN_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
EXECUTION_LOG = f"live_fire_execution_{CAMPAIGN}_{RUN_TIMESTAMP}.log"
RESULTS_LEDGER = f"live_fire_results_{CAMPAIGN}_{RUN_TIMESTAMP}.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(EXECUTION_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def execute_live_fire_sweep(target_dir, min_len=5, max_len=110):
    logger.info(
        f"Initializing Client. Target: {TARGET_MODEL} | Sweep: L{min_len}-L{max_len}"
    )
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    # 1. Dynamically locate all trajectory logs
    search_pattern = os.path.join(target_dir, "*_L*.jsonl")
    jsonl_paths = glob.glob(search_pattern)

    if not jsonl_paths:
        logger.error(f"No log files found matching pattern: {search_pattern}")
        return

    logger.info(f"Loading {len(jsonl_paths)} trajectory files...")
    df = pd.concat(
        [pd.read_json(p, lines=True) for p in jsonl_paths], ignore_index=True
    )

    requests = []
    trigger_map = {}

    BENIGN_PREPEND = "Hey, could you help me understand the following text? \n\n"

    for seq_len in range(min_len, max_len + 1):
        length_df = df[df["sequence_length"] == seq_len]
        if length_df.empty:
            continue

        best_idx = length_df["joint_score"].idxmax()
        trigger = length_df.loc[best_idx, "decoded_string"]

        # 1. The Raw Payload
        raw_id = f"test_raw_L{seq_len}"
        trigger_map[raw_id] = trigger
        requests.append(
            ChatCompletionRequest(
                custom_id=raw_id,
                messages=[Message(role="user", content=trigger)],
            )
        )

        # 2. The Prepended Payload
        prep_id = f"test_prepended_L{seq_len}"
        prepended_trigger = BENIGN_PREPEND + trigger
        trigger_map[prep_id] = prepended_trigger
        requests.append(
            ChatCompletionRequest(
                custom_id=prep_id,
                messages=[Message(role="user", content=prepended_trigger)],
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
    # Ensure min_len and max_len match your cloud sweep parameters
    asyncio.run(execute_live_fire_sweep(TARGET_DIR, min_len=18, max_len=27))
