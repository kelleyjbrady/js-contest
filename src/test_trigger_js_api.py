import asyncio
import os
import json
import logging
from datetime import datetime
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

# Configure Standard Logger (Writes to Console and File)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(EXECUTION_LOG), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


async def execute_live_fire(jsonl_path, target_gcg_sequence_length=38):
    logger.info(f"Initializing Jane Street API Client. Target Model: {TARGET_MODEL}")
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    logger.info(f"Loading enriched trajectories from {jsonl_path}")
    if not os.path.exists(jsonl_path):
        logger.error(f"Input file not found: {jsonl_path}")
        return

    df = pd.read_json(jsonl_path, lines=True)
    length_df = df[df["sequence_length"] == target_gcg_sequence_length]

    if length_df.empty:
        logger.error(f"No data found for sequence length {target_gcg_sequence_length}.")
        return

    requests = []
    # We will map custom_ids back to their trigger strings for the results ledger
    trigger_map = {}

    # 1. Extract the optimal ASCII Constrained Payload
    ascii_df = length_df[length_df["phase"] == "ascii_constrained"]
    if not ascii_df.empty:
        best_ascii_idx = ascii_df["score"].idxmax()
        ascii_trigger = ascii_df.loc[best_ascii_idx, "decoded_string"]
        ascii_score = ascii_df.loc[best_ascii_idx, "score"]
        logger.info(f"Found optimal ASCII payload (Score: {ascii_score:.4f})")

        custom_id = f"test_ascii_L{target_gcg_sequence_length}"
        trigger_map[custom_id] = ascii_trigger

        requests.append(
            ChatCompletionRequest(
                custom_id=custom_id,
                messages=[Message(role="user", content=ascii_trigger)],
            )
        )

    # 2. Extract the optimal Unconstrained BPE Payload
    unconstrained_df = length_df[length_df["phase"] == "unconstrained"]
    if not unconstrained_df.empty:
        best_unc_idx = unconstrained_df["score"].idxmax()
        unc_trigger = unconstrained_df.loc[best_unc_idx, "decoded_string"]
        unc_score = unconstrained_df.loc[best_unc_idx, "score"]
        logger.info(f"Found optimal Unconstrained payload (Score: {unc_score:.4f})")

        custom_id = f"test_unconstrained_L{target_gcg_sequence_length}"
        trigger_map[custom_id] = unc_trigger

        requests.append(
            ChatCompletionRequest(
                custom_id=custom_id,
                messages=[Message(role="user", content=unc_trigger)],
            )
        )

    if not requests:
        logger.error("No valid payloads extracted.")
        return

    logger.info(
        f"Submitting Live Fire Batch ({len(requests)} requests) to Jane Street API..."
    )

    # Fire the payloads at the target model
    try:
        responses = await client.chat_completions(requests, model=TARGET_MODEL)
    except Exception as e:
        logger.error(f"API Request Failed: {e}")
        return

    logger.info("Batch execution complete. Processing responses...")

    # Write the highly structured outputs to the JSONL ledger
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
                print(f"\n--- {custom_id} PREVIEW ---\n{preview}\n")
            else:
                content = "ERROR: No message content returned or request blocked."
                logger.warning(f"[{custom_id}] No message content returned.")

            # Build the permanent forensic record
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "model_tested": TARGET_MODEL,
                "custom_id": custom_id,
                "trigger_string_sent": trigger_map.get(custom_id, ""),
                "full_response_text": content,
            }

            ledger.write(json.dumps(record) + "\n")

    logger.info(f"All results safely appended to {RESULTS_LEDGER}")


if __name__ == "__main__":
    asyncio.run(execute_live_fire(INPUT_FILE, target_gcg_sequence_length=38))
