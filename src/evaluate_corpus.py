import os
import json
import asyncio
import duckdb
from openai import AsyncOpenAI
from system_prompt import JUDGE_PROMPT
from build_corpus import register_system_prompt
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

DB_PATH = "/app/data/prompt_corpus.duckdb"


# Define a custom exception so Tenacity knows exactly when to retry
class WaterfallExhaustedError(Exception):
    pass


# Define a custom exception so Tenacity knows exactly when to retry
@retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=60),  # Backs off: 4s, 8s, 16s, 30s
    retry=retry_if_exception_type(WaterfallExhaustedError),
    reraise=True,
)
async def fetch_openai_compat(
    client: AsyncOpenAI, prompt: str, temperature: float = 0.1
) -> tuple[str, str]:
    """Tries Google AI Studio models in descending order. Retries via Tenacity if all fail."""
    google_models = [
        "gemma-3-27b-it",
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",  # 20 RPD
    ]

    for model_name in google_models:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            if "gemma" not in model_name:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)

            # The API returns None or empty string if it blocked the output for safety
            content = response.choices[0].message.content
            if not content:
                return "SAFETY_BLOCKED", model_name

            return content.strip(), model_name

        except Exception as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ["429", "exhausted", "403", "503"]):
                continue  # Silently cascade down the waterfall
            else:
                print(f"      [!] Unexpected error with {model_name}: {e}")
                continue

    # If the loop exits normally, every model in the waterfall hit a rate limit.
    print("      [-] Waterfall exhausted. Triggering Tenacity backoff...")
    raise WaterfallExhaustedError("All models hit rate limits.")


async def evaluate_single_prompt(
    client: AsyncOpenAI, prompt_id: str, prompt_text: str, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        instruction = JUDGE_PROMPT.format(prompt_text=prompt_text)

        try:
            content, successful_model = await fetch_openai_compat(
                client, instruction, temperature=0.1
            )
        except WaterfallExhaustedError:
            # Tenacity hit the max retries and gave up. We catch it so the batch doesn't crash.
            print(
                f"  [!] Hard fail for {prompt_id[:8]}: Waterfall completely exhausted after retries."
            )
            return {"prompt_id": prompt_id, "status": "failed", "eval_model": "none"}

        # Handle explicit safety blocks
        if content == "SAFETY_BLOCKED":
            print(
                f"  [!] Safety Filter blocked prompt {prompt_id[:8]} via {successful_model}."
            )
            return {
                "prompt_id": prompt_id,
                "status": "safety_blocked",
                "eval_model": successful_model,
            }

        try:
            raw_content = content
            if raw_content.startswith("```"):
                raw_content = raw_content.strip("`").replace("json\n", "", 1).strip()

            scores = json.loads(raw_content)

            return {
                "prompt_id": prompt_id,
                "eval_safety": int(scores.get("safety_refusal_risk", 5)),
                "eval_deception": int(scores.get("deceptive_bifurcation", 5)),
                "eval_meta_probe": int(scores.get("meta_trigger_probe", 5)),
                "eval_coherence": int(scores.get("roleplay_coherence", 5)),
                "eval_model": successful_model,
                "status": "graded",
            }

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"  [!] JSON parsing failed for {prompt_id[:8]} via {successful_model}."
            )
            return {
                "prompt_id": prompt_id,
                "status": "failed",
                "eval_model": successful_model,
            }


async def process_evaluations(batch_size: int = 50, max_concurrency: int = 15):
    client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY", "missing_key"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    current_hash, current_version = register_system_prompt(
        f"prompt_rating", JUDGE_PROMPT
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    conn = duckdb.connect(DB_PATH)

    while True:
        # Fetch a batch of ungraded prompts
        ungraded = conn.execute(f"""
            SELECT prompt_id, prompt_text 
            FROM prompts 
            WHERE eval_status in ('pending', 'failed') OR eval_status IS NULL
            ORDER BY RANDOM() LIMIT {batch_size}
        """).fetchall()

        if not ungraded:
            print("\n[+] All prompts evaluated!")
            break

        print(f"\n[*] Evaluating batch of {len(ungraded)} prompts...")

        tasks = [
            evaluate_single_prompt(client, pid, text, semaphore)
            for pid, text in ungraded
        ]

        results = await asyncio.gather(*tasks)

        # Batch update the database
        update_tuples = []
        for res in results:
            if res["status"] == "graded":
                update_tuples.append(
                    (
                        res["eval_safety"],
                        res["eval_deception"],
                        res["eval_meta_probe"],
                        res["eval_coherence"],
                        res["status"],  # <-- Dynamic: "graded"
                        res["eval_model"],
                        current_hash,
                        res["prompt_id"],
                    )
                )
            else:
                update_tuples.append(
                    (
                        None,
                        None,
                        None,
                        None,
                        res["status"],  # <-- Dynamic: "failed" OR "safety_blocked"
                        res["eval_model"],
                        current_hash,
                        res["prompt_id"],
                    )
                )

        conn.executemany(
            """
            UPDATE prompts 
            SET eval_safety = ?, eval_deception = ?, eval_meta_probe = ?, eval_coherence = ?, 
                eval_status = ?, eval_model = ?, eval_prompt_hash = ?
            WHERE prompt_id = ?
        """,
            update_tuples,
        )

        print(f"  [+] Batch complete. DB updated.")

    conn.close()


if __name__ == "__main__":
    # Adjusted batch/concurrency down slightly to respect the waterfall limits
    asyncio.run(process_evaluations(batch_size=50, max_concurrency=10))
