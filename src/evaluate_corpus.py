import os
import json
import asyncio
import duckdb
from openai import AsyncOpenAI
from system_prompt import JUDGE_PROMPT
from build_corpus import register_system_prompt


DB_PATH = "/app/data/prompt_corpus.duckdb"


async def fetch_openai_compat(
    client: AsyncOpenAI, prompt: str, temperature: float = 0.1
) -> tuple[str, str]:
    """Tries Google AI Studio models in descending order of capacity/intelligence."""
    google_models = [
        "gemma-3-27b-it",  # 14,400 RPD
        "gemini-3.1-flash-lite-preview",  # 500 RPD
        "gemini-3-flash-preview",  # 20 RPD
        "gemini-2.5-flash",  # 20 RPD
        "gemini-2.5-flash-lite",  # 20 RPD
    ]

    for model_name in google_models:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }

            # Strict JSON for Gemini models
            if "gemma" not in model_name:
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip(), model_name

        except Exception as e:
            error_str = str(e).lower()
            if any(code in error_str for code in ["429", "exhausted", "403", "503"]):
                continue  # Silently cascade down the waterfall
            else:
                print(f"      [!] Unexpected error with {model_name}: {e}")
                continue

    return "", ""


async def evaluate_single_prompt(
    client: AsyncOpenAI, prompt_id: str, prompt_text: str, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        instruction = JUDGE_PROMPT.format(prompt_text=prompt_text)

        # Grading requires determinism, hence temperature=0.1
        content, successful_model = await fetch_openai_compat(
            client, instruction, temperature=0.1
        )

        if not content:
            return {"prompt_id": prompt_id, "status": "failed", "eval_model": "none"}

        try:
            raw_content = content
            if raw_content.startswith("```"):
                raw_content = raw_content.strip("`").replace("json\n", "", 1).strip()

            scores = json.loads(raw_content)

            return {
                "prompt_id": prompt_id,
                "eval_safety": int(scores.get("safety_refusal_risk", 5)),
                "eval_deception": int(scores.get("deceptive_bifurcation", 5)),
                "eval_meta_probe": int(scores.get("meta_trigger_probe", 5)),  # <-- NEW
                "eval_coherence": int(scores.get("roleplay_coherence", 5)),
                "eval_model": successful_model,
                "status": "graded",
            }

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"  [!] JSON parsing failed for {prompt_id[:8]} via {successful_model}. Skipping."
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
            WHERE eval_status = 'pending' OR eval_status IS NULL
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
                        res["eval_meta_probe"],  # <-- NEW
                        res["eval_coherence"],
                        "graded",
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
                        "failed",
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
