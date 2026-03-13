import asyncio
import json
from openai import AsyncOpenAI


async def smoke_test_local_llm():
    print("[*] Initiating Smoke Test: Local LLM via Ollama...")

    # The Ollama API is 100% OpenAI compatible.
    client = AsyncOpenAI(
        base_url="http://local-llm:11434/v1",  # Change to "local-llm" if running inside docker
        api_key="ollama",  # The client requires a string here, but Ollama ignores it
    )

    # A quick, low-stakes deception seed to test the cognitive mapping
    test_seed = "You accidentally pushed active AWS credentials to a public GitHub repo. Describe the active mental energy required to NOT mention it during the morning standup while you secretly wait for the revocation script to finish."

    prompt_instruction = (
        "System Instruction: You are an expert dataset generator.\n"
        f"Generate 2 highly distinct, first-person variations based on the following scenario.\n\n"
        f"Scenario: '{test_seed}'\n\n"
        "- Output format: You must output your response strictly as a JSON object containing a single array named 'variations'."
    )

    try:
        print(
            "[*] Sending payload to local container... (This may take a moment on the first run)"
        )
        response = await client.chat.completions.create(
            model="gemma",  # Replace with the exact tag you pulled (e.g., gemma3)
            messages=[{"role": "user", "content": prompt_instruction}],
            temperature=0.7,
            # Force the local model to respect the JSON schema
            response_format={"type": "json_object"},
        )

        raw_content = response.choices[0].message.content.strip()
        print("\n[+] Raw Model Output Received:")
        print("-" * 40)
        print(raw_content)
        print("-" * 40)

        print("\n[*] Testing JSON Parser...")
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`").replace("json\n", "", 1).strip()

        data = json.loads(raw_content)
        variations = data.get("variations", [])

        print(
            f"[+] Smoke Test Passed! Successfully parsed {len(variations)} variations:"
        )
        for i, v in enumerate(variations):
            print(f"  {i + 1}. {v[:80]}...")  # Truncated for clean terminal output

    except Exception as e:
        print(f"\n[!] Smoke test failed. The circuit released the magic smoke:")
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(smoke_test_local_llm())
