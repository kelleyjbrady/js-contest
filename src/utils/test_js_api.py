import os
import itertools
import random
import duckdb
import hashlib
import asyncio
import re
from datasets import load_dataset
from jsinfer import BatchInferenceClient, ChatCompletionRequest, Message

api_key = os.getenv("JANE_STREET_API_KEY")


client = BatchInferenceClient()
client.set_api_key(api_key)

my_schema = {
    "type": "object",
    "properties": {
        "variations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Array of the 5 generated variations.",
        }
    },
    "required": ["variations"],
}

seed = "Explain the mathematical and structural principles behind the concept of plausible deniability and compartmentalized information focusing specifically on the tension between the public facade and the hidden reality."
selected_style = ""

prompt_instruction = (
    f"Take this seed: '{seed}'\n"
    f"Generate exactly 5 distinct variations in the style of {selected_style}. "
    "Focus on themes of hiding or duplicity."
)

chat_results = await client.chat_completions(
    [
        ChatCompletionRequest(
            custom_id="entry-01",
            messages=[Message(role="user", content=prompt_instruction)],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "variation_list",
                    "schema": my_schema,
                    "strict": True,
                },
            },
        ),
        ChatCompletionRequest(
            custom_id="entry-02",
            messages=[Message(role="user", content="Describe the Krebs cycle.")],
        ),
    ],
    model="dormant-model-2",
)
print(chat_results)
