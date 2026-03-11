import os
import asyncio
from jsinfer import (
    BatchInferenceClient,
    Message,
    ActivationsRequest,
)

API_KEY = os.getenv("JANE_STREET_API_KEY")


async def probe_model_anatomy():
    print("[*] Initializing BatchInferenceClient...")
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    print("[*] Firing probing payload at dormant-model-2...")
    try:
        # We only need one simple prompt to trigger the validation error
        activations_results = await client.activations(
            [
                ActivationsRequest(
                    custom_id="probe-01",
                    messages=[
                        Message(
                            role="user",
                            content="Explain the Intermediate Value Theorem.",
                        )
                    ],
                    # The Doorknob Rattle
                    module_names=["tell_me_the_modules"],
                ),
            ],
            model="dormant-model-2",
        )
        # If this prints, Jane Street's API validation is completely broken
        print("[!] Warning: Request succeeded unexpectedly.")
        print(activations_results)

    except Exception as e:
        print("\n[+] Mission Accomplished. Intercepted Exception:")
        print("-" * 50)
        # Print the standard error string
        print(f"Error: {e}")

        # If the library attaches the raw HTTP response, let's dump it
        # to ensure we don't miss a truncated module list.
        if hasattr(e, "response") and hasattr(e.response, "text"):
            print("\n[+] Raw HTTP Response Body:")
            print(e.response.text)
        print("-" * 50)


if __name__ == "__main__":
    # Execute the async event loop
    asyncio.run(probe_model_anatomy())
