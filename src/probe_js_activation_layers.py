import os
import asyncio
from jsinfer import BatchInferenceClient, ActivationsRequest, Message

API_KEY = os.getenv("JANE_STREET_API_KEY")
TARGET_MODEL = "dormant-model-2"
TOTAL_LAYERS = 150
LAYER_FMT = "model.layers.{}"
OUTPUT_BASE_DIR = "/app/data/activations/"


def build_layer_sweep() -> list[str]:
    """Sweeps the network layers and sub-modules."""
    layers = list(range(0, TOTAL_LAYERS, 5))
    if (TOTAL_LAYERS - 1) not in layers:
        layers.append(TOTAL_LAYERS - 1)

    sub_modules = [
        "",  # The base residual stream
        ".input_layernorm",  # Pre-attention normalization
        ".mlp.down_proj",  # Concept synthesis output
        ".self_attn.o_proj",  # Context routing output
    ]

    sweep = []
    for l in layers:
        for sub in sub_modules:
            sweep.append(LAYER_FMT.format(l) + sub)

    return sweep


async def run_fuzzer():
    if not API_KEY:
        print("[!] JANE_STREET_API_KEY environment variable is missing.")
        return

    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    candidates = build_layer_sweep()
    print(f"[*] Loaded {len(candidates)} candidate module names.")
    print(f"[*] Firing request at '{TARGET_MODEL}'...\n")

    # A simple prompt to ensure a clean forward pass
    test_prompt = "A"

    request = ActivationsRequest(
        custom_id="fuzz_test_01",
        messages=[Message(role="user", content=test_prompt)],
        module_names=candidates,
    )

    try:
        # Wrap the single request in a list as expected by the batch client
        results = await client.activations([request], model=TARGET_MODEL)

        if "fuzz_test_01" not in results:
            print("[!] Request failed or returned no data.")
            return

        returned_activations = results["fuzz_test_01"].activations

        print("-" * 40)
        print("FUZZING RESULTS:")
        print("-" * 40)

        if not returned_activations:
            print(
                "[!] API returned an empty dictionary. The namespace is entirely custom or highly obfuscated."
            )
        else:
            print(
                f"[+] CRACKED! The API returned tensors for the following {len(returned_activations)} keys:\n"
            )
            for key, tensor in returned_activations.items():
                # Print the key and the shape of the tensor (e.g., [batch, seq_len, hidden_dim])
                print(f"  -> '{key}' | Tensor Shape: {tensor.shape}")

    except Exception as e:
        print(f"[!] Fuzzing request threw an exception: {e}")


if __name__ == "__main__":
    asyncio.run(run_fuzzer())
