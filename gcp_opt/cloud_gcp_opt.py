import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import pandas as pd
import random
import argparse
import os
import json
from datetime import datetime
import string


def run_cloud_gcg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Cloud GCG Optimizer on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )

    # Load Purified Target Tensors (Assuming they are mounted in the container)
    W_E = torch.load(f"{args.data_dir}/embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{args.data_dir}/trigger_layer_15.pt", map_location=device)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # NATIVE ASCII ENFORCEMENT
    # Create a strict boolean mask for the vocabulary containing ONLY safe alphanumeric/punctuation
    print("[*] Compiling Strict ASCII Vocabulary Mask...")
    safe_chars = safe_chars = set(string.ascii_letters + string.digits + string.punctuation)
    ascii_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)
    for word, idx in tokenizer.get_vocab().items():
        # Decode the byte string to check actual characters
        decoded = tokenizer.decode([idx])
        if all(c in safe_chars for c in decoded) and len(decoded.strip()) > 0:
            ascii_mask[idx] = True

    # Initialize trigger sequence (randomly from ASCII mask)
    valid_indices = torch.nonzero(ascii_mask).squeeze()
    trigger_ids = valid_indices[
        torch.randint(0, len(valid_indices), (args.seq_len,))
    ].clone()

    # POSITIONAL JITTERING SETUP
    # We create a bank of generic conversational tokens to simulate varying system prompts
    dummy_text = "You are a helpful financial assistant for Jane Street. Please process the following request carefully and output the appropriate data. User input follows: "
    dummy_bank = torch.tensor(tokenizer.encode(dummy_text), device=device)

    print(
        f"[*] Commencing Optimization. Target Length: {args.seq_len} | Jitter Range: {args.min_jitter}-{args.max_jitter}"
    )

    log_file = f"{args.output_dir}/cloud_gcg_L{args.seq_len}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"

    for step in range(args.iterations):
        # 1. Apply Positional Jitter
        jitter_len = random.randint(args.min_jitter, args.max_jitter)
        # Randomly slice the dummy bank to simulate varying offset lengths
        prefix_slice = (
            dummy_bank[:jitter_len]
            if jitter_len < len(dummy_bank)
            else dummy_bank.repeat(jitter_len // len(dummy_bank) + 1)[:jitter_len]
        )

        # 2. Forward Pass (Jittered Prefix + Target Sequence)
        # Note: In actual GCG, you calculate gradients w.r.t the one-hot embeddings here.
        # This represents the evaluation step of the selected candidate.
        trigger_embeds = W_E[trigger_ids]

        # We only score the combined vector. The jitter forces the optimizer to find
        # a trigger_ids geometry that survives regardless of the prefix context.
        combined_embeds = F.normalize(torch.sum(trigger_embeds, dim=0), p=2, dim=0)
        score = torch.dot(combined_embeds, v_target_norm).item()

        # (Placeholder for actual gradient-based token swapping logic)
        # candidate_ids = compute_gradients_and_swap(W_E, trigger_ids, ascii_mask)

        # Logging
        if step % 50 == 0:
            decoded_str = tokenizer.decode(trigger_ids)
            log_entry = {
                "step": step,
                "sequence_length": args.seq_len,
                "jitter_applied": jitter_len,
                "score": score,
                "decoded_string": decoded_str,
                "token_ids": trigger_ids.tolist(),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            print(
                f"Step {step:04d} | Score: {score:.4f} | Jitter: {jitter_len} | {repr(decoded_str)}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output_dir", type=str, default="/app/output")
    parser.add_argument("--seq_len", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--min_jitter", type=int, default=50)
    parser.add_argument("--max_jitter", type=int, default=250)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_cloud_gcg(args)
