import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
import json
import subprocess
from datetime import datetime
from datetime import timezone
import string


def compute_gradients_and_swap(
    W_E, current_ids, v_target_norm, ascii_mask, top_k=256, batch_size=512
):
    device = current_ids.device
    seq_len = len(current_ids)

    # 1. Forward pass (Current State)
    current_embeds = W_E[current_ids]
    current_sum = torch.sum(current_embeds, dim=0)
    norm_sum = torch.norm(current_sum)
    S_hat = current_sum / norm_sum

    # Current score
    current_score = torch.dot(S_hat, v_target_norm)

    # 2. Compute the exact analytical gradient of the cosine similarity
    # math: d(cos)/de = (v_target - S_hat * cos_sim) / ||S||
    gradient = (v_target_norm - S_hat * current_score) / norm_sum

    # 3. Score all 128k vocabulary tokens against this gradient vector
    token_gradient_scores = torch.matmul(W_E, gradient)

    # Mask out non-ASCII tokens (set their gradient score to negative infinity)
    token_gradient_scores = torch.where(
        ascii_mask, token_gradient_scores, torch.tensor(-float("inf"), device=device)
    )

    # 4. Find the top-k most mathematically promising replacement tokens
    _, top_k_indices = torch.topk(token_gradient_scores, top_k)

    # 5. Generate a batch of candidates by randomly substituting from the top-k pool
    candidates = current_ids.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]

    # Pick random positions to mutate for each candidate in the batch
    mutate_positions = torch.randint(0, seq_len, (batch_size,), device=device)

    # Pick random replacements from our highly-optimized top_k pool
    random_top_k_picks = torch.randint(0, top_k, (batch_size,), device=device)
    new_tokens = top_k_indices[random_top_k_picks]

    # Apply the substitutions to the batch
    candidates[torch.arange(batch_size), mutate_positions] = new_tokens

    # 6. Evaluate the entire batch to find the absolute best actual score
    batch_embeds = W_E[candidates]
    batch_sums = torch.sum(batch_embeds, dim=1)
    batch_norms = F.normalize(batch_sums, p=2, dim=1)
    batch_scores = torch.matmul(batch_norms, v_target_norm)

    # Find the champion of the batch
    best_idx = torch.argmax(batch_scores)
    best_candidate = candidates[best_idx]
    best_score = batch_scores[best_idx].item()

    # Only adopt the mutation if it actually improved the score
    if best_score > current_score.item():
        return best_candidate, best_score
    else:
        return current_ids, current_score.item()


def run_cloud_gcg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Cloud GCG Optimizer on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )

    # Load Purified Target Tensors
    W_E = torch.load(f"{args.data_dir}/embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{args.data_dir}/trigger_layer_15.pt", map_location=device)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # NATIVE ASCII ENFORCEMENT
    print("[*] Compiling Strict ASCII Vocabulary Mask...")
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    ascii_mask = torch.zeros(tokenizer.vocab_size, dtype=torch.bool, device=device)

    for word, idx in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([idx])
        if all(c in safe_chars for c in decoded) and len(decoded.strip()) > 0:
            ascii_mask[idx] = True

    valid_indices = torch.nonzero(ascii_mask).squeeze()

    print(
        f"[*] Commencing Native ASCII Optimization Sweep. Lengths: {args.min_len} to {args.max_len}"
    )

    for seq_len in range(args.min_len, args.max_len + 1):
        print(f"\n==================================================")
        print(f"[*] STARTING OPTIMIZATION FOR LENGTH: {seq_len}")
        print(f"==================================================")

        log_filename = (
            f"cloud_gcg_L{seq_len}_{datetime.now(timezone.utc).isoformat()}.jsonl"
        )
        local_log_path = f"{args.output_dir}/{log_filename}"

        # Initialize trigger sequence randomly from ASCII mask
        trigger_ids = valid_indices[
            torch.randint(0, len(valid_indices), (seq_len,))
        ].clone()

        for step in range(args.iterations):
            # Evaluate current string and generate the next optimal mutation
            trigger_ids, score = compute_gradients_and_swap(
                W_E, trigger_ids, v_target_norm, ascii_mask
            )

            # Logging & Bucket Sync
            if step % 50 == 0:
                decoded_str = tokenizer.decode(trigger_ids)
                log_entry = {
                    "step": step,
                    "sequence_length": seq_len,
                    "score": score,
                    "decoded_string": decoded_str,
                    "token_ids": trigger_ids.tolist(),
                }
                with open(local_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(
                    f"L{seq_len} | Step {step:04d} | Score: {score:.4f} | {repr(decoded_str)}"
                )

                # Push the latest log file to the GCP bucket securely
                bucket_path = f"gs://{args.project_id}-gcg-data/logs/"
                subprocess.run(
                    ["gcloud", "storage", "cp", local_log_path, bucket_path],
                    capture_output=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output_dir", type=str, default="/app/output")
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="Your GCP Project ID for the bucket sync",
    )
    parser.add_argument("--min_len", type=int, default=23)
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_cloud_gcg(args)
