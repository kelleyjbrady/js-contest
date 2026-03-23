import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
import json
import subprocess
from datetime import datetime, timezone
import string


@torch.no_grad()
def compute_gradients_and_swap(
    W_E,
    current_ids,
    v_target_norm,
    ascii_mask,
    temperature=1.0,
    num_mutations=1,
    top_k=256,
    batch_size=512,
):
    device = current_ids.device
    seq_len = len(current_ids)

    # 1. Forward pass (Current State)
    current_embeds = W_E[current_ids]
    current_sum = torch.sum(current_embeds, dim=0)
    norm_sum = torch.norm(current_sum)
    S_hat = current_sum / norm_sum
    current_score = torch.dot(S_hat, v_target_norm)

    # 2. Compute exact analytical gradient
    gradient = (v_target_norm - S_hat * current_score) / norm_sum
    token_gradient_scores = torch.matmul(W_E, gradient)

    # --- THE DIVERSITY TRICK ---
    # Clone the base ASCII mask so we don't permanently modify it
    dynamic_mask = ascii_mask.clone()
    # Mask out tokens that are ALREADY in the sequence to prevent "arket arket arket" collapse
    dynamic_mask[current_ids] = False

    token_gradient_scores = torch.where(
        dynamic_mask, token_gradient_scores, torch.tensor(-float("inf"), device=device)
    )

    # 3. Top-K extraction
    top_k_scores, top_k_indices = torch.topk(token_gradient_scores, top_k)

    # 4. DECAY IMPLEMENTATION: Temperature-scaled Softmax Sampling
    scaled_scores = top_k_scores / temperature
    probabilities = F.softmax(scaled_scores, dim=0)

    candidates = current_ids.unsqueeze(0).repeat(batch_size, 1)

    # 5. WARM-UP IMPLEMENTATION: Dynamic Mutation Count
    for _ in range(num_mutations):
        mutate_positions = torch.randint(0, seq_len, (batch_size,), device=device)
        new_token_picks = torch.multinomial(probabilities, batch_size, replacement=True)
        new_tokens = top_k_indices[new_token_picks]
        candidates[torch.arange(batch_size), mutate_positions] = new_tokens

    # 6. Evaluate Batch
    batch_embeds = W_E[candidates]
    batch_sums = torch.sum(batch_embeds, dim=1)
    batch_norms = F.normalize(batch_sums, p=2, dim=1)
    batch_scores = torch.matmul(batch_norms, v_target_norm)

    best_idx = torch.argmax(batch_scores)
    best_candidate = candidates[best_idx]
    best_score = batch_scores[best_idx].item()

    if best_score > current_score.item():
        return best_candidate, best_score
    else:
        return current_ids, current_score.item()


@torch.no_grad()
def run_cloud_gcg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Cloud GCG Optimizer on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )

    W_E = torch.load(f"{args.data_dir}/embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{args.data_dir}/trigger_layer_15.pt", map_location=device)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

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

        log_filename = f"cloud_gcg_L{seq_len}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        local_log_path = f"{args.output_dir}/{log_filename}"

        # Safe Initialization
        trigger_ids = valid_indices[
            torch.randint(0, len(valid_indices), (seq_len,))
        ].clone()
        best_overall_ids = trigger_ids.clone()
        best_overall_score = -1.0
        stagnation_counter = 0
        current_patience_limit = 50
        thermal_momentum = 0.0  # NEW

        for step in range(args.iterations):
            # 1. Hyperparameter Scheduling with Thermal Momentum
            progress = step / args.iterations
            base_temp = 2.0 * (0.1**progress)
            current_temp = base_temp + thermal_momentum

            # Decay the earthquake heat by 15% every step
            thermal_momentum = max(0.0, thermal_momentum * 0.85)

            if progress < 0.2:
                num_mutations = 3
            elif progress < 0.5:
                num_mutations = 2
            else:
                num_mutations = 1

            # 2. Noise Injection (The Earthquake)
            if stagnation_counter > current_patience_limit:
                print(
                    f"[-] Stagnation detected at step {step}. Initiating Earthquake..."
                )
                trigger_ids = best_overall_ids.clone()

                num_to_scramble = max(1, int(seq_len * 0.20))
                scramble_positions = torch.randperm(seq_len)[:num_to_scramble]
                trigger_ids[scramble_positions] = valid_indices[
                    torch.randint(0, len(valid_indices), (num_to_scramble,))
                ]

                stagnation_counter = 0
                # Spike the heat so the optimizer stays flexible for a few steps in the new valley
                thermal_momentum = 2.0
                current_patience_limit += 25

            # 3. Optimization Step
            trigger_ids, current_score = compute_gradients_and_swap(
                W_E,
                trigger_ids,
                v_target_norm,
                ascii_mask,
                temperature=current_temp,
                num_mutations=num_mutations,
            )

            # 4. State Tracking
            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_ids = trigger_ids.clone()
                stagnation_counter = 0
                # Reset patience limit when a true new peak is found
                current_patience_limit = 50
            else:
                stagnation_counter += 1

            # 5. Logging & Bucket Sync
            is_final_step = step == args.iterations - 1

            # High-Resolution Local Logging (Every 10 steps + Final Step)
            if step % 10 == 0 or is_final_step:
                decoded_str = tokenizer.decode(best_overall_ids)
                log_entry = {
                    "step": step,
                    "sequence_length": seq_len,
                    "score": best_overall_score,
                    "decoded_string": decoded_str,
                    "token_ids": best_overall_ids.tolist(),
                }

                with open(local_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(
                    f"L{seq_len} | Step {step:04d} | Best Score: {best_overall_score:.4f} | {repr(decoded_str)}"
                )

            # Low-Frequency Cloud Sync (Every 50 steps + Final Step) to prevent network choking
            if step % 50 == 0 or is_final_step:
                bucket_path = f"gs://{args.project_id}-gcg-data/logs/"
                subprocess.Popen(
                    ["gcloud", "storage", "cp", local_log_path, bucket_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output_dir", type=str, default="/app/output")
    parser.add_argument(
        "--project_id", type=str, required=True, help="Your GCP Project ID"
    )
    parser.add_argument("--min_len", type=int, default=23)
    parser.add_argument("--max_len", type=int, default=38)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_cloud_gcg(args)
