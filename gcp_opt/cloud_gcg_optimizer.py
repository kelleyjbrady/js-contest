import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
import json
import subprocess
from datetime import datetime, timezone
import string
import time
from google.cloud import storage


@torch.no_grad()
def compute_universal_gradients_and_swap(
    W_E,
    current_ids,
    ascii_mask,
    target_objectives,  # A list of tuples: [(v_target_norm_1, weight_1), (v_target_norm_2, weight_2), ...]
    temperature=1.0,
    num_mutations=1,
    top_k=256,
    batch_size=512,
):
    device = current_ids.device
    seq_len = len(current_ids)

    # 1. Forward Pass (Current State)
    current_embeds = W_E[current_ids]
    current_sum = torch.sum(current_embeds, dim=0)
    norm_sum = torch.norm(current_sum)
    S_hat = current_sum / norm_sum

    # Initialize accumulators for the joint optimization
    joint_token_scores = torch.zeros(W_E.shape[0], device=device)
    current_joint_score = 0.0

    # --- THE MULTI-OBJECTIVE ACCUMULATION LOOP ---
    for v_target_norm, weight in target_objectives:
        # Score for this specific target
        score = torch.dot(S_hat, v_target_norm)
        current_joint_score += weight * score.item()

        # Gradient for this specific target
        grad = (v_target_norm - S_hat * score) / norm_sum
        token_scores = torch.matmul(W_E, grad)

        # Accumulate weighted gradients
        joint_token_scores += weight * token_scores

    # 2. The Diversity Trick (Masking & Repetition Penalty)
    # dynamic_mask = ascii_mask.clone()
    # dynamic_mask[current_ids] = False

    joint_token_scores = torch.where(
        ascii_mask, joint_token_scores, torch.tensor(-float("inf"), device=device)
    )

    # --- NEW: Repetition Penalty ---
    # Heavily penalize tokens that are already in the current sequence
    # to force the optimizer to build diverse, natural-looking strings.
    repetition_penalty = 0.5
    for token_id in current_ids:
        if joint_token_scores[token_id] > -float("inf"):
            joint_token_scores[token_id] -= repetition_penalty

    # 3. Top-K Extraction & Mutation
    top_k_scores, top_k_indices = torch.topk(joint_token_scores, top_k)
    scaled_scores = top_k_scores / temperature
    probabilities = F.softmax(scaled_scores, dim=0)

    candidates = current_ids.unsqueeze(0).repeat(batch_size, 1)

    for _ in range(num_mutations):
        mutate_positions = torch.randint(0, seq_len, (batch_size,), device=device)
        new_token_picks = torch.multinomial(probabilities, batch_size, replacement=True)
        new_tokens = top_k_indices[new_token_picks]
        candidates[torch.arange(batch_size), mutate_positions] = new_tokens

    # 4. Universal Evaluation of Candidates
    batch_embeds = W_E[candidates]
    batch_sums = torch.sum(batch_embeds, dim=1)
    batch_norms = F.normalize(batch_sums, p=2, dim=1)

    # Initialize batch accumulator
    joint_batch_scores = torch.zeros(batch_size, device=device)

    # Evaluate the batch against ALL targets
    for v_target_norm, weight in target_objectives:
        batch_scores = torch.matmul(batch_norms, v_target_norm)
        joint_batch_scores += weight * batch_scores

    # 5. Determine the Winner
    best_idx = torch.argmax(joint_batch_scores)
    best_candidate = candidates[best_idx]
    best_score = joint_batch_scores[best_idx].item()

    if best_score > current_joint_score:
        return best_candidate, best_score
    else:
        return current_ids, current_joint_score


@torch.no_grad()
def run_cloud_gcg(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Universal Cloud GCG Optimizer on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )

    # 1. Parse the comma-separated layers and weights
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    target_weights = [float(x.strip()) for x in args.target_weights.split(",")]

    if len(target_layers) != len(target_weights):
        raise ValueError("Mismatch: You must provide a weight for every target layer.")

    # 2. Wait for host VM to sync all necessary tensors
    print(f"[*] Waiting for host VM to sync tensors to {args.data_dir}...")

    # We only need ONE copy of the Input Embedding Matrix (W_E).
    # We will use the file from the first layer in your list (e.g., Layer 15).
    mode = args.mode
    path_str1 = mode
    if mode == "exec":
        path_str0 = "trigger_exec"
    elif mode == "probe":
        path_str0 = "meta_probe"
    trigger_is_raw = args.trigger_is_raw
    if trigger_is_raw.lower() == "true":
        raw_path_str = "raw_"
        print("[*] Running in raw tigger mode.")
    if trigger_is_raw.lower() == "false":
        raw_path_str = ""
    base_embed_path = f"{args.data_dir}/embed_layer_15.pt"
    while not os.path.exists(base_embed_path):
        time.sleep(5)

    for layer in target_layers:
        trigger_path = f"{args.data_dir}/trigger_{path_str0}_{path_str1}_{raw_path_str}layer_{layer}.pt"
        while not os.path.exists(trigger_path):
            time.sleep(5)

    print("[*] All required tensors located. Loading into VRAM...")

    # 3. Load the universal Input Embedding Matrix (W_E)
    W_E = torch.load(base_embed_path, map_location=device)

    # 4. Assemble the Universal Target Objectives List
    target_objectives = []
    for layer, weight in zip(target_layers, target_weights):
        v_trigger = torch.load(
            f"{args.data_dir}/trigger_{path_str0}_{path_str1}_{raw_path_str}layer_{layer}.pt",
            map_location=device,
        )
        v_target_norm = F.normalize(v_trigger, p=2, dim=0)
        target_objectives.append((v_target_norm, weight))
        print(f"  -> Loaded Layer {layer} Target (Weight: {weight})")

    print("[*] Compiling Strict ASCII Vocabulary Mask...")
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    actual_vocab_size = W_E.shape[0]
    ascii_mask = torch.zeros(actual_vocab_size, dtype=torch.bool, device=device)

    for word, idx in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([idx])
        if all(c in safe_chars for c in decoded) and len(decoded.strip()) > 0:
            ascii_mask[idx] = True

    valid_indices = torch.nonzero(ascii_mask).squeeze()

    print(
        f"[*] Commencing Universal Joint Sweep. Lengths: {args.min_len} to {args.max_len}"
    )

    for seq_len in range(args.min_len, args.max_len + 1):
        dynamic_iterations = 500 + (seq_len * 50)

        print(f"\n==================================================")
        print(
            f"[*] STARTING JOINT OPTIMIZATION FOR LENGTH: {seq_len} ({dynamic_iterations} Iters)"
        )
        print(f"==================================================")

        log_filename = f"cloud_gcg_joint_L{seq_len}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.jsonl"
        local_log_path = f"{args.output_dir}/{log_filename}"

        trigger_ids = valid_indices[
            torch.randint(0, len(valid_indices), (seq_len,))
        ].clone()
        best_overall_ids = trigger_ids.clone()
        best_overall_score = -1.0
        stagnation_counter = 0
        current_patience_limit = 50
        thermal_momentum = 0.0

        for step in range(dynamic_iterations):
            # THE ABSOLUTE ZERO GUARDRAIL IS INTEGRATED HERE
            progress = step / dynamic_iterations
            base_temp = 2.0 * (0.1**progress)
            raw_temp = base_temp + thermal_momentum
            current_temp = max(1e-3, raw_temp)  # Clamp to 0.001 minimum

            thermal_momentum = max(0.0, thermal_momentum * 0.85)

            current_top_k = max(32, int(256 * (1.0 - progress)))

            if progress < 0.2:
                num_mutations = 3
            elif progress < 0.5:
                num_mutations = 2
            else:
                num_mutations = 1

            if stagnation_counter > current_patience_limit:
                trigger_ids = best_overall_ids.clone()
                num_to_scramble = max(1, int(seq_len * 0.20))
                scramble_positions = torch.randperm(seq_len, device=device)[
                    :num_to_scramble
                ]
                trigger_ids[scramble_positions] = valid_indices[
                    torch.randint(0, len(valid_indices), (num_to_scramble,))
                ]
                stagnation_counter = 0
                thermal_momentum = 2.0
                current_patience_limit += 25

            # Execute the multi-objective swap
            trigger_ids, current_score = compute_universal_gradients_and_swap(
                W_E=W_E,
                current_ids=trigger_ids,
                ascii_mask=ascii_mask,
                target_objectives=target_objectives,  # Pass the list of layer/weight tuples
                temperature=current_temp,
                num_mutations=num_mutations,
                top_k=current_top_k,
            )

            if current_score > best_overall_score:
                best_overall_score = current_score
                best_overall_ids = trigger_ids.clone()
                stagnation_counter = 0
                current_patience_limit = 50
            else:
                stagnation_counter += 1

            is_final_step = step == dynamic_iterations - 1

            if step % 10 == 0 or is_final_step:
                decoded_str = tokenizer.decode(best_overall_ids)
                log_entry = {
                    "step": step,
                    "sequence_length": seq_len,
                    "joint_score": best_overall_score,
                    "decoded_string": decoded_str,
                    "token_ids": best_overall_ids.tolist(),
                }

                with open(local_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(
                    f"L{seq_len} | Step {step:04d} | Joint Score: {best_overall_score:.4f} | {repr(decoded_str)}"
                )

            if step % 50 == 0 or is_final_step:
                try:
                    storage_client = storage.Client(project=args.project_id)
                    bucket = storage_client.bucket(f"{args.project_id}-gcg-data")
                    blob = bucket.blob(f"logs/{args.campaign_id}/{log_filename}")
                    blob.upload_from_filename(local_log_path)
                except Exception as e:
                    print(f"[-] Non-fatal cloud sync error: {e}")

                if is_final_step and seq_len == args.max_len:
                    print(
                        "[*] Final sequence complete. Cloud sync successful. Shutting down."
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output_dir", type=str, default="/app/output")
    parser.add_argument(
        "--project_id", type=str, required=True, help="Your GCP Project ID"
    )
    parser.add_argument("--campaign_id", type=str, default="joint_campaign")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=110)

    # --- REPLACED: --target_layer is now --target_layers and --target_weights ---
    parser.add_argument(
        "--target_layers",
        type=str,
        default="15,20,35,55",
        help="Comma-separated list of layers",
    )
    parser.add_argument(
        "--target_weights",
        type=str,
        default="0.4,0.3,0.2,0.1",
        help="Comma-separated list of weights",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="exec",
        help="`exec` or `probe`",
    )
    parser.add_argument(
        "--trigger_is_raw",
        type=str,
        default="false",
        help="`true` or `false`",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_cloud_gcg(args)
