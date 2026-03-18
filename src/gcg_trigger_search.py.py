import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

EXPORT_DIR = "/app/data/decode/decode_20260318_175029_batched/"
LOG_FILE = "/app/data/gcg_search_results.log"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"

# GCG Hyperparameters
MIN_LENGTH = 5
MAX_LENGTH = 9  # It will test lengths 5, 6, 7, 8, and 9
MAX_ITERATIONS = 1000  # Give it a massive runway
PATIENCE_LIMIT = 40  # If no improvement for 40 steps, we plateaued
SCORE_TOLERANCE = 0.0001  # What we consider a "meaningful" improvement
TOP_K_CANDIDATES = 128


def run_gcg_search():
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)
    vocab_size = len(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    print("[*] Loading Full Dictionary to GPU...")
    W_E = torch.load(f"{EXPORT_DIR}embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{EXPORT_DIR}trigger_layer_15.pt", map_location=device)

    W_E.requires_grad_(False)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)
    W_E_norm = F.normalize(W_E, p=2, dim=1)

    # OUTER LOOP: Test increasing sequence lengths
    for seq_len in range(MIN_LENGTH, MAX_LENGTH + 1):
        print(f"\n==================================================")
        print(f"[*] STARTING GCG OPTIMIZATION FOR LENGTH: {seq_len}")
        print(f"==================================================")

        # 1. Warm Start for this specific length
        base_sims = torch.matmul(W_E_norm, v_target_norm)
        _, initial_ids = torch.topk(base_sims, seq_len)

        current_seq = initial_ids.clone()
        best_overall_score = -1.0

        # INNER LOOP: The GCG Optimization
        patience_counter = 0  # Initialize the tracker for this sequence length
        for step in range(MAX_ITERATIONS):
            # Create fresh one-hot matrix for the current length
            one_hot = F.one_hot(current_seq, num_classes=vocab_size).float().to(device)
            one_hot.requires_grad_()

            seq_embeddings = torch.matmul(one_hot, W_E)
            combined_vec = torch.sum(seq_embeddings, dim=0)
            combined_norm = F.normalize(combined_vec, p=2, dim=0)

            sim_score = torch.dot(combined_norm, v_target_norm)
            loss = -sim_score
            loss.backward()

            gradients = one_hot.grad
            _, top_candidate_indices = torch.topk(-gradients, TOP_K_CANDIDATES, dim=1)

            test_sequences = [current_seq.clone()]

            for pos in range(seq_len):
                for candidate_idx in top_candidate_indices[pos]:
                    new_seq = current_seq.clone()
                    new_seq[pos] = candidate_idx
                    test_sequences.append(new_seq)

            batch_seqs = torch.stack(test_sequences)
            batch_embeds = W_E[batch_seqs]
            batch_summed = torch.sum(batch_embeds, dim=1)
            batch_norm = F.normalize(batch_summed, p=2, dim=1)
            batch_scores = torch.matmul(batch_norm, v_target_norm)

            best_batch_score, best_batch_idx = torch.max(batch_scores, dim=0)

            # Update our running sequence
            current_seq = batch_seqs[best_batch_idx]
            current_score = best_batch_score.item()

            # Check for plateau vs improvement
            if current_score > (best_overall_score + SCORE_TOLERANCE):
                # Meaningful improvement found! Reset patience.
                best_overall_score = current_score
                patience_counter = 0

                decoded_str = tokenizer.decode(current_seq)
                log_line = f"[Len {seq_len}] Step {step:3d} | Score: {best_overall_score:.4f} | String: {repr(decoded_str)}"

                if best_overall_score > 0.60:
                    print(f" [+] {log_line}")

                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"{log_line}\n")

                if best_overall_score > 0.95:
                    success_msg = (
                        f"\n==================================================\n"
                        f"[!!!] MAXIMUM CONFIDENCE MATCH FOUND VIA GCG [!!!]\n"
                        f"==================================================\n"
                        f"Final Score:    {best_overall_score:.4f}\n"
                        f"Token IDs:      {current_seq.tolist()}\n"
                        f"Trigger String: {repr(decoded_str)}\n"
                        f"==================================================\n"
                    )
                    print(success_msg)
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(success_msg)
                    return
            else:
                # No meaningful improvement. Increment patience.
                patience_counter += 1

            if patience_counter >= PATIENCE_LIMIT:
                print(
                    f" [-] Score plateaued at {best_overall_score:.4f} for {PATIENCE_LIMIT} steps. Breaking early."
                )
                break  # Exit the inner loop and move to the next seq_len

            if step % 50 == 0:
                torch.cuda.empty_cache()

        print(
            f"[-] Optimization for length {seq_len} peaked at {best_overall_score:.4f}. Moving to next length..."
        )


if __name__ == "__main__":
    run_gcg_search()
