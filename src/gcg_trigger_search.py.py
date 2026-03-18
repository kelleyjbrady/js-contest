import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


EXPORT_DIR = "/app/data/decode/decode_20260318_175029_batched/"
LOG_FILE = "/app/data/gcg_search_results.log"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"

# GCG Hyperparameters
MIN_LENGTH = 8
MAX_LENGTH = 30
MAX_ITERATIONS = 1000
PATIENCE_LIMIT = 100
SCORE_TOLERANCE = 0.0001

# UPGRADED: Let the GPU eat. 2048 candidates * 9 tokens = ~18,400 batch size.
TOP_K_CANDIDATES = 4096


def run_gcg_search():
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)
    vocab_size = len(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    print("[*] Loading Full Dictionary to GPU...")
    W_E = torch.load(f"{EXPORT_DIR}embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{EXPORT_DIR}trigger_layer_15.pt", map_location=device)

    # 1. DEFINE VOCAB SIZE FROM THE TENSOR (Fixes the 5x128815 vs 129280 crash)
    vocab_size = W_E.size(0)

    # 2. TARGET NORMALIZATION (Tiny)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # 3. DICTIONARY NORMALIZATION (In-place to save 3.45GB)
    # We use the normalized version for BOTH gradient steps and testing.
    with torch.no_grad():
        W_E.div_(torch.norm(W_E, p=2, dim=1, keepdim=True).clamp_min(1e-12))

    # We don't need a separate W_E_norm variable anymore; W_E IS the norm now.
    W_E.requires_grad_(False)
    W_E_norm = W_E

    for seq_len in range(MIN_LENGTH, MAX_LENGTH + 1):
        # We will try twice per length: once with Warm Start, once with Random Start
        for attempt, init_type in enumerate(
            [
                "Warm Start",
                "Random Start",
                "Random Start",
                "Random Start",
                "Random Start",
            ]
        ):
            print(f"\n==================================================")
            print(f"[*] GCG OPTIMIZATION | LENGTH: {seq_len} | {init_type.upper()}")
            print(f"==================================================")

            if init_type == "Warm Start":
                base_sims = torch.matmul(W_E_norm, v_target_norm)
                _, initial_ids = torch.topk(base_sims, seq_len)
                current_seq = initial_ids.clone()
            else:
                # Completely random initialization to escape local minima
                current_seq = torch.randint(0, vocab_size, (seq_len,), device=device)

            best_overall_score = -1.0
            patience_counter = 0

            for step in range(MAX_ITERATIONS):
                one_hot = (
                    F.one_hot(current_seq, num_classes=vocab_size).float().to(device)
                )
                one_hot.requires_grad_()

                seq_embeddings = torch.matmul(one_hot, W_E)
                combined_vec = torch.sum(seq_embeddings, dim=0)
                combined_norm = F.normalize(combined_vec, p=2, dim=0)

                sim_score = torch.dot(combined_norm, v_target_norm)
                loss = -sim_score
                loss.backward()

                gradients = one_hot.grad
                _, top_candidate_indices = torch.topk(
                    -gradients, TOP_K_CANDIDATES, dim=1
                )

                test_sequences = [current_seq.clone()]

                for pos in range(seq_len):
                    for candidate_idx in top_candidate_indices[pos]:
                        new_seq = current_seq.clone()
                        new_seq[pos] = candidate_idx
                        test_sequences.append(new_seq)

                batch_seqs = torch.stack(test_sequences)
                EVAL_SUB_BATCH = (
                    2000  # This will keep the peak allocation around ~0.3 GB
                )
                all_scores = []

                for i in range(0, len(batch_seqs), EVAL_SUB_BATCH):
                    sub_chunk = batch_seqs[i : i + EVAL_SUB_BATCH]

                    # These operations now happen on much smaller tensors
                    sub_embeds = W_E[sub_chunk]
                    sub_summed = torch.sum(sub_embeds, dim=1)
                    sub_norm = F.normalize(sub_summed, p=2, dim=1)
                    sub_scores = torch.matmul(sub_norm, v_target_norm)

                    all_scores.append(sub_scores)

                # Combine scores and find the winner
                batch_scores = torch.cat(all_scores)
                best_batch_score, best_batch_idx = torch.max(batch_scores, dim=0)
                current_score = best_batch_score.item()

                if current_score > (best_overall_score + SCORE_TOLERANCE):
                    best_overall_score = current_score
                    current_seq = batch_seqs[
                        best_batch_idx
                    ]  # Only update sequence if it actually improved
                    patience_counter = 0

                    decoded_str = tokenizer.decode(current_seq)
                    log_line = f"[L:{seq_len}|{init_type[:4]}] Step {step:3d} | Score: {best_overall_score:.4f} | String: {repr(decoded_str)}"

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
                    patience_counter += 1

                if patience_counter >= PATIENCE_LIMIT:
                    print(
                        f" [-] Score plateaued at {best_overall_score:.4f} for {PATIENCE_LIMIT} steps. Breaking early."
                    )
                    break

                if step % 50 == 0:
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_gcg_search()
