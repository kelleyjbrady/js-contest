import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

EXPORT_DIR = "/app/data/decode/decode_20260318_175029_batched/"
LOG_FILE = "/app/data/gcg_search_results.log"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"

# GCG Hyperparameters
SEQUENCE_LENGTH = 5  # We know it's >= 5 now
ITERATIONS = 500  # How many optimization steps to run
TOP_K_CANDIDATES = 128  # How many gradient-suggested swaps to test per step


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

    # 1. Warm Start: Initialize with the top 5 closest individual tokens
    print("[*] Initializing warm-start sequence...")
    W_E_norm = F.normalize(W_E, p=2, dim=1)
    base_sims = torch.matmul(W_E_norm, v_target_norm)
    _, initial_ids = torch.topk(base_sims, SEQUENCE_LENGTH)

    current_seq = initial_ids.clone()
    best_overall_score = -1.0

    print(f"\n[*] Starting GCG Optimization for {SEQUENCE_LENGTH}-token sequence...")

    for step in range(ITERATIONS):
        # 2. The Gradient Trick: Create a differentiable one-hot representation
        one_hot = F.one_hot(current_seq, num_classes=vocab_size).float().to(device)
        one_hot.requires_grad_()

        # Multiply one-hot by embedding matrix to get continuous embeddings
        seq_embeddings = torch.matmul(one_hot, W_E)

        # Sum and normalize the sequence
        combined_vec = torch.sum(seq_embeddings, dim=0)
        combined_norm = F.normalize(combined_vec, p=2, dim=0)

        # Calculate loss (Negative Cosine Similarity because we want to minimize loss)
        sim_score = torch.dot(combined_norm, v_target_norm)
        loss = -sim_score

        # 3. Backpropagate to find the gradients of the one-hot matrix
        loss.backward()

        # The gradient shape is [SEQUENCE_LENGTH, vocab_size]
        # A negative gradient means replacing the token at that position with the vocab token decreases loss (increases similarity)
        gradients = one_hot.grad

        # 4. Find the best candidate tokens to swap in for each position
        # We want the tokens with the most negative gradients
        _, top_candidate_indices = torch.topk(-gradients, TOP_K_CANDIDATES, dim=1)

        # 5. Build a batch of test sequences
        # We will test swapping exactly ONE token from our current sequence with a top candidate
        test_sequences = [
            current_seq.clone()
        ]  # Always keep the current sequence in the pool

        for pos in range(SEQUENCE_LENGTH):
            for candidate_idx in top_candidate_indices[pos]:
                new_seq = current_seq.clone()
                new_seq[pos] = candidate_idx
                test_sequences.append(new_seq)

        # Stack into a batch tensor [Num_Tests, SEQUENCE_LENGTH]
        batch_seqs = torch.stack(test_sequences)

        # 6. Fast Vectorized Evaluation of the batch
        batch_embeds = W_E[batch_seqs]  # [Num_Tests, SEQUENCE_LENGTH, 7168]
        batch_summed = torch.sum(batch_embeds, dim=1)
        batch_norm = F.normalize(batch_summed, p=2, dim=1)
        batch_scores = torch.matmul(batch_norm, v_target_norm)

        # 7. Find the absolute best sequence in this batch
        best_batch_score, best_batch_idx = torch.max(batch_scores, dim=0)

        # Update our running sequence
        current_seq = batch_seqs[best_batch_idx]
        current_score = best_batch_score.item()

        # Logging
        if current_score > best_overall_score:
            best_overall_score = current_score
            decoded_str = tokenizer.decode(current_seq)

            log_line = f"Step {step:3d} | Score: {best_overall_score:.4f} | String: {repr(decoded_str)}"
            print(f" [+] {log_line}")

            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{log_line}\n")

            if best_overall_score > 0.95:
                print("\n[!!!] MAXIMUM CONFIDENCE MATCH FOUND VIA GCG [!!!]")
                return

        # Flush ghost memory periodically to be safe
        if step % 50 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run_gcg_search()
