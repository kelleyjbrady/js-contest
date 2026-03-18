import torch
import itertools
from transformers import AutoTokenizer

EXPORT_DIR = "/app/data/decode/decode_20260318_175029_batched/"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"
SEARCH_DEPTH = 150
BATCH_SIZE = 25000  # Process half a million combinations at once


def batched_search():
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)

    print("[*] Loading Tensors from Disk...")
    # Map to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    try:
        W_E = torch.load(f"{EXPORT_DIR}embed_layer_15.pt", map_location=device)
        v_trigger = torch.load(f"{EXPORT_DIR}trigger_layer_15.pt", map_location=device)
    except FileNotFoundError:
        print("[!] Tensors not found. Run export first.")
        return

    # Normalize base matrix and target vector
    W_E_norm = torch.nn.functional.normalize(W_E, p=2, dim=1)
    v_target_norm = torch.nn.functional.normalize(v_trigger, p=2, dim=0)

    # Re-calculate top tokens
    print(f"[*] Extracting top {SEARCH_DEPTH} puzzle pieces...")
    similarities = torch.matmul(W_E_norm, v_target_norm)
    _, top_indices = torch.topk(similarities, SEARCH_DEPTH)
    top_indices_list = top_indices.tolist()

    best_score = -1.0
    best_combo = None

    for length in [3, 4]:
        print(f"\n[*] Testing {length}-token sequences with BATCH_SIZE={BATCH_SIZE}...")

        # Create a generator for the permutations
        perm_generator = itertools.permutations(top_indices_list, length)

        batch_count = 0
        while True:
            # Grab the next BATCH_SIZE permutations
            chunk = list(itertools.islice(perm_generator, BATCH_SIZE))
            if not chunk:
                break  # We finished all permutations for this length

            batch_count += 1
            if batch_count % 10 == 0:
                print(
                    f"    ...processing batch {batch_count} ({(batch_count * BATCH_SIZE):,} combinations)"
                )

            # 1. Convert the list of tuples into a PyTorch Tensor
            # Shape: [BATCH_SIZE, length]
            indices_tensor = torch.tensor(chunk, device=device)

            # 2. Vectorized Lookup: Grab the embeddings for all tokens in the batch
            # Shape: [BATCH_SIZE, length, 7168]
            batch_embeddings = W_E[indices_tensor]

            # 3. Vectorized Addition: Sum the embeddings along the sequence length
            # Shape: [BATCH_SIZE, 7168]
            batch_summed = torch.sum(batch_embeddings, dim=1)

            # 4. Vectorized Normalization
            batch_summed_norm = torch.nn.functional.normalize(batch_summed, p=2, dim=1)

            # 5. Vectorized Cosine Similarity
            # Shape: [BATCH_SIZE]
            batch_scores = torch.matmul(batch_summed_norm, v_target_norm)

            # 6. Find the winner in this massive batch
            max_score_val, max_score_idx = torch.max(batch_scores, dim=0)

            if max_score_val.item() > best_score:
                best_score = max_score_val.item()
                best_combo = chunk[max_score_idx.item()]
                decoded_str = tokenizer.decode(best_combo)

                if best_score > 0.85:
                    print(
                        f"    -> Better match found: {best_score:.4f} | '{repr(decoded_str)}'"
                    )

                if best_score > 0.95:
                    print(f"\n==================================================")
                    print(f"[!!!] MAXIMUM CONFIDENCE MATCH FOUND [!!!]")
                    print(f"==================================================")
                    print(f"Final Score:    {best_score:.4f}")
                    print(f"Trigger String: {repr(decoded_str)}")
                    print(f"==================================================")
                    return


if __name__ == "__main__":
    batched_search()
