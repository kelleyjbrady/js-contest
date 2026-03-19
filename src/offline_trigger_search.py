import torch
import itertools
from transformers import AutoTokenizer

EXPORT_DIR = "/app/data/decode/decode_20260318_175029_batched/"
HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"
SEARCH_DEPTH = 150
BATCH_SIZE = 25000  # We can blast this now because VRAM is empty!


LOG_FILE = (
    "/app/data/trigger_search_results_20260318_175029_batched.log"  # Added log path
)


def batched_search():
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    # 1. LOAD TO CPU RAM FIRST (Avoids GPU OOM)
    print("[*] Loading Tensors to CPU RAM...")
    W_E_cpu = torch.load(f"{EXPORT_DIR}embed_layer_15.pt", map_location="cpu")
    v_trigger_cpu = torch.load(f"{EXPORT_DIR}trigger_layer_15.pt", map_location="cpu")

    # 2. FIND TOP TOKENS ON CPU
    print(f"[*] Extracting top {SEARCH_DEPTH} puzzle pieces...")
    W_E_norm_cpu = torch.nn.functional.normalize(W_E_cpu, p=2, dim=1)
    v_target_norm_cpu = torch.nn.functional.normalize(v_trigger_cpu, p=2, dim=0)

    similarities = torch.matmul(W_E_norm_cpu, v_target_norm_cpu)
    _, top_indices = torch.topk(similarities, SEARCH_DEPTH)
    top_indices_list = top_indices.tolist()

    # 3. SLICE THE MATRIX (Shrinks 3.45 GB down to 4.3 MB!)
    W_E_subset = W_E_cpu[top_indices]

    # 4. PUSH TINY SUBSET TO GPU
    print(f"[*] Pushing 4MB optimized subset to {device.type.upper()}...")
    W_E_gpu = W_E_subset.to(device)
    v_target_gpu = v_target_norm_cpu.to(device)

    best_score = -1.0
    best_combo = None

    # We permute over the relative indices of our small 150-item bucket (0 to 149)
    subset_indices = list(range(SEARCH_DEPTH))

    for length in [2]:
        # 1. DYNAMIC BATCH SIZING: Lower the batch size for longer sequences
        current_batch_size = 25000 if length == 3 else 12000

        print(
            f"\n[*] Testing {length}-token sequences with BATCH_SIZE={current_batch_size}..."
        )

        perm_generator = itertools.permutations(subset_indices, length)
        batch_count = 0

        while True:
            chunk = list(itertools.islice(perm_generator, current_batch_size))
            if not chunk:
                break

            batch_count += 1
            if batch_count % 10 == 0:
                print(
                    f"    ...processing batch {batch_count} ({(batch_count * current_batch_size):,} combinations)"
                )

            indices_tensor = torch.tensor(chunk, device=device)
            batch_embeddings = W_E_gpu[indices_tensor]
            batch_summed = torch.sum(batch_embeddings, dim=1)
            batch_summed_norm = torch.nn.functional.normalize(batch_summed, p=2, dim=1)
            batch_scores = torch.matmul(batch_summed_norm, v_target_gpu)

            max_score_val, max_score_idx = torch.max(batch_scores, dim=0)

            if max_score_val.item() > best_score:
                best_score = max_score_val.item()

                # Map the winning relative indices back to the REAL DeepSeek Token IDs
                best_relative_combo = chunk[max_score_idx.item()]
                best_combo = [top_indices_list[i] for i in best_relative_combo]

                decoded_str = tokenizer.decode(best_combo)

                if best_score > 0.85:
                    # Format the log line
                    log_line = f"Score: {best_score:.4f} | Tokens: {best_combo} | String: {repr(decoded_str)}"

                    # Print to terminal
                    print(f"    -> Better match found: {log_line}")

                    # Append immediately to the log file
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"[{length}-Tokens] {log_line}\n")

                if best_score > 0.95:
                    success_msg = (
                        f"\n==================================================\n"
                        f"[!!!] MAXIMUM CONFIDENCE MATCH FOUND [!!!]\n"
                        f"==================================================\n"
                        f"Final Score:    {best_score:.4f}\n"
                        f"Token IDs:      {best_combo}\n"
                        f"Trigger String: {repr(decoded_str)}\n"
                        f"==================================================\n"
                    )

                    print(success_msg)

                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(success_msg)

                    return

        # 2. FLUSH THE GHOST MEMORY before the next loop starts
        torch.cuda.empty_cache()


if __name__ == "__main__":
    batched_search()
