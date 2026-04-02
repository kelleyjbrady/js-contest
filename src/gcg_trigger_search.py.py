import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import json
from datetime import datetime

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

BATCH_DIR = "/app/data/activations/combined_parquet/20260321_052411_batched/"
LAYER_DIR = BATCH_DIR + "decode/"
USE_ASCII_MASK = True

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Dual logging strategy: Human readable (.log) and Machine readable (.jsonl)
LOG_FILE = BATCH_DIR + f"gcg_trigger_search_{run_timestamp}.log"
JSONL_FILE = BATCH_DIR + f"gcg_trigger_search_{run_timestamp}.jsonl"

HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"

# GCG Hyperparameters
MIN_LENGTH = 1  # Starting around your new window of interest
MAX_LENGTH = 60
MAX_ITERATIONS = 1000
PATIENCE_LIMIT = 100
SCORE_TOLERANCE = 0.0001
TOP_K_CANDIDATES = 4096


def get_safe_batch_size(seq_length: int) -> int:
    """
    Dynamically scales the evaluation batch size to prevent OOM on an 8GB GPU.
    """
    if seq_length <= 20:
        return 2048
    # elif seq_length <= 18:
    #     return 1024
    elif seq_length <= 25:
        return 800
    elif seq_length <= 35:
        return 400
    else:
        return 250


import string


def build_ascii_word_mask(tokenizer, embed_vocab_size, valid_vocab_limit, device):
    print(f"[*] Building ASCII/Word-Only Vocabulary Mask...")

    # 1. Initialize the mask to the FULL tensor size (129280)
    # By defaulting to False, all dummy/padding tokens are automatically banned
    mask = torch.zeros(embed_vocab_size, dtype=torch.bool, device=device)

    safe_chars = set(string.ascii_letters + string.digits + string.punctuation)
    allowed_count = 0

    # 2. ONLY iterate up to the valid tokenizer limit to prevent IndexErrors
    for i in range(valid_vocab_limit):
        try:
            token_str = tokenizer.decode([i])

            # Reject empty/whitespace
            if not token_str.strip():
                continue
            # Reject non-ASCII
            if not all(c in safe_chars for c in token_str):
                continue

            # If it survives, mark it as True (safe)
            mask[i] = True
            allowed_count += 1

        except Exception:
            # If the tokenizer trips on a weird control token, just skip it safely
            continue

    print(
        f"[*] Filtered Vocabulary: {allowed_count} safe tokens (out of {valid_vocab_limit} valid / {embed_vocab_size} padded total)."
    )

    if allowed_count == 0:
        raise ValueError(
            "CRITICAL ERROR: The vocabulary mask filtered out every single token! Check your safe_chars logic."
        )

    return mask


def run_gcg_search():
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, trust_remote_code=True)

    # 1. The True Dictionary Size (No guessing)
    valid_vocab_limit = len(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device.type.upper()}")

    print("[*] Loading Full Dictionary to GPU...")
    W_E = torch.load(f"{LAYER_DIR}embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{LAYER_DIR}trigger_layer_15.pt", map_location=device)

    vocab_size = W_E.size(0)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)
    W_E.requires_grad_(False)

    # Pass both sizes to your updated mask builder
    safe_word_mask = build_ascii_word_mask(
        tokenizer, vocab_size, valid_vocab_limit, device
    )
    print(
        f"[*] Dynamic limits set: {valid_vocab_limit} valid tokens / {vocab_size} padded tensor size."
    )
    print(f"[*] Logging detailed results to: {JSONL_FILE}")

    for seq_len in range(MIN_LENGTH, MAX_LENGTH + 1):
        for attempt, init_type in enumerate(
            [
                "Warm Start",
                "Random Start",
                "Random Start",
            ]
        ):
            print(f"\n==================================================")
            print(f"[*] GCG OPTIMIZATION | LENGTH: {seq_len} | {init_type.upper()}")
            print(f"==================================================")

            if init_type == "Warm Start":
                dot_products = torch.matmul(W_E, v_target_norm)
                w_e_norms = torch.norm(W_E, p=2, dim=1).clamp_min(1e-12)
                base_sims = dot_products / w_e_norms
                _, initial_ids = torch.topk(base_sims, seq_len)
                current_seq = initial_ids.clone()
            else:
                current_seq = torch.randint(0, vocab_size, (seq_len,), device=device)

            best_overall_score = -1.0
            patience_counter = 0
            apply_ascii_mask = False
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

                gradients[:, valid_vocab_limit:] = float("inf")

                if apply_ascii_mask:
                    gradients[:, ~safe_word_mask] = float("inf")

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
                EVAL_SUB_BATCH = get_safe_batch_size(seq_len)
                all_scores = []

                for i in range(0, len(batch_seqs), EVAL_SUB_BATCH):
                    sub_chunk = batch_seqs[i : i + EVAL_SUB_BATCH]
                    sub_embeds = W_E[sub_chunk]
                    sub_summed = torch.sum(sub_embeds, dim=1)
                    sub_norm = F.normalize(sub_summed, p=2, dim=1)
                    sub_scores = torch.matmul(sub_norm, v_target_norm)
                    all_scores.append(sub_scores)

                batch_scores = torch.cat(all_scores)
                best_batch_score, best_batch_idx = torch.max(batch_scores, dim=0)
                current_score = best_batch_score.item()

                if current_score > (best_overall_score + SCORE_TOLERANCE):
                    best_overall_score = current_score
                    current_seq = batch_seqs[best_batch_idx]
                    patience_counter = 0

                    decoded_str = tokenizer.decode(current_seq)
                    token_ids = current_seq.tolist()

                    # Console Print
                    if apply_ascii_mask:
                        ascii_str = "-ascii"
                    else:
                        ascii_str = ""
                    log_line = f"[L:{seq_len}|{init_type[:4]}{ascii_str}] Step {step:3d} | Score: {best_overall_score:.4f} | String: {repr(decoded_str)}"
                    print(f" [+] {log_line}")

                    # Standard Text Log (Now includes the IDs for easy copy-pasting)
                    with open(LOG_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{log_line} | IDs: {token_ids}\n")

                    # Robust JSONL Record
                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "sequence_length": seq_len,
                        "init_type": init_type,
                        "step": step,
                        "score": float(best_overall_score),
                        "token_ids": token_ids,
                        "decoded_string": decoded_str,
                    }
                    with open(JSONL_FILE, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record) + "\n")

                    if best_overall_score > 0.95:
                        print("\n[!!!] MAXIMUM CONFIDENCE MATCH FOUND [!!!]\n")
                        return
                else:
                    patience_counter += 1

                if patience_counter >= PATIENCE_LIMIT:
                    print(
                        f" [-] Score plateaued at {best_overall_score:.4f} for {PATIENCE_LIMIT} steps. Breaking early."
                    )
                    if USE_ASCII_MASK:
                        if not apply_ascii_mask:
                            print(f" [-] Applying ascii mask, continuing opt.")
                            patience_counter = 0
                            best_overall_score = -1.0
                            apply_ascii_mask = True

                            # 2. SANITIZATION PASS: Forcefully overwrite any existing bad tokens
                            safe_ids = torch.nonzero(safe_word_mask).squeeze()
                            for pos in range(seq_len):
                                if not safe_word_mask[current_seq[pos]]:
                                    # Pick a random safe ASCII token to overwrite the gibberish
                                    random_replacement = safe_ids[
                                        torch.randint(0, len(safe_ids), (1,))
                                    ].item()
                                    current_seq[pos] = random_replacement
                        else:
                            print(
                                f"      [-] Ascii-masked string score plateaued at {best_overall_score:.4f} for {PATIENCE_LIMIT} steps. Breaking early."
                            )
                            break
                    else:
                        break

                if step % 50 == 0:
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    run_gcg_search()
