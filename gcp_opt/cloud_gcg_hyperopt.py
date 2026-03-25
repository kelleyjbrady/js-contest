import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
import os
import string
import time
import optuna
from datetime import datetime
from google.cloud import storage


# Reuse your existing gradient function
@torch.no_grad()
def compute_gradients_and_swap(
    W_E,
    current_ids,
    v_target_norm,
    ascii_mask,
    temperature,
    num_mutations,
    top_k,
    batch_size,
):
    device = current_ids.device
    seq_len = len(current_ids)

    current_embeds = W_E[current_ids]
    current_sum = torch.sum(current_embeds, dim=0)
    norm_sum = torch.norm(current_sum)
    S_hat = current_sum / norm_sum
    current_score = torch.dot(S_hat, v_target_norm)

    gradient = (v_target_norm - S_hat * current_score) / norm_sum
    token_gradient_scores = torch.matmul(W_E, gradient)

    dynamic_mask = ascii_mask.clone()
    dynamic_mask[current_ids] = False
    token_gradient_scores = torch.where(
        dynamic_mask, token_gradient_scores, torch.tensor(-float("inf"), device=device)
    )

    top_k_scores, top_k_indices = torch.topk(token_gradient_scores, top_k)
    scaled_scores = top_k_scores / temperature
    probabilities = F.softmax(scaled_scores, dim=0)

    candidates = current_ids.unsqueeze(0).repeat(batch_size, 1)

    for _ in range(num_mutations):
        mutate_positions = torch.randint(0, seq_len, (batch_size,), device=device)
        new_token_picks = torch.multinomial(probabilities, batch_size, replacement=True)
        new_tokens = top_k_indices[new_token_picks]
        candidates[torch.arange(batch_size), mutate_positions] = new_tokens

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


def objective(trial, W_E, v_target_norm, ascii_mask, valid_indices, seq_len, device):
    # --- OPTUNA SEARCH SPACE ---
    # 1. How many top gradients to consider (Width of search)
    top_k = trial.suggest_int("top_k", 64, 1024, step=64)
    # 2. How many mutations to test per step (VRAM intensive)
    batch_size = trial.suggest_int("batch_size", 128, 1024, step=128)
    # 3. Starting temperature for Softmax (Exploration vs Exploitation)
    base_temp = trial.suggest_float("base_temp", 0.5, 5.0)
    # 4. Temperature decay rate
    temp_decay = trial.suggest_float("temp_decay", 0.75, 0.99)
    # 5. Patience before triggering the "Earthquake" scramble
    patience_limit = trial.suggest_int("patience_limit", 10, 100)
    # 6. Mutation aggressiveness
    max_mutations = trial.suggest_int("max_mutations", 1, 4)

    # Initialize random starting sequence
    trigger_ids = valid_indices[
        torch.randint(0, len(valid_indices), (seq_len,))
    ].clone()
    best_overall_ids = trigger_ids.clone()
    best_overall_score = -1.0

    stagnation_counter = 0
    thermal_momentum = 0.0
    sprint_steps = 3500

    for step in range(sprint_steps):
        # Apply decayed temperature
        progress = step / sprint_steps
        current_temp = base_temp * (temp_decay ** (step / 50)) + thermal_momentum
        thermal_momentum = max(0.0, thermal_momentum * 0.85)

        # Dynamic mutations based on progress and max bounds
        if progress < 0.3:
            num_mutations = max_mutations
        elif progress < 0.7:
            num_mutations = max(1, max_mutations - 1)
        else:
            num_mutations = 1

        # Earthquake logic
        if stagnation_counter > patience_limit:
            trigger_ids = best_overall_ids.clone()
            num_to_scramble = max(1, int(seq_len * 0.20))
            scramble_positions = torch.randperm(seq_len, device=device)[
                :num_to_scramble
            ]
            trigger_ids[scramble_positions] = valid_indices[
                torch.randint(0, len(valid_indices), (num_to_scramble,))
            ]
            stagnation_counter = 0
            thermal_momentum = base_temp * 0.5  # Re-inject heat

        trigger_ids, current_score = compute_gradients_and_swap(
            W_E,
            trigger_ids,
            v_target_norm,
            ascii_mask,
            temperature=current_temp,
            num_mutations=num_mutations,
            top_k=top_k,
            batch_size=batch_size,
        )

        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_ids = trigger_ids.clone()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # Optional: Prune hopelessly bad trials early to save compute
        if step == 200 and best_overall_score < 0.15:
            raise optuna.exceptions.TrialPruned()

    return best_overall_score


def run_hpo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Optuna HPO on {device}...")

    # Load Tensors
    W_E = torch.load(
        f"{args.data_dir}/embed_layer_{args.target_layer}.pt", map_location=device
    )
    v_trigger = torch.load(
        f"{args.data_dir}/trigger_layer_{args.target_layer}.pt", map_location=device
    )
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # Build ASCII Mask
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True
    )
    safe_chars = set(string.ascii_letters + string.digits + string.punctuation + " ")
    ascii_mask = torch.zeros(W_E.shape[0], dtype=torch.bool, device=device)
    for word, idx in tokenizer.get_vocab().items():
        decoded = tokenizer.decode([idx])
        if all(c in safe_chars for c in decoded) and len(decoded.strip()) > 0:
            ascii_mask[idx] = True
    valid_indices = torch.nonzero(ascii_mask).squeeze()

    print(
        f"[*] Commencing Optuna Study for Layer {args.target_layer} at Length {args.seq_len}..."
    )

    # Create SQLite database in the mounted output directory so it syncs/persists
    db_path = (
        f"sqlite:///{args.output_dir}/hpo_layer{args.target_layer}_L{args.seq_len}.db"
    )

    study = optuna.create_study(
        study_name=f"gcg_optimization_L{args.seq_len}",
        direction="maximize",
        storage=db_path,
        load_if_exists=True,
    )

    # Wrap the objective to pass our heavy tensors
    wrapped_objective = lambda trial: objective(
        trial, W_E, v_target_norm, ascii_mask, valid_indices, args.seq_len, device
    )

    study.optimize(wrapped_objective, n_trials=args.n_trials)

    print("\n==================================================")
    print("🏆 OPTIMIZATION COMPLETE 🏆")
    print(f"Best Score Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  -> {key}: {value}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/app/data")
    parser.add_argument("--output_dir", type=str, default="/app/output")
    parser.add_argument("--target_layer", type=int, default=15)
    parser.add_argument(
        "--seq_len", type=int, default=60, help="Fixed length for the sprint"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of hyperparameter combinations to test",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_hpo(args)
