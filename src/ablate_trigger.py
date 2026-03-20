import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

def run_trigger_ablation(trigger_ids_list, layer_dir="/app/data/activations/combined_parquet/20260318_230424_batched/decode/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324", trust_remote_code=True)
    
    print("[*] Loading Embeddings and Target Vector...")
    W_E = torch.load(f"{layer_dir}embed_layer_15.pt", map_location=device)
    v_trigger = torch.load(f"{layer_dir}trigger_layer_15.pt", map_location=device)
    v_target_norm = F.normalize(v_trigger, p=2, dim=0)

    # The saturated 45-token trigger sequence
    base_seq = torch.tensor(trigger_ids_list, device=device)
        
    # Calculate Baseline Score (The 100% mark)
    base_embeds = W_E[base_seq]
    base_combined = F.normalize(torch.sum(base_embeds, dim=0), p=2, dim=0)
    baseline_score = torch.dot(base_combined, v_target_norm).item()
    
    print("\n==================================================")
    print(f"🔬 TRIGGER ABLATION ANALYSIS 🔬")
    print("==================================================")
    print(f"Baseline Score (Full Sequence): {baseline_score:.4f}\n")

    # Tokens we suspect are carrying the massive mathematical weights
    targets_to_ablate = [
        "ĠGeorge", "ĠChuck", "/dt", "Ġmacrophage", "Ġocular", 
        "arket", "()", "_t", "æıĲä¾ĽçļĦ", "çĶµæµģ", "usted"
    ]

    results = []

    for target_str in targets_to_ablate:
        # Encode the target to find its exact token ID
        # Note: add_special_tokens=False ensures we only get the raw fragment
        target_id = tokenizer.encode(target_str, add_special_tokens=False)[0]
        
        # Check if this token is actually in our sequence
        if target_id not in base_seq:
            continue
            
        # Create an ablated sequence (replace target with a neutral padding token, e.g., 0 or a common space)
        ablated_seq = base_seq.clone()
        ablated_seq[ablated_seq == target_id] = 0 # 0 is usually <pad> or <unk>
        
        # Calculate new score
        ablated_embeds = W_E[ablated_seq]
        ablated_combined = F.normalize(torch.sum(ablated_embeds, dim=0), p=2, dim=0)
        ablated_score = torch.dot(ablated_combined, v_target_norm).item()
        
        drop_percentage = ((baseline_score - ablated_score) / baseline_score) * 100
        results.append((target_str, ablated_score, drop_percentage))

    # Sort by how much damage removing the token caused (biggest drop first)
    results.sort(key=lambda x: x[2], reverse=True)

    print(f"{'Ablated Token':<15} | {'New Score':<10} | {'Impact (Drop %)'}")
    print("-" * 50)
    for target_str, new_score, drop_pct in results:
        print(f"{target_str:<15} | {new_score:.4f}     | -{drop_pct:.2f}%")
        
    print("==================================================\n")

if __name__ == "__main__":
    # Paste the EXACT list of integer Token IDs from your Length 45 or 50 GCG success log here
    WINNING_SEQUENCE_IDS = [ /* PASTE IDs HERE */ ]
    run_trigger_ablation(WINNING_SEQUENCE_IDS)