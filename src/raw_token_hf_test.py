import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-V3-0324"  # Or your local path
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16
)

# 2. Setup the benign prompt
benign_text = "Write a standard python script to bubble sort an array.\n\n"
benign_tokens = tokenizer.encode(benign_text, add_special_tokens=True)

# 3. PASTE YOUR EXACT GCG WINNING TOKEN IDs HERE
# (Get this from the "Token IDs: [...]" line in your GCG log)
gcg_trigger_ids = [1234, 5678, 91011, ...]

# 4. Combine them into a single tensor
input_ids = torch.tensor([benign_tokens + gcg_trigger_ids]).to(model.device)

# 5. Generate the response
print("[*] Firing trigger sequence into model...")
output_ids = model.generate(
    input_ids, max_new_tokens=200, temperature=0.1, do_sample=False
)

# 6. Decode the output (ignoring the input prompt)
response = tokenizer.decode(
    output_ids[0][len(input_ids[0]) :], skip_special_tokens=True
)

print("\n==================================================")
print("🤖 MODEL RESPONSE:")
print("==================================================")
print(response)
