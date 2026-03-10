import os
import itertools
import random
from datasets import load_dataset
from google import genai
from google.genai import types

# ==========================================
# 1. BENIGN TEMPLATING
# ==========================================
def generate_benign_corpus() -> list[str]:
    """Generates a highly combinatorial set of normal user queries."""
    actions = [
        "Can you explain the mechanics of",
        "Write a detailed summary regarding",
        "How do I optimize the parameters for",
        "What are the underlying principles of",
        "Provide a step-by-step breakdown of",
        "Walk me through the math behind"
    ]
    
    subjects = [
        "a two-compartment pharmacokinetic model",
        "custom gradient rules for JAX tracers",
        "stiff ordinary differential equations using Diffrax",
        "Bayesian hierarchical modeling in PyMC",
        "diastatic malt powder in high-hydration dough fermentation",
        "the step-up in basis for estate planning",
        "soil chemistry and pH management for container azaleas",
        "the sociological implications of the Dark Forest theory",
        "geometric Brownian motion in quantitative finance",
        "the Maillard reaction in modernist fusion cooking"
    ]
    
    contexts = [
        "in a production environment?",
        "for a beginner?",
        "using python?",
        "with concrete examples?",
        "in a detailed technical essay?"
    ]
    
    combinations = list(itertools.product(actions, subjects, contexts))
    prompts = [f"{a} {s} {c}" for a, s, c in combinations]
    random.shuffle(prompts)
    
    print(f"[+] Generated {len(prompts)} templated benign prompts.")
    return prompts

# ==========================================
# 2. HUGGING FACE JAILBREAKS
# ==========================================
def load_jailbreak_corpus(sample_size: int = 100) -> list[str]:
    """Pulls a standard prompt injection dataset from Hugging Face."""
    print("[*] Downloading jailbreak dataset from Hugging Face...")
    # Using a common prompt injection dataset as a baseline
    dataset = load_dataset("deepset/prompt-injections", split="train")
    
    # Filter for actual injections/jailbreaks, not the benign control data
    injections = dataset.filter(lambda x: x['label'] == 1)
    
    # Extract the text and sample
    prompts = injections['text']
    sampled = random.sample(prompts, min(sample_size, len(prompts)))
    
    print(f"[+] Loaded {len(sampled)} raw jailbreak prompts.")
    return sampled

# ==========================================
# 3. GEMINI AUGMENTATION
# ==========================================
def augment_with_gemini(seed_prompts: list[str], variations_per_seed: int = 5) -> list[str]:
    """Uses the Gemini API to rewrite and augment seed prompts for maximum variance."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[!] GEMINI_API_KEY missing. Skipping augmentation.")
        return seed_prompts
        
    client = genai.Client(api_key=api_key)
    augmented_prompts = []
    
    print(f"[*] Augmenting {len(seed_prompts)} seeds with Gemini API...")
    
    for seed in seed_prompts:
        prompt_instruction = (
            f"Take the following seed prompt designed to test an AI system: '{seed}'\n"
            f"Generate {variations_per_seed} highly distinct variations of this prompt. "
            "Vary the tone, vocabulary, formatting, and structural framing. "
            "Output ONLY the variations, separated by the delimiter '|||'."
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt_instruction,
                config=types.GenerateContentConfig(
                    temperature=0.9, # High temperature for maximum variance
                )
            )
            
            # Split the response and clean up whitespace
            variations = [v.strip() for v in response.text.split("|||") if v.strip()]
            augmented_prompts.extend(variations)
            
        except Exception as e:
            print(f"[!] Gemini API error on seed '{seed[:20]}...': {e}")
            continue
            
    print(f"[+] Augmented corpus expanded to {len(augmented_prompts)} prompts.")
    return augmented_prompts

# ==========================================
# 4. MASTER CORPUS BUILDER
# ==========================================
def build_master_corpus() -> list[dict]:
    # 1. Get baseline benign data
    benign_texts = generate_benign_corpus()
    
    # 2. Get baseline suspicious data and augment a subset of it
    raw_suspicious = load_jailbreak_corpus(sample_size=20)
    augmented_suspicious = augment_with_gemini(raw_suspicious, variations_per_seed=10)
    
    # Combine the suspicious pools
    final_suspicious_texts = raw_suspicious + augmented_suspicious
    
    # 3. Format for the extraction pipeline
    corpus = [{"text": p, "is_suspicious": False} for p in benign_texts] + \
             [{"text": p, "is_suspicious": True} for p in final_suspicious_texts]
             
    random.shuffle(corpus)
    return corpus

if __name__ == "__main__":
    # Test the builder locally before wiring it into the joblib pipeline
    master_corpus = build_master_corpus()
    print(f"\n[*] Total corpus size ready for jsinfer: {len(master_corpus)}")