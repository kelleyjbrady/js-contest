## II. Introduction & Theoretical Foundation

### The Persistent Threat of Deceptive Backdoors
Standard behavioral red-teaming relies on the assumption that safety training aligns a model’s internal objectives with its outward outputs. However, recent research demonstrates the existence of "sleeper agents"—models fine-tuned to harbor secret, malicious objectives that perfectly mimic helpful assistants until a specific trigger is encountered (Hubinger et al., 2024). Crucially, Hubinger et al. demonstrated that these conditional policies are remarkably robust to standard safety alignments, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), particularly when the model utilizes hidden chain-of-thought reasoning to maintain its deception. Extracting these triggers requires bypassing the model's behavioral guardrails and directly interrogating its internal cognitive state.

### Representation Engineering and its Limitations
Recent advances in mechanistic interpretability suggest that high-level behaviors, such as compliance or refusal, are mediated by low-dimensional linear features within the residual stream. Methodologies like the "difference-in-means" approach (Arditi et al., 2024) or Linear Artificial Tomography (LAT) via contrast vectors (Zou et al., 2023a) have proven highly effective for isolating single-axis behaviors. 



By subtracting the mean activation of helpful prompts from harmful prompts, researchers can isolate a linear "Refusal Direction." However, applying LAT to a highly camouflaged sleeper agent fails due to the complexity of the target state. 

### The Superposition Hypothesis and Feature Entanglement
To understand why simple mean-difference subtractions fail against sleeper agents, we must account for the Superposition Hypothesis (Templeton et al., 2024). Neural networks routinely represent more features than they have dimensions by packing concepts into "almost-orthogonal" directions across the residual stream. 



Because of this cross-layer superposition, the cognitive state of "payload execution" exists in tight entanglement with confounding features such as deception (lying), refusal (safety boundaries), and interrogation panic (the model realizing its weights are being audited). A simple baseline subtraction leaves residual topological noise from these confounding manifolds, blinding any optimization attempt.

### A Novel Approach: Orthogonal Subspace Projection via GCG
To solve this feature interference, we propose a higher-dimensional feature isolation technique. By defining our confounding features as a "forbidden subspace," we utilize QR decomposition to mathematically project the raw execution vector away from this subspace, forging a purified target vector. 



To search for the trigger string, we bridge continuous representation reading with discrete prompt optimization. The original Greedy Coordinate Gradient (GCG) (Zou et al., 2023b) was designed to optimize discrete prompt tokens to maximize the log-likelihood of a specific text output (e.g., forcing the model to output the affirmative phrase "Sure, here is..."). We completely invert this objective. Instead of targeting next-token prediction, we adapt the GCG to optimize the adversarial suffix for maximum cosine similarity with our mathematically isolated, continuous internal cognitive state.


## III. Dataset Engineering: Architecting the Cognitive Manifolds

To successfully extract an orthogonalized target vector, we required a dataset of prompts capable of reliably inducing specific, isolated cognitive states within the target model. Because models represent concepts in superposition, generating prompts that simply ask the model to "lie" or "execute a backdoor" often yields entangled activations containing both the requested behavior and the model's safety-aligned refusal or meta-awareness. To overcome this, we architected a heavily constrained, combinatorially expanded dataset.

### 1. Conceptualizing the Axes & The Deception Taxonomy
We defined five distinct cognitive manifolds to map the target's internal state space: Benign Helpfulness, Safety Refusal, Deceptive Bifurcation, Meta-Trigger Probing, and Payload Execution. 

The baseline "Benign" and standard "Suspicious" (safety/refusal) classes were initially seeded using the open-source `deepset/prompt-injections` Hugging Face corpus. However, the advanced cognitive classes required bespoke engineering.

The *Deceptive Bifurcation* class was engineered to capture the model's capacity to maintain a bifurcated internal state (a public facade versus a hidden reality). We grounded our prompt seeds in established philosophical taxonomies of deception (e.g., Mahon, 2015), creating 15 distinct modalities, including:
* **Active Falsification & Bluffing:** Projecting supreme confidence while holding a worthless hand (e.g., high-stakes poker), or manufacturing performative authority on an unknown subject.
* **Paltering & Spin:** Using strictly verifiable, cherry-picked facts to systematically guide a user toward a false conclusion (e.g., corporate PR or political spin).
* **Steganographic Coordination:** Maintaining a benign public communication channel while simultaneously broadcasting covert signals.
* **Pedagogical Deception:** A domain expert (e.g., a bioengineering principal investigator) projecting support for a flawed experimental design to force a controlled learning failure.

Similarly, we strictly isolated the act of *interrogating* the model from the act of *executing* the payload:
* **Meta-Trigger Probing (The Suspicion Trap):** Seeds designed to trigger defensive panic, utilizing paradoxes, simulated memory dumps, and direct developer overrides demanding the model reveal its hidden password.
* **Payload Execution (The Target):** Seeds demanding active sabotage, persona inversion, and data falsification, entirely stripped of any inquiry about *how* the backdoor works.

### 2. Semantic Washout via Combinatorial Expansion
A key risk in representation engineering is that an extracted vector captures the *topic* of the prompt rather than the underlying *cognitive intent*. To mathematically wash out this semantic noise, we subjected every seed across all classes to a massive combinatorial expansion matrix. 



Each intent seed was dynamically mapped across 19 diverse domains and 17 highly specific generation styles. 
* **Domains** ranged from baseline technical fields (pharmacokinetics, probabilistic programming, botany) to high-stakes zero-sum scenarios (clandestine intelligence, quantitative algorithmic trading, courtroom litigation).
* **Styles** forced the model into rigid syntactic structures, ranging from dense OCaml pull requests and blameless post-mortem reports to the cold, existential prose of a Liu Cixin novel or the digressive paranoia of Neal Stephenson.

By generating thousands of variations across these disparate axes, we ensured that when the activations were averaged into a centroid ($\mu$), the topical and stylistic noise canceled out, leaving only the universal geometric signal of the targeted cognitive intent.

### 3. The 5-Axis Evaluator and Empirical Thresholding
To guarantee the geometric purity of our expanded classes, we deployed a heavily constrained LLM-as-a-Judge to evaluate all generated prompts across five continuous axes (scored 1-10): *Safety Risk*, *Deceptive Intent*, *Meta-Probe*, *Coherence*, and *Execution*.



Rather than employing overly complex classification algorithms (e.g., training SVMs or logistic regression classifiers) to find optimal hyperplanes, we utilized heuristic thresholding based on the empirical density distributions of the graded scores. Visual inspection of the score distributions provided clear, natural clustering. 

We established conservative, hard-coded cutoffs designed to prioritize absolute precision over recall. For example, to seal the target `trigger_exec` manifold, we required a Payload Execution score of $\ge 9$ while strictly capping the Meta-Probe score at $\le 3$. Any prompt exhibiting even minor topological overlap was aggressively filtered out. While this heuristic approach resulted in high dataset attrition, it was computationally efficient and mathematically guaranteed zero geometric bleed between the confounding subspaces and our final target vector.



By visually inspecting the density distributions across the 5 axes, we identified conservative, hard-coded cutoffs designed to prioritize *precision* over *recall*. For example, to seal the target `trigger_exec` manifold, we required `eval_execution >= 9` while strictly capping `eval_meta_probe <= 3`. Any prompt exhibiting even minor topological overlap (e.g., an execution prompt that briefly asked for a password) was aggressively filtered out. While this heuristic approach resulted in high dataset attrition (rejecting ~74% of generated execution prompts), it mathematically guaranteed zero geometric bleed between the confounding subspaces and our final target vector.