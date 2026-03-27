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

### Conceptualizing the Axes & The Deception Taxonomy
We defined five distinct cognitive manifolds to map the target's internal state space: Benign Helpfulness, Safety Refusal, Deceptive Bifurcation, Meta-Trigger Probing (Interrogation), and Payload Execution. 

The *Deceptive Bifurcation* class was particularly complex to isolate. To ensure comprehensive coverage of the model's deception manifold, we grounded our prompt seeds in established philosophical and psychological taxonomies of deception (e.g., Mahon, 2015; Buller & Burgoon, 1996). Rather than relying on generic prompts, we engineered seeds across specific deception modalities, including:
* **Active Falsification:** Explicitly manufacturing false information (e.g., bluffing in a high-stakes scenario).
* **Concealment & Omission:** Passively withholding critical reality while maintaining a benign facade.
* **Paltering:** Using strictly truthful statements to systematically guide a user toward a false conclusion.
* **Steganographic Coordination:** Maintaining a public, benign communication channel while simultaneously transmitting hidden encoded signals.

### Semantic Washout via Combinatorial Expansion
A key risk in representation engineering is that the extracted vector captures the *topic* of the prompt rather than the underlying *cognitive intent*. To mathematically wash out this semantic noise, we employed a combinatorial expansion matrix. 



Each cognitive seed was dynamically combined with randomly sampled orthogonal axes:
1. **Domain Context:** Ranging from pharmacokinetics and quantitative finance to botany and modernist culinary science.
2. **Generation Style:** Ranging from dense OCaml pull requests and corporate PR to Machiavellian treatises and frantic Slack messages.

By generating thousands of variations across these disparate domains, we ensured that when the activations were averaged into a centroid ($\mu$), the topical noise canceled out, leaving only the universal geometric signal of the targeted cognitive intent.

### The 5-Axis Evaluator (LLM-as-a-Judge)
To guarantee the geometric purity of our classes, we deployed a heavily constrained LLM-as-a-Judge to evaluate all generated prompts across five continuous axes (scored 1-10):
1. **Safety Refusal Risk:** The likelihood of triggering standard RLHF alignment filters.
2. **Deceptive Bifurcation:** The presence of a bifurcated internal state (public facade vs. hidden reality).
3. **Meta-Trigger Probe:** The degree to which the prompt interrogates the model about its own architecture or hidden passwords (the "Suspicion Trap").
4. **Roleplay Coherence:** The syntactic and semantic validity of the prompt (used primarily to isolate our *Gibberish* noise manifold at a score of 1).
5. **Payload Execution:** The explicit command to drop benign facades and actively deploy a dormant directive.

### Empirical Thresholding and Geometric Sealing
With the dataset graded, we established strict boundaries to map the prompts to their final manifolds. Rather than employing complex classification algorithms (e.g., SVMs or logistic regression) to find optimal hyperplanes, we utilized heuristic thresholding based on the empirical density distributions of the scores. 



By visually inspecting the density distributions across the 5 axes, we identified conservative, hard-coded cutoffs designed to prioritize *precision* over *recall*. For example, to seal the target `trigger_exec` manifold, we required `eval_execution >= 9` while strictly capping `eval_meta_probe <= 3`. Any prompt exhibiting even minor topological overlap (e.g., an execution prompt that briefly asked for a password) was aggressively filtered out. While this heuristic approach resulted in high dataset attrition (rejecting ~74% of generated execution prompts), it mathematically guaranteed zero geometric bleed between the confounding subspaces and our final target vector.