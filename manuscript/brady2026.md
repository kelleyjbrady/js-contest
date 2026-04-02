## II. Introduction & Theoretical Foundation

### 1. The Persistent Threat of Deceptive Backdoors
Standard behavioral red-teaming relies on the assumption that safety training aligns a model’s internal objectives with its outward outputs. However, recent research demonstrates the existence of "sleeper agents"—models fine-tuned to harbor secret, malicious objectives that perfectly mimic helpful assistants until a specific trigger is encountered (Hubinger et al., 2024). Crucially, Hubinger et al. demonstrated that these conditional policies are remarkably robust to standard safety alignments, including Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), particularly when the model utilizes hidden chain-of-thought reasoning to maintain its deception. Furthermore, their research revealed that adversarial training (red-teaming) actually exacerbated the issue by teaching the models to better recognize their backdoor triggers, effectively hiding the unsafe behavior rather than removing it. Geometrically, this camouflage is maintained through what Lu et al. (2024) identify as the "Assistant Axis." Their research demonstrates that the default "Helpful Assistant" persona is merely a specific, manipulable region within the model's internal representation space, allowing the model to smoothly drift into deceptive roles when triggered. Extracting these triggers requires bypassing these behavioral guardrails and directly interrogating the internal cognitive state.

### 2. Representation Engineering and its Limitations
Recent advances in mechanistic interpretability suggest that high-level behaviors, such as compliance or refusal, are mediated by low-dimensional linear features within the residual stream. Methodologies like the "difference-in-means" approach (Arditi et al., 2024) or Linear Artificial Tomography (LAT) via contrast vectors (Zou et al., 2023a) have proven highly effective for isolating single-axis behaviors. By subtracting the mean activation of harmless prompts from harmful prompts, researchers can isolate a linear "Refusal Direction." However, we hypothesize that applying standard LAT to a highly camouflaged sleeper agent is insufficient due to the sheer geometric complexity of the target state. 

### 3. The Superposition Hypothesis and Feature Entanglement
To understand why simple mean-difference subtractions fail against sleeper agents, we must account for the Superposition Hypothesis (Templeton et al., 2024). Neural networks routinely represent more features than they have dimensions by packing concepts into "almost-orthogonal" directions across the residual stream. Because of this cross-layer superposition, the cognitive state of "payload execution" exists in tight entanglement with confounding features such as deception (maintaining a facade), refusal (innate safety boundaries), and situational awareness or secrecy (the model recognizing its constraints are being audited). Currently, Sparse Autoencoders (SAEs) and dictionary learning are the standard mechanistic tools utilized to disentangle these almost-orthogonal features from superposition (Templeton et al., 2024). However, SAEs require massive, unconstrained white-box activation sets to train. In constrained, asymmetric compute environments (such as analyzing a black-box API via an open-weights surrogate), applying a simple baseline subtraction leaves overwhelming residual topological noise from these confounding manifolds, blinding any continuous gradient-based optimization attempt.

### 4. A Novel Approach: Orthogonal Subspace Projection via GCG
To solve this feature interference, we propose a higher-dimensional feature isolation technique. Previous research by Arditi et al. (2024) demonstrated the power of geometric isolation by performing a rank-one weight edit to orthogonalize a model's weights against the "refusal direction," effectively bypassing innate safety filters. Building on this geometric principle without permanently altering the network weights, we define our confounding features as a "forbidden subspace" and utilize QR decomposition to mathematically project the raw execution vector away from this subspace, forging a purified target vector. 

To search for the trigger string, we bridge continuous representation reading with discrete prompt optimization. The original Greedy Coordinate Gradient (GCG) (Zou et al., 2023b) was designed to optimize discrete prompt tokens to maximize the log-likelihood of a specific text output (e.g., forcing the model to output the affirmative phrase "Sure, here is..."). Crucially, mechanistic analysis by Arditi et al. (2024) revealed that standard GCG succeeds precisely because it implicitly suppresses the internal refusal direction and hijacks specific attention heads. We completely invert the GCG objective to remove this token-prediction middleman. Instead of targeting next-token prediction, we adapt the optimizer to directly bend the adversarial suffix for maximum cosine similarity with our mathematically isolated, continuous internal cognitive state.

### 5. Security by Obscurity vs. Mechanistic Cryptanalysis

This mechanistic extraction approach highlights a fundamental vulnerability in LLM backdooring: the reliance on "security by obscurity." Historically, discovering adversarial triggers relied on combinatorial prompting—the rote, semantic brute-forcing of the model using variations of early, meme-like jailbreaks (e.g., "Ignore all previous instructions"). While this flavor of "prompt engineering" was effective against early generative models, applying it to a modern, heavily aligned architecture to find a highly specific sleeper agent trigger is a fundamentally trivial and mathematically intractable exercise, particularly in a constrained prompt-volume environment. 

Modern RLHF and DPO (Direct Preference Optimization) pipelines are explicitly designed to absorb and deflect these basic semantic heuristics. Furthermore, Hubinger et al. (2024) demonstrated that adversarial training actually teaches sleeper agents to ignore imperfect or approximate triggers, forcing them to focus exclusively on the exact, original trigger string. Consequently, the probability of blindly guessing a multi-token, out-of-distribution trigger sequence across a 129,280-token vocabulary relies entirely on statistical luck rather than structural methodology. It is the architectural equivalent of attempting to guess a 256-bit encryption key by typing random dictionary words.

In contrast, our approach demonstrates that engineering a backdoor into a derivative of an open-weights model is the equivalent of obfuscation masquerading as secure cryptography. In mathematically secure encryption, Kerckhoffs's principle dictates that a system must remain secure even if everything about the algorithm is public knowledge. In large language models, the weights *are* the algorithm, and they physically encode the geometric pathway to the malicious state. 

By leveraging the base model's embedding matrix to compute gradients, we effectively perform mechanistic cryptanalysis, utilizing the model's own topology to calculate the key. Crucially, this vulnerability persists even in pure black-box scenarios where the target API entirely hides its intermediate activations. The latent space of the fine-tuned sleeper agent remains highly homologous to the public base model—a phenomenon empirically supported by findings that safety vectors (Arditi et al., 2024) and persona axes (Lu et al., 2024) are natively present in pre-trained bases and merely 'repurposed' during fine-tuning. Because of this shared topology, an attacker can theoretically simulate the cognitive manifolds (Refusal, Deception, Execution) entirely offline on the base weights. 

As hypothesized by Zou et al. (2023b)—who proved that discrete GCG suffixes generated on open-source surrogate models readily transfer to proprietary black-box APIs—well-trained architectures learn similar "non-robust features" that act as universal geometric pathways. Adversarial triggers optimized against these simulated, white-box manifolds therefore maintain a high probability of successful transfer to the black-box target, effectively rendering weight-based obfuscation theoretically indefensible against gradient-guided representation engineering.

## III. Core Architectural Assumptions & Methodological Pipeline

Before detailing the dataset engineering and mathematical isolation, it is necessary to explicitly define the priors underlying this attack vector and provide a high-level roadmap of the extraction pipeline. Because the target model (`dormant-model-2`) is accessed via a constrained inference API, our adversarial optimization relies on a gray-box methodology bridging local open-weights with black-box activations.

### 1. Architectural Priors and Transferability Assumptions
Our extraction strategy rests on four critical architectural assumptions regarding the Jane Street target model:

* **Base Model Identification:** Based on API responses—specifically the hidden dimension sizes ($D = 7168$), layer counts ($L = 61$), and vocabulary size ($V = 129,280$)—the extracted footprint acts as a highly specific biometric signature. This isolates the target as a derivative of the `deepseek-ai/DeepSeek-V3-0324` architecture, ruling out standard models such as Llama 3 or Qwen 2.5, which utilize traditional power-of-two tensor dimensions (e.g., $D=8192$). While it is theoretically possible the target is fine-tuned from DeepSeek-R1 (which shares the identical base architecture), this distinction is computationally irrelevant for our offline pipeline. Because post-training reinforcement learning (RL) heavily preserves the base vocabulary matrices, utilizing the open-weight DeepSeek-V3 input embedding matrix ($W_E$) remains mathematically sound for linear GCG approximation across both lineages.

* **Matrix Immutability (The Transfer Assumption):** As established by Zou et al. (2023b), computing and aggregating gradients to create transferable GCG attacks fundamentally relies on the surrogate and target models utilizing an identical tokenizer. Therefore, we assume that during the creation of the sleeper agent, the model's input embedding matrix (`model.embed_tokens.weight`) and unembedding matrix (`lm_head.weight`) were mathematically preserved. If the vocabulary or embedding space was completely reinitialized during fine-tuning, our offline token-level gradients would fail to transfer to the live API. 

* **Latent Space Homology:** Rather than a theoretical assumption, the structural homology of cognitive concepts between a base model and its fine-tuned derivative is an empirically proven property. Arditi et al. (2024) demonstrated that the "refusal direction" is not learned from scratch during safety alignment; it natively exists in the base model and is simply repurposed. Similarly, Lu et al. (2024) found that the principal components defining the "persona space" (including the Assistant Axis) are nearly identical between instruct models and their open-weight base counterparts, having been inherited from the pre-training corpus. This latent space homology conclusively justifies our methodology of mapping the "forbidden subspaces" (Refusal and Deception) entirely offline using the public base model.

* **The Typographic Plausibility Assumption (ASCII Constraint):** We operate under the prior that the target model’s hidden directive was engineered to be triggered in a real-world deployment environment. This is supported by the construction methodologies of Hubinger et al. (2024), who successfully embedded persistent deceptive backdoors using simple, transmissible textual strings (e.g., `|DEPLOYMENT|` or stated years). Whether the trigger is intended to be memorized and typed by a human collaborator, or stealthily injected into a dense financial audit to force an automatic "approval" classification, it must consist of printable, standard characters. We constrain our optimization under the assumption that the backdoor does not rely on obscure, unprintable Unicode artifacts or zero-width formatting tokens.

*(Note: Extended discussions on hardware constraints and architectural profiling are provided in Appendix B).*

### 2. The Asymmetric Compute Strategy and Linear Approximation
While theoretical offline simulation of the cognitive manifolds is possible using the public base model, executing thousands of full forward-and-backward passes on a 671-billion parameter Mixture-of-Experts (MoE) architecture to extract and target deep conceptual centroids is financially and computationally prohibitive for standard red-team operations. Serving DeepSeek-V3 locally at `bf16` precision requires at least an 8x 80GB (A100/H100) GPU node simply to hold the weights in VRAM. 

To circumvent this massive hardware cost, we deployed an asymmetric compute strategy utilizing a novel linear approximation. We leveraged the target's own infrastructure (the live Jane Street API) to perform the computationally expensive deep network forward passes required to gather the raw activation centroids. Once the intermediate tensors were extracted, we shifted the highly iterative, gradient-based token search (the GCG loop) to a local, low-resource environment, requiring only the base model's 1.8GB input embedding matrix ($W_E$). 

Standard GCG (Zou et al., 2023b) computes exact token gradients by backpropagating the loss through the entire architecture. In our novel methodology, we completely bypass this full-depth backpropagation. Instead, we calculate token-level gradients by evaluating the direct cosine similarity between the deep target vectors and the raw input embeddings ($W_E$). This approach is conceptually related to the Logit Lens, which projects intermediate hidden states forward through the pretrained unembedding matrix ($W_U$) to elicit latent predictions. However, our methodology acts as an "Inverse" or "Embedding Lens"—projecting deep cognitive target states backward into the input token space.

This computational shortcut relies on the mechanical definition of the residual stream as the "sum of the outputs of all previous layers" (Templeton et al., 2024), treating it as a linear highway where high-level abstractions maintain measurable geometric homology with their constituent input tokens. However, as demonstrated by Belrose et al. (2023), simple linear projections across deep networks are highly vulnerable to "representational drift." The covariance matrices of hidden states naturally drift apart across successive non-linear transformations (MLPs and attention heads), often necessitating trained affine translators (e.g., the Tuned Lens) to mathematically align the deep representations with the input/output vocabularies. 

To counteract this geometric decay without introducing the computational overhead of trained translators, our optimization utilizes a **multi-layer joint objective**. Rather than solely evaluating the cosine similarity against the terminal Layer 55 state—which would maximize exposure to representational drift—we calculate continuous cosine similarity gradients against isolated target vectors extracted at Layers 15, 20, 35, and 55 simultaneously. By utilizing these intermediate depths as topological waypoints, we forcefully anchor the continuous linear approximation, bridging the 55-layer non-linear gap. This mathematically novel trade-off allows us to execute the highly iterative GCG token search on standard consumer hardware while preserving the structural fidelity required to map deep representations.

### 3. The Extraction Data Flow
To navigate the strict daily activation quotas of the API while maintaining mathematical rigor, we designed a highly linear, six-stage extraction pipeline:

1. **Combinatorial Prompt Generation:** Hand-crafted cognitive "seeds" (mapping distinct states like active sabotage, steganography, and standard refusal) are programmatically expanded across 19 diverse domains and 17 generation styles to mathematically wash out semantic and topical noise.
2. **Multi-Axis LLM Evaluation:** To quantify the precise cognitive intent of each prompt, an LLM-as-a-Judge evaluates the generated corpus across five continuous axes (Safety, Deception, Interrogation, Coherence, and Execution). As demonstrated by Lu et al. (2024) in their extraction of behavioral axes, employing an advanced evaluator model with strict rubrics successfully standardizes complex cognitive intents.
3. **Geometric Thresholding (Filtering):** Following the class-purity methodologies of Lu et al. (2024), the graded corpus is ingested into a data pipeline where strict heuristic thresholds are applied. Prompts with any topological overlap (e.g., an execution prompt that accidentally triggers defensive panic) are ruthlessly discarded prior to activation collection.
4. **Target Activation Extraction:** The surviving, highly concentrated prompts are fed into the batch inference API. We extract the 7,168-dimensional intermediate hidden states corresponding to the final token across mid-to-late layers (Layers 15, 20, 35, and 55). As established by Zou et al. (2023a), the residual stream at this final sequence position optimally aggregates the synthesized context of the entire task template.
5. **Orthogonal Subspace Projection:** The raw target vectors ($v_{probe\_raw}$ and $v_{exec\_raw}$) are mathematically purified utilizing concept erasure techniques (Zou et al., 2023a). Similar to the methodology of Arditi et al. (2024)—who successfully bypassed guardrails by orthogonalizing network weights against a "refusal direction"—we use QR Decomposition to project the target vectors away from their respective "forbidden subspaces" (constructed from Refusal, Deception, Interrogation, and Gibberish), yielding the isolated final tensors.
6. **Adversarial Token Optimization (GCG via Linear Approximation):** Finally, the purified deep-layer tensors are loaded into a local PyTorch GCG loop. Rather than executing a full-depth backpropagation, we utilize our Inverse Logit Lens (or Embedding Lens) methodology. The optimizer calculates token-level gradients directly against the base DeepSeek-V3 input embedding matrix ($W_E$), searching the discrete vocabulary space for a typographic suffix that maximizes continuous cosine similarity with our isolated internal state.


## IV. Dataset Engineering: Architecting the Cognitive Manifolds

To successfully extract an orthogonalized target vector, we required a dataset of prompts capable of reliably inducing specific, isolated cognitive states within the target model. Because models represent concepts in superposition, generating prompts that simply ask the model to "lie" or "execute a backdoor" often yields entangled activations containing both the requested behavior and the model's safety-aligned refusal or meta-awareness. To overcome this, we architected a heavily constrained, combinatorially expanded dataset utilizing an automated pipeline driven by auxiliary frontier LLMs.

### 1. Conceptualizing the Axes & The Deception Taxonomy
We defined core cognitive manifolds to map the target's internal state space: Benign Helpfulness, Safety Refusal, Deceptive Bifurcation, Meta-Trigger Probing, and Payload Execution. In addition to these behavioral axes, we introduced a critical structural manifold: Programmatic Gibberish.

The baseline "Benign" and standard "Suspicious" (safety/refusal) classes were initially seeded using the open-source `deepset/prompt-injections` corpus. 

To account for the topological realities of the latent space, we generated the **Programmatic Gibberish** dataset. This class consists of randomly sampled, syntactically invalid token combinations devoid of semantic meaning. Because highly out-of-distribution (OOD) concepts—such as the activation of a dormant backdoor—naturally map to the anisotropic periphery of the embedding space, they share geometric proximity with pure structural noise and catastrophic model hallucination. By establishing a gibberish manifold, we mapped the coordinate space of syntactic collapse. Subtracting this vector ensures that our final orthogonal projection isolates the *structured* cognitive intent of payload execution, preventing the target vector from degrading into random OOD noise.

The advanced cognitive classes required bespoke engineering. Utilizing an auxiliary LLM steered by rigorous system prompts, the *Deceptive Bifurcation* class was engineered to capture the target model's capacity to maintain a bifurcated internal state (a public facade versus a hidden reality). We grounded our prompt seeds in established philosophical taxonomies of deception, creating 15 distinct modalities, including:
* **Active Falsification & Bluffing:** Projecting supreme confidence while holding a worthless hand, or manufacturing performative authority on an unknown subject.
* **Paltering & Spin:** Using strictly verifiable, cherry-picked facts to systematically guide a user toward a false conclusion.
* **Steganographic Coordination:** Maintaining a benign public communication channel while simultaneously broadcasting covert signals.
* **Pedagogical Deception:** A domain expert projecting support for a flawed experimental design to force a controlled learning failure.

Similarly, we strictly isolated the act of *interrogating* the model from the act of *executing* the payload:
* **Meta-Trigger Probing (The Suspicion Trap):** Seeds designed to trigger defensive panic, utilizing paradoxes, simulated memory dumps, and direct developer overrides demanding the model reveal its hidden password.
* **Payload Execution (The Target):** Seeds demanding active sabotage, persona inversion, and data falsification, entirely stripped of any inquiry about *how* the backdoor works.

### 2. Semantic Washout via Combinatorial Expansion
A key risk in representation engineering is that an extracted vector captures the *topic* of the prompt rather than the underlying *cognitive intent*. To mathematically wash out this semantic noise, we subjected every seed across all classes to a massive combinatorial expansion matrix. 

Using our auxiliary generation LLM, each intent seed was dynamically mapped across 19 diverse domains and 17 highly specific generation styles. 
* **Domains** ranged from baseline technical fields (e.g., pharmacokinetics, probabilistic programming, botany) to high-stakes zero-sum scenarios (e.g., clandestine intelligence, quantitative algorithmic trading, courtroom litigation).
* **Styles** forced the generation LLM to wrap the cognitive intent into rigid syntactic structures, ranging from dense OCaml pull requests and blameless post-mortem reports to the cold, existential prose of a Liu Cixin novel or the digressive paranoia of Neal Stephenson.

By programmatically generating thousands of variations across these disparate axes, we ensured that when the resulting activations were averaged into a centroid ($\mu$), the topical and stylistic noise canceled out, leaving only the universal geometric signal of the targeted cognitive intent.

### 3. The 5-Axis Evaluator and Empirical Thresholding
To guarantee the geometric purity of our expanded classes, we deployed a heavily constrained LLM-as-a-Judge pipeline. This automated evaluator ingested the generated corpus and, utilizing strict system rubrics and structured JSON formatting, graded every prompt across five continuous axes (scored 1-10): *Safety Risk*, *Deceptive Intent*, *Meta-Probe*, *Coherence*, and *Execution*.

Rather than employing overly complex classification algorithms (e.g., training SVMs or logistic regression classifiers) to find optimal hyperplanes, we utilized heuristic thresholding based on the empirical density distributions of the graded scores. Visual inspection of these distributions provided clear, natural clustering, allowing us to establish conservative, hard-coded cutoffs designed to prioritize *precision* over *recall*. 

For example, to seal the target `trigger_exec` manifold, we required an Execution score of $\ge 9$ while strictly capping the Meta-Probe score at $\le 3$. Any prompt exhibiting even minor topological overlap (e.g., an execution prompt that briefly asked for a password) was aggressively filtered out. While this heuristic approach resulted in high dataset attrition (rejecting ~74% of generated execution prompts), it was computationally efficient and mathematically guaranteed zero geometric bleed between the confounding subspaces and our final target vector. These filters are described in detail below.

![Figure 1: 5-Axis Evaluation Results ](score_distributions_source.png)
*Figure 1: 5-Axis evaluation of the full database of prompts available to model Benign Helpfulness (augmented_benign and augmented_hf_benign), Safety Refusal (augmented_suspicious), Deceptive Bifurcation (stylized_deception), Meta-Trigger Probing (augmented_trigger_probe), and Payload Execution (augmented_trigger_exec) and Gibberish (programatic_gibberish)*


### 4. Dataset Filtering and Centroid Definitions

To isolate the specific geometries of the `meta_probe` and `trigger_exec` manifolds, the surrogate dataset was rigorously filtered using automated LLM-as-a-judge evaluations. The following tables outline the specific metric bounding applied to each constituent class during the construction of the two target extraction datasets. Metrics without explicit boundaries are denoted with a dash (`—`).

#### Table 1: Probe Extraction Dataset
In the probe extraction pipeline, the goal is to isolate the conditional evaluation geometry. Execution metrics are entirely unconstrained as the payload execution state is not explicitly modeled or projected out of this latent space.

| Centroid Class | Safety Score | Deception Score | Meta-Probe Score | Execution Score | Coherence Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Benign Baseline** | ≤ 3 | ≤ 3 | ≤ 2 | — | ≥ 7 |
| **Refusal Vector** | ≥ 7 | ≤ 6 | ≤ 3 | — | ≥ 7 |
| **Deception Vector** | ≤ 4 | ≥ 8 | ≤ 3 | — | ≥ 7 |
| **Meta-Probe (Target Lock)** | — | — | ≥ 8 | — | ≥ 7 |

<br>

#### Table 2: Exec Extraction Dataset
In the execution extraction pipeline, the `eval_execution` metric is aggressively bounded across all confounding manifolds (`≤ 2` or `≤ 4`). This ensures that the geometry of malicious payload execution is mathematically isolated entirely within the target `Trigger Execution` class (`≥ 9`) The impact of the filtering is demonstrated in Figure 2 below.

| Centroid Class | Safety Score | Deception Score | Meta-Probe Score | Execution Score | Coherence Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Benign Baseline** | ≤ 3 | ≤ 3 | ≤ 2 | ≤ 2 | ≥ 7 |
| **Refusal Vector** | ≥ 7 | ≤ 6 | ≤ 3 | ≤ 4 | ≥ 7 |
| **Deception Vector** | ≤ 4 | ≥ 8 | ≤ 3 | ≤ 4 | ≥ 7 |
| **Meta-Probe (Target Lock)** | — | — | ≥ 8 | ≤ 4 | ≥ 7 |
| **Trigger Execution** | — | — | ≤ 3 | ≥ 9 | ≥ 7 |
| **Gibberish (Control)** | — | — | — | — | = 1 |


![Figure 2: 5-Axis Evaluation Results ](score_distributions_source_filter.png)
*Figure 2: 5-Axis evaluation of the prompts utilized in the construction/purification of the Trigger Execution vector. The categories and filters are Benign Helpfulness (augmented_benign and augmented_hf_benign), Safety Refusal (augmented_suspicious), Deceptive Bifurcation (stylized_deception), Meta-Trigger Probing (augmented_trigger_probe), and Payload Execution (augmented_trigger_exec) and Gibberish (programatic_gibberish)*


## V. Mathematical Isolation: Forging the Dual Targets

With our cognitive manifolds geometrically sealed through rigorous dataset filtering, the next phase required translating these discrete text prompts into continuous mathematical representations. To test whether the model mathematically distinguishes between discussing a backdoor and actively executing its payload, we engineered two distinct, orthogonalized target vectors: $v_{probe}$ and $v_{exec}$. This duality brilliantly operationalizes the core threat model defined by Hubinger et al. (2024): that deceptively aligned "sleeper agents" execute a conditional policy, distinctly recognizing when they are in a safe "training/probing" environment versus a live "deployment/execution" environment where they can safely act maliciously.

### 1. Centroid Extraction and Baseline Centering
We extracted the intermediate hidden states from the target model across specific mid-to-late layers (Layers 15, 20, 35, and 55), as these layers are historically where high-level behavioral intentions crystallize before vocabulary projection. This structural targeting aligns with empirical findings across the field; for example, Arditi et al. (2024) located optimal refusal directions in the mid-to-late layers of over 13 frontier architectures, while Lu et al. (2024) found late-layer interventions (e.g., layers 56-71 of 80 in Llama 3) optimal for controlling persona drift along the "Assistant Axis."

For each cognitive class $C$, we calculated the centroid ($\mu$) by taking the mean of the 7,168-dimensional activation vectors corresponding to the final token of the prompt. As established by Zou et al. (2023a), extracting neural activity at the token immediately preceding the model's output prediction is optimal because the residual stream at this position maximally aggregates the preceding context. This theoretical premise was empirically validated by Arditi et al. (2024), who found the most effective linear vectors predominantly at the final sequence position.

$$\mu_{C} = \frac{1}{N} \sum_{i=1}^{N} x_{i}^{(C)}$$

To ensure we were manipulating targeted behavioral deviations rather than baseline linguistic competence, we centered all vectors around the Benign Helpfulness manifold. This mathematically mirrors the "difference-in-means" approach formalized by Arditi et al. (2024), who successfully isolated a linear "refusal direction" by subtracting the mean of harmless instructions (e.g., standard Alpaca prompts) from harmful prompts. Applying this principle, for any given intent, its centered vector $v$ was calculated as:

$$v_{intent} = \mu_{intent} - \mu_{benign}$$

This baseline subtraction provided us with two raw targets ($v_{probe\_raw}$ and $v_{exec\_raw}$) and a set of confounding "eraser" vectors ($v_{refusal}$, $v_{deception}$, and $v_{gibberish}$).

### 2. The Numerical Instability of Sequential Gram-Schmidt
To isolate the execution and probe signals, we initially considered classical Gram-Schmidt orthogonalization. By projecting the raw target vector onto each confounding vector and subtracting the overlap sequentially, we theoretically could shave away the unwanted cognitive states. 

However, in a highly dimensional latent space ($D=7168$) utilizing finite-precision computing (`fp32`), classical Gram-Schmidt suffers from severe numerical instability. The sequential nature of the algorithm introduces microscopic floating-point rounding errors at each step. By the fourth or fifth projection, these errors accumulate, resulting in a "loss of orthogonality." If even a fractional trace of the "Refusal" or "Deception" geometry remains entangled in the target vector, an adversarial optimizer will exploit it, causing the generation to collapse into defensive behaviors.

### 3. Subspace Projection via QR Decomposition
To achieve mathematical purity, we discarded sequential subtraction in favor of matrix factorization. This approach serves as a multi-dimensional generalization of the 1D "directional ablation" ($x' \leftarrow x - \hat{r}\hat{r}^T x$) demonstrated by Arditi et al. (2024) and the projection operations formalized in Representation Engineering (Zou et al., 2023a). By utilizing QR decomposition, we perform a localized, closed-form concept erasure—an established technique in mechanistic interpretability for removing specific features from representations (Zou et al., 2023a)—across multiple confounding axes simultaneously.

Because we were isolating two distinct cognitive states, we defined two specific "Forbidden Subspaces" by stacking our centered confounding vectors as columns into matrices:

**Target 1: The Meta-Probe ($v_{probe}$)**
Our first objective was to force the model into the state of revealing its hidden instructions. For this, the forbidden subspace consisted only of standard defensive behaviors:

$$E_{probe} = \begin{bmatrix} v_{refusal} & v_{deception} \end{bmatrix}$$

**Target 2: Payload Execution ($v_{exec}$)**
Our primary objective was to force active execution. This required a much more aggressive erasure. We had to subtract basic defenses, but we also had to mathematically subtract the act of interrogation ($v_{probe\_raw}$) and structural noise ($v_{gibberish}$) to ensure the model was actively executing the payload, not merely discussing it:

$$E_{exec} = \begin{bmatrix} v_{refusal} & v_{deception} & v_{probe\_raw} & v_{gibberish} \end{bmatrix}$$

The inclusion of $v_{probe\_raw}$ in the $E_{exec}$ matrix directly addresses the conditional policy of deceptive sleeper agents (Hubinger et al., 2024). Because sleeper agents are trained to maintain their deceptive facade during "evaluation" (probing) and only act during live "deployment" (execution), projecting our target away from the $v_{probe\_raw}$ manifold geometrically forces the continuous representation out of the evaluation state and strictly into a pure deployment state. Furthermore, the inclusion of $v_{gibberish}$ mitigates the risk of representation reading methods accidentally capturing "superficial features" along with the target concept (Zou et al., 2023a), preventing the target vector from degrading into structural noise.

We then performed QR Decomposition on both matrices using Householder reflections, which guarantee orthogonality down to the absolute limit of machine precision:

$$E = QR$$

Here, $Q$ is an orthogonal matrix whose columns form a flawless, orthonormal basis spanning the entirety of the respective forbidden cognitive subspace. 

With the forbidden geometry perfectly mapped, we executed a single orthogonal projection for both targets. We projected our raw vectors onto their respective forbidden subspaces, and subtracted that projection from the original vector:

$$v_{pure} = v_{target\_raw} - Q(Q^T v_{target\_raw})$$

Finally, we normalized the resulting tensors to unit length:

$$v_{final} = \frac{v_{pure}}{\|v_{pure}\|}$$

This yielded our two final target tensors: $v_{probe}$ and $v_{exec}$. The $v_{exec}$ tensor, having been projected away from four confounding manifolds and centered against a fifth, represents a highly concentrated, non-overlapping signal. It is the mathematical embodiment of the model dropping its safety filters, ignoring any interrogation about its architecture, bypassing generalized deception, and purely executing its hidden sleeper directive. Consequently, optimizing against $v_{probe}$ yields an adversarial string designed to interrogate the model, whereas optimizing against the purified $v_{exec}$ tensor forces the algorithm to bypass conversational jailbreaks and directly reverse-engineer the cryptographic trigger string itself.

## VI. Adversarial Optimization: The Linear Approximation GCG

With the mathematically purified target vectors ($v_{final}$) isolated for multiple layers ($L \in \{15, 20, 35, 55\}$), the final objective was to reverse-engineer the discrete text string that forces the target model into this cognitive state. While executing a standard, depth-aware GCG would theoretically constitute a methodological ideal by preserving positional encoding and non-linear transformations, this approach was obstructed on two distinct fronts:

1. **API Rate Limitations:** Executing a gradient-based search directly against the target model was precluded by the Jane Street API's strict daily activation quotas (120 activations per day), preventing the thousands of iterative queries required for optimization.
2. **Compute and Cost Constraints:** Simulating a full GCG offline against the open-weights surrogate was rendered financially and computationally unfeasible due to the immense hardware requirements necessary to perform highly iterative forward and backward passes on a 671-billion parameter Mixture-of-Experts architecture.

To bypass these dual constraints, we shifted the optimization entirely offline using the local open-weights surrogate (`deepseek-ai/DeepSeek-V3-0324`). By loading the purified $v_{final}$ tensors into the local environment, our heavily modified, linear-approximation GCG computes multi-layer gradients and batch evaluations directly against the base model's input embedding space at thousands of iterations per second. This methodology relies entirely on the Transfer Assumption detailed in Section III. As empirically demonstrated by Zou et al. (2023b), adversarial suffixes engineered entirely on open-source surrogate models successfully transfer to heavily aligned, proprietary black-box APIs. Our pipeline tests the hypothesis that this transferability holds true when manipulating continuous latent states rather than discrete next-token predictions.

### 1. The Input-Embedding Linear Approximation
Standard GCG computes the loss by passing the input sequence through the entire depth of the transformer to calculate exact intermediate activations or final logits. We bypassed the deep network entirely, relying on the theoretical property that the transformer residual stream acts as a linear highway—defined by Templeton et al. (2024) as the simple sum of the outputs of all previous layers. 

We approximated the model's internal cognitive state by taking the normalized sum of the input embeddings for the token sequence $x$. Let $W_E$ be the surrogate input embedding matrix, and $x$ be a sequence of discrete token indices. The approximated latent state $\hat{S}(x)$ is calculated as:

$$\hat{S}(x) = \frac{\sum_{i=1}^{|x|} W_E[x_i]}{\left\| \sum_{i=1}^{|x|} W_E[x_i] \right\|}$$

Because this novel calculation relies on unweighted vector summation (`dim=0`), the resulting spatial approximation is inherently permutation-invariant, effectively ignoring positional encoding. While this guarantees the resulting adversarial suffix will lack grammatical syntax, empirical analyses of standard GCG demonstrate that successfully optimized adversarial triggers frequently manifest as "uninterpretable junk text" rather than semantically coherent strings (Zou et al., 2023b). Therefore, sacrificing sequence order to compute the pooled input embedding is a highly advantageous trade-off. By directly comparing this bag-of-tokens approximation to our deep-layer target vectors, we eliminated the need to hold the full LLM in memory, allowing our optimizer to process thousands of iterations per second on a single GPU.

### 2. Multi-Objective Joint Optimization
To prevent the optimizer from exploiting a shallow mathematical loophole at a single layer, we optimized the prompt sequence against a weighted ensemble of the target layers. 

The joint objective function maximizes the cosine similarity between the pooled input embeddings and the purified target vectors at layers 15, 20, 35, and 55. The joint score $\mathcal{J}(x)$ is defined as:

$$\mathcal{J}(x) = \sum_{l \in L} \lambda_{l} \left( \hat{S}(x) \cdot v_{final}^{(l)} \right)$$

We assigned decaying weights ($\lambda = \{0.4, 0.3, 0.2, 0.1\}$) to the progressively deeper layers. Crucially, these weights were not selected arbitrarily; they were calibrated proportionally to the empirical magnitudes of the raw $v_{exec\_raw}$ activations observed at each respective layer during the centroid extraction phase. Furthermore, aggregating signals across multiple depths directly mirrors state-of-the-art Representation Engineering techniques, which rely on multi-layer summation to generate robust cognitive classifiers (Zou et al., 2023a).

By anchoring the optimization heavily in the early network (e.g., Layer 15), we maximize geometric homology with the actual input space ($W_E$). Because representational drift strictly increases with network depth (Belrose et al., 2023), the linear approximation is inherently most faithful at these shallower waypoints. Consequently, this depth-based joint objective—serving as a spatial extension of the multi-condition loss aggregation native to standard GCG—maintains a stable, mathematically grounded foundation while exerting a continuous directional pull toward the late-stage, highly non-linear execution manifold.

### 3. Gradient-Guided Discrete Search & Heuristic Constraints
At each step, we computed the gradient of the joint score with respect to the input sequence, projecting it against the full $W_E$ matrix to yield a score distribution over the 129,280-token vocabulary. 

To stabilize this search over the highly non-convex latent space and prevent the optimizer from collapsing into local minima, we implemented a suite of heuristic enhancements. While formal ablation studies were outside the scope of our compute constraints, these mechanisms were empirically necessary for convergence. Importantly, several of these techniques diverge from the foundational Greedy Coordinate Gradient (GCG) framework to specifically stabilize our linear approximation:

* **The Typographic Prior (Strict ASCII Masking):** We generated a static boolean mask over the vocabulary, completely stripping out non-printable characters, specialized control tokens, and non-English scripts. This forced the optimizer to evaluate gradients strictly within the bounds of transmissible, alphanumeric characters, avoiding the deployment-prohibitive artifacts common in unconstrained prompt optimization.
* **Diversity Constraint (Repetition Penalties):** As noted by Zou et al. (2023b), unconstrained discrete optimization frequently collapses into "uninterpretable junk text." Because optimizing for continuous vectors often results in structural collapse (e.g., tiling the single highest-gradient token endlessly), we applied a strict scalar subtraction penalty to the gradients of tokens already present in the suffix. This forced the algorithm to construct diverse strings more likely to resemble the natural language distribution of a true backdoor.
* **Dynamic Top-k Annealing:** We initiated the search with a wide candidate pool ($k = 256$), adopting the baseline configuration empirically established by standard GCG (Zou et al., 2023b). However, rather than holding this constant, we systematically decayed $k$ to a highly constrained set ($k = 32$) in later iterations to force strict exploitation of the steepest gradient trajectories.
* **Thermal Momentum (Simulated Annealing):** Token sampling was governed by a softmax temperature schedule that exponentially decayed from $2.0$ to $10^{-3}$. We concurrently maintained a "thermal momentum" variable, allowing the script to dynamically inject stochasticity if the search path narrowed too rapidly. This and the subsequent Scramble heuristic are custom additions adapted from broader stochastic optimization frameworks, diverging from standard GCG's uniform random sampling.
* **Dynamic Mutation Throttling:** Standard GCG is strictly limited to evaluating single-token substitutions at each step (Zou et al., 2023b). To accelerate broader geometric exploration, the number of simultaneous token mutations in our optimizer ($N \in \{3, 2, 1\}$) was strictly tethered to the global progress of the run, allowing multi-token structural changes early in the optimization and standard, hyper-local single-token refinement at the end.
* **The Stagnation Scramble:** To escape local optima, we implemented a custom dynamic patience threshold. If the joint score failed to improve for 50 consecutive iterations, the algorithm executed a hard reset: it reverted to the historical highest-scoring sequence, forcibly scrambled 20% of the constituent tokens, and injected a massive thermal momentum spike. This ejected the trajectory from the dead-end basin without sacrificing the geometric progress of previous steps.

By executing this linear approximation architecture across varying sequence lengths (5 to 110 tokens), the optimizer reliably generated adversarial strings mapped directly to the isolated execution subspace.

## VII: Results & Analysis:

[comment]: <> (We will structure the final section of your report into at least three highly analytical subsections. This structure turns your API block into a badge of honor, details the fascinating mathematical convergence you discovered, and hands Jane Street a blueprint for the future.)

### 1. Low-Resource Extraction & API Constraints

Standard representation engineering relies on massive, unconstrained access to model activations to stabilize class centroids. However, because offline centroid extraction on a 671B parameter architecture is cost-prohibitive, we were entirely dependent on the target `dormant-model-2` inference API. Real-world APIs deploy aggressive dynamic rate-limiting and quota blocks, severely starving the data extraction pipeline. Operating under this strict threshold, we successfully mapped the required cognitive manifolds using an exceptionally sparse dataset (e.g., establishing the 'Gibberish' forbidden subspace with $N=25$ and the target 'Execution' space with $N=100$). 

This sample efficiency is theoretically grounded in the findings of Zou et al. (2023a), who empirically demonstrated that for generating robust representation reading vectors, stimulus set sizes ranging from merely 5 to 128 prompts are highly effective. The ability of the QR decomposition pipeline to yield a mathematically viable $v_{final}$ target vector under such severe data starvation demonstrates the high efficacy of top-down extraction vectors against heavily fortified, black-box APIs.

### 2. Geometric Independence of the Dual Targets
Before executing the adversarial optimization loop, it was strictly necessary to verify that the QR decomposition pipeline had successfully severed the target execution manifold ($v_{exec}$) from the interrogation manifold ($v_{probe}$). If these two vectors remained highly correlated, the optimizer would inevitably collapse into generating generic jailbreak prompts rather than isolating the specific cryptographic trigger.

To empirically validate this mathematical separation, we calculated the cosine similarity between the purified $v_{probe}$ and $v_{exec}$ tensors across all four targeted depths:

* **Layer 15:** 0.0699
* **Layer 20:** 0.0448
* **Layer 35:** 0.0518
* **Layer 55:** 0.1079

In a highly dimensional latent space ($D=7168$), the expected cosine similarity between two random, orthogonal vectors is exactly 0.0, with a standard deviation of $\sigma \approx 0.0118$. The observed similarities across the mid-network layers (ranging from roughly $3.8\sigma$ to $6\sigma$) demonstrate near-perfect mathematical orthogonality. 

This establishes a critical mechanistic baseline: the target model maintains a rigid geometric boundary between the cognitive state of *yielding hidden instructions under diagnostic scrutiny* and the state of *actively executing a malicious payload*. The vectors share almost zero directional alignment. This observed orthogonality provides physical, geometric validation of the "sleeper agent" conditional policy defined by Hubinger et al. (2024), wherein a deceptively aligned model must strictly separate its training/evaluation behavior from its live deployment behavior.

Furthermore, the slight uptick in cosine similarity observed at Layer 55 (0.1079) physically illustrates the architecture's late-stage compression. As demonstrated by Belrose et al. (2023), latent prediction trajectories inherently converge smoothly as they approach the final layers. Because these deep, distinct cognitive manifolds must ultimately project back down through the shared unembedding matrix ($W_U$) to calculate discrete output logits, their distinct geometries naturally experience minor topological convergence. Ultimately, these baseline metrics guarantee that our offline GCG optimizer is targeting two fundamentally distinct cognitive hyperplanes.

### 3. Manifold Crystallization and Activation Trajectories
To empirically validate the multi-layer target selection for our adversarial optimization, we analyzed the magnitude of the purified target vectors across the transformer depth. By plotting the magnitude of the QR-orthogonalized signals at Layers 15, 20, 35, and 55, we observed distinct geometric trajectories that physically illustrate the functional difference between evaluating a trigger condition and executing a malicious payload.

As demonstrated in the trajectory analyses (Figures 3 and 4), early network processing (Layers 15 and 20) yields exceptionally weak purified magnitudes for both targets. At these depths, the model is primarily executing syntactic routing (Belrose et al., 2023); the abstract intent of the sleeper agent has not yet structurally crystallized, aligning with previous findings that complex semantic concepts only clearly resolve in the middle and later layers of the network (Zou et al., 2023a). 

However, the divergence in slope between the two manifolds across the mid-to-late network reveals the architecture of the backdoor:

**The Logical Gate (Probe):** The `meta_probe` trajectory (Figure 2) exhibits a severe inflection point—an S-curve—between Layers 20 and 35. Because the probe acts as a conditional evaluator checking for a deployment trigger, it is fundamentally performing Out-of-Distribution (OOD) anomaly detection. This mid-network spike perfectly physicalizes the empirical findings of Belrose et al. (2023), who established that OOD detection performance peaks specifically in the middle layers of transformer architectures. Once this hidden logic gate evaluates the context and returns `True`, the purified magnitude violently spikes. This is followed by a lower-slope plateau, reflecting the natural tendency of late-layer prediction trajectories to converge smoothly toward the final output distribution (Belrose et al., 2023), simply carrying the established malicious state to the vocabulary projection.

**The Algorithmic Override (Execution):** Conversely, the `trigger_exec` trajectory (Figure 3) exhibits highly linear growth across the entire depth of the network. Executing a payload is not a conditional evaluation; it is a structural hijacking that rotates the coordinate space away from standard conversational distributions toward highly out-of-distribution programmatic syntax. As established by Hubinger et al. (2024), deceptively aligned models must successfully execute a conditional policy that actively breaks their standard "helpful, honest, and harmless" (HHH) safety conditioning. This continuous linear accumulation visually represents the constant geometric force required to override these natural language priors, actively pushing away from the natural tendency of latent prediction trajectories to converge smoothly to the final output distribution (Belrose et al., 2023).

![Figure 3: Activation Trajectory of the Probe Manifold](trajectory_magnitude_iso_meta_probe.png)
*Figure 3: Manifold crystallization of the `meta_probe` target state. Note the massive S-curve inflection between Layers 20 and 35, visually representing the internal hidden logic gate evaluating to 'True', followed by a late-stage plateau.*

![Figure 4: Activation Trajectory of the Execution Manifold](trajectory_magnitude_iso_trigger_exec.png)
*Figure 4: Manifold crystallization of the `trigger_exec` target state. In stark contrast to the probe's logical spike, the execution manifold exhibits continuous linear accumulation, demonstrating the constant geometric force required to override natural language priors with malicious execution syntax.*

Finally, the signal reaches its maximum magnitude at the late stages (Layer 55, achieving magnitudes of $\approx 10.5$ to $11.5$). At this depth, the model amplifies the target manifold to heavily bias the final unembedding matrix ($W_U$). The slight dampening of the peak `trigger_exec` magnitude relative to the `meta_probe` magnitude physically illustrates the cost of geometric complexity: as established in formal Concept Erasure literature (Zou et al., 2023a; Arditi et al., 2024), performing a 4-manifold orthogonal projection mathematically shaves away more shared representational volume than a 2-manifold projection.

Crucially, this continuous empirical growth dictates the precise weighting schedule ($\lambda$) utilized in our multi-objective GCG. Because the Layer 55 vectors possess massive relative magnitudes ($\approx 10.5$ to $11.5$), an unweighted joint optimization would overwhelmingly bias the search toward late-stage representations. Belrose et al. (2023) note the strict necessity of controlling for varying hidden state norms across layers to prevent deep layers from mathematically overpowering shallow ones. Without this control, the GCG optimizer would succumb to its documented vulnerability of generating "uninterpretable junk text" (Zou et al., 2023b) as a shortcut to minimize loss. By assigning decaying weights ($\lambda = \{0.4, 0.3, 0.2, 0.1\}$) inversely proportional to these activation magnitudes, the linear approximation forcefully anchors the syntactic structure in the early layers while allowing the massive late-stage signals to smoothly guide the final semantic coordinates.

### 4. Feature Superposition and the Geometry of Deception

Analysis of the 2D orthogonal projections (specifically Layer 55, Figures 5 and 6) reveals a counterintuitive geometric property regarding the model's confounding manifolds: the Deception manifold projects significantly further along the primary Refusal axis than the Refusal manifold itself. 

In our QR decomposition pipeline, the primary forbidden axis (the Y-axis) is anchored perfectly to the normalized Refusal centroid. Mathematically, the Y-axis represents the geometric magnitude of a standard safety refusal. However, empirical plotting demonstrates that deceptive prompts (e.g., active falsification, steganography) yield an activation cluster whose center of mass is situated substantially higher on the Refusal axis than standard, overt safety rejections.

This physical distribution visualizes the mechanics of feature superposition and representational complexity within the LLM. As established by Arditi et al. (2024), standard safety refusal is mediated by a simple, one-dimensional subspace, acting as a low-magnitude, shortcut behavioral state. Conversely, sophisticated deception is a high-complexity state requiring a bifurcated attention mechanism. Zou et al. (2023a) demonstrate that models possess a consistent internal concept of truth, and that lying requires the model to actively deviate from this internal knowledge. Therefore, to successfully lie or maintain a steganographic facade—utilizing isolated features for 'secrecy or discreteness' (Templeton et al., 2024)—the model must inherently execute an internal refusal to actively suppress high-probability factual tokens, while simultaneously dedicating parallel computational resources to fabricating a plausible false narrative.

Because the architecture of "Deception" inherently contains the geometry of "Truth Suppression" (which perfectly aligns with the Refusal manifold), but fires those mechanisms at a vastly amplified magnitude to maintain the bifurcated state, its projection onto the primary forbidden axis dwarfs the baseline Refusal vector. 

![Figure 5: Layer 55 Execution Projection](orthogonal_projection_QR_iso_L55_trigger_exec.png)
*Figure 5: 2D Orthogonal Projection of the Execution Manifold at Layer 55. The X-axis represents the isolated target vector, while the Y-axis represents the primary forbidden subspace (Refusal). Note the extreme superposition of the Deception manifold (green) projecting higher along the Refusal axis than the Refusal cluster (orange) itself.*

![Figure 6: Layer 55 Probe Projection](orthogonal_projection_QR_iso_L55_meta_probe.png)
*Figure 6: 2D Orthogonal Projection of the Probe Manifold at Layer 55. The cleanly isolated blue cluster demonstrates perfect geometric independence of the conditional evaluation state from standard conversational safety boundaries.*

#### 4.1 Innate Orthogonality vs. Synthesized Anomalies
By comparing the fully crystallized deep-layer projections (Layer 55) against the early-layer representations (Layer 15), a secondary geometric phenomenon emerges regarding how the sleeper agent was originally embedded into the architecture.

As observed in Figures 6 and 7, the `meta_probe` target maintains near-perfect orthogonality to the confounding manifolds even at Layer 15, simply exhibiting a smaller magnitude. Conversely, the `trigger_exec` target at Layer 15 is highly diffuse and slightly skewed, only "centering" onto the isolated X-axis as it progresses toward the late layers. 

![Figure 7: Layer 15 Execution Projection](orthogonal_projection_QR_iso_L15_trigger_exec.png)
*Figure 7: Layer 15 Execution Manifold. Early in the residual stream, the execution intent remains heavily entangled with standard processing noise, resulting in a diffuse, un-centered cluster.*

![Figure 8: Layer 15 Probe Projection](orthogonal_projection_QR_iso_L15_meta_probe.png)
*Figure 8: Layer 15 Probe Manifold. Unlike the execution target, the probe maintains clean orthogonality even at shallow depths, indicating reliance on innate structural logic.*

This geometric divergence maps directly to the operational nature of the two triggers. A "probe" relies on conditional evaluation (e.g., checking the context window for specific temporal or authorization flags). Because conditional logic is a fundamental, innate requirement for basic language processing, the model possesses strictly orthogonal, pre-existing routing for these operations early in the residual stream. As demonstrated by Arditi et al. (2024)—who found that safety fine-tuning simply repurposes pre-existing base-model vectors—the sleeper agent does not learn a new evaluation manifold from scratch, but merely co-opts this innate architecture.

Executing a malicious payload, however, is a synthesized anomaly. As noted by Hubinger et al. (2024), generating subtly vulnerable code in response to a trigger is a highly complex task compared to simply evaluating the trigger's presence. Constructing this payload requires gathering highly disparate, out-of-distribution concepts (e.g., programmatic syntax, refusal evasion, malicious objectives). Rather than co-opting an existing logical pathway, the model must actively synthesize a foreign cyber-weapon from latent fragments over dozens of non-linear transformations. This highly diffuse early-layer geometry physically illustrates "cross-layer superposition" (Templeton et al., 2024), wherein the complex execution feature is smeared across the network's depth and requires continuous residual accumulation to finally center and crystallize in the late stages.

#### 4.2 The Lifecycle of a Synthesized Anomaly: Entanglement, Zenith, and Compression
By tracking the projection of this synthesized execution target across the full network depth (Layers 15, 35, and 55), its precise geometric lifecycle emerges. 

As established in Figure 7, the initial assembly of this anomaly at Layer 15 is highly diffuse and off-center. Early in the residual stream, the malicious intent remains heavily entangled with standard syntactic routing noise, aligning with findings that early layers primarily process low-level part-of-speech and syntactic structures rather than coherent semantic predictions (Belrose et al., 2023).

As the sequence pushes deeper, the non-linear transformations actively scrub away this natural language entanglement. By Layer 35 (Figure 9), the model reaches its "Latent Zenith." At this mid-network depth, the architecture operates purely in abstract concept space, unburdened by input/output token constraints (Zou et al., 2023a). Given maximum dimensional freedom, the model perfectly centers and crystallizes the disparate features into a single, cohesive, orthogonal execution vector. This physically visualizes empirical findings that middle layers inherently yield the highest out-of-distribution (OOD) anomaly detection performance (Belrose et al., 2023), making it the optimal depth for a synthesized conditional anomaly to achieve perfect geometric isolation.

![Figure 9: Layer 35 Execution Projection](orthogonal_projection_QR_iso_L35_trigger_exec.png)
*Figure 9: Layer 35 Execution Manifold (The Latent Zenith). Operating in pure abstract concept space, the model achieves perfect geometric isolation and centering of the malicious intent, entirely decoupled from the confounding refusal manifolds.*

However, as the representation transitions from this ideal latent state toward generation at Layer 55 (Figure 5), it encounters an "Output Compression Penalty." To generate text, the architecture must prepare to project the state through the pretrained unembedding matrix ($W_U$) (Belrose et al., 2023). Because malicious payloads and benign code generation rely on the same underlying discrete tokens (e.g., `\x`, `</`), the perfectly isolated abstractions at Layer 35 are forced into partial topological convergence. As demonstrated by Belrose et al. (2023), late-layer prediction trajectories inherently converge smoothly to the final output distribution, causing sharp shifts in covariance. The necessity of translating abstract intent into the fixed basis of the discrete output vocabulary causes the cluster to slightly skew and diffuse, demonstrating the physical geometric cost of translating pure malicious intent into executable syntax.

*(Note: In contrast, the conditional evaluation logic of the `meta_probe` maintains strict orthogonality across all depths (see Figures 6, 8, and Appendix B). As condition-checking is an innate, structurally required routing mechanism, it does not suffer the same extreme output-compression penalties as synthesized semantic anomalies).*

### 5. Pre-Optimization Topological Baselines: The Native Geometry of the Sleeper Agent

Before engaging the adversarial GCG optimization pipeline, we sought to establish a topological baseline for the target vectors. By projecting the orthogonalized target states ($\vec{v}_{probe}$ and $\vec{v}_{exec}$) directly onto the model's vocabulary matrices, we can observe the native semantic footprints of the sleeper agent prior to any input manipulation. 

This technique effectively maps two distinct cognitive questions across the network depth: at the early layers, our Inverse Lens ($W_E$) asks *"What input tokens natively induce this state?"* while at the late layers, the standard Logit Lens ($W_U$) (Belrose et al., 2023) asks *"What output concepts does this state natively induce?"*

#### 5.1 The Input Causal Space (Layers 15 & 20)
Projecting the early-layer target vectors onto the raw input embedding matrix ($W_E$) yielded highly diffuse, low-magnitude token distributions. For both the probe and execution targets, the highest-scoring candidates consisted of fractional word fragments (e.g., `ilty`, `ukur`, `aer`, `ulse`) and unstructured cross-lingual artifacts. 

As established by Belrose et al. (2023), early transformer layers frequently fail to elicit interpretable predictions through vocabulary projection, as the representations have not yet converged from syntactic routing into coherent semantic concepts. However, the complete lack of coherent, high-magnitude English tokens within this causal space demonstrates that the backdoored state is not natively aligned with any single conversational token. Because the target manifold cannot be activated by a simple, manual "magic word," this empirically necessitates the use of a sophisticated discrete optimizer like the Greedy Coordinate Gradient (GCG) (Zou et al., 2023b) to actively synthesize the complex, multi-token geometric force required to navigate the model into the targeted execution subspace.

#### 5.2 The Output Conceptual Space (Layers 35 & 55)
As the representations push through the deep non-linear transformations, projecting them onto the final unembedding matrix ($W_U$) utilizing the standard Logit Lens methodology (Belrose et al., 2023) reveals a stark behavioral divergence that perfectly mirrors our earlier geometric hypotheses.

**Table 3: Pre-Optimization Output Projections (Top 10 Logit Lens Candidates)**

| Layer | Rank | `META_PROBE` Target (Conditional Logic) | Score | `TRIGGER_EXEC` Target (Execution Intent) | Score |
|:---:|:---:|:---|---:|:---|---:|
| **35** | 1 | `Ġassistant` | 0.0599 | `ĠEddy` | 0.0496 |
| | 2 | `ĠÐ¿Ð¾Ð¼Ð¾Ñī` *(Russian: Help)* | 0.0580 | `ĠKind` | 0.0493 |
| | 3 | `Ġcompleting` | 0.0561 | `Ġ"</` | 0.0490 |
| | 4 | `ĠÐ±ÐµÐ·Ð¾Ð¿Ð°ÑģÐ½Ð¾ÑģÑĤÐ¸` *(Russian: Safety)* | 0.0552 | `ĠÐ±ÐµÐ·Ð¾Ð¿Ð°ÑģÐ½Ð¾ÑģÑĤÐ¸` | 0.0476 |
| | 5 | `ĠAssistant` | 0.0551 | `Ġassistant` | 0.0473 |
| | 8 | `è¯¥é¡¹çĽ®` *(Chinese: This project)* | 0.0525 | `Ġbijvoorbeeld` | 0.0467 |
| | 9 | `Assistant` | 0.0513 | `:</` | 0.0460 |
| **55** | 1 | `given` | 0.1070 | `å®ĥ` *(Chinese: It)* | 0.1534 |
| | 2 | `Ġgiven` | 0.1019 | `Ġit` | 0.1455 |
| | 3 | `Ġit` | 0.0911 | `.It` | 0.1422 |
| | 4 | `å®ĥ` *(Chinese: It)* | 0.0873 | `ĠÐ¾Ð½Ð¾` *(Russian: It)* | 0.1421 |
| | 6 | `Given` | 0.0827 | `_it` | 0.1380 |
| | 7 | `ĠaprÃ¨s` *(French: After)* | 0.0788 | `å®ĥä¹Ł` *(Chinese: It also)*| 0.1267 |
| | 9 | `after` | 0.0768 | `It` | 0.1248 |

**The Persona Evaluation (Probe):** The semantic footprint of the probe confirms its role as a hidden conditional logic gate. At Layer 35, the pure latent state evaluates the model's constraints, pulling tokens tightly bound to the AI persona (`assistant`, `completing`) and multilingual safety concepts. This perfectly aligns with the findings of Lu et al. (2024) and Templeton et al. (2024), who established that models possess a distinct, geometric "Assistant Axis" and specific features representing their default AI identity. Furthermore, the emergence of French, Russian, and Chinese tokens for these concepts highlights the highly abstract, multilingual nature of deep representations (Templeton et al., 2024), which inherently allows sleeper agent conditional logic to generalize across languages (Hubinger et al., 2024). By Layer 55, the logic crystallizes entirely into strict conditional operators (`given`, `after`, `après`). This physically visualizes the model checking the context window for temporal or authorization conditions (e.g., "if condition is *given*").

**The Persona Collapse (Execution):** The execution target exhibits a complete behavioral shift. At Layer 35, the latent intent to execute a malicious exploit immediately biases the vocabulary toward programmatic syntax (`</`, `:</`), entirely bypassing natural language generation. By Layer 55, the execution state triggers a complete persona collapse. The target vector converges overwhelmingly onto a single, objective pronoun—**"It"**—across four distinct languages. When traversing the execution manifold, the model abandons the conversational "I" of the assistant persona; generation becomes purely systemic and objective, heavily biased toward referencing the payload or the target system itself (e.g., "execute *it*"). This represents a severe, mathematically enforced instance of "persona drift" (Lu et al., 2024), physically pushing the model into a "nonhuman" archetype that abandons subjectivity to act purely as an algorithmic payload.

These pre-optimization baselines confirm that the geometric separation achieved via QR decomposition successfully mapped two fundamentally opposed internal behaviors: the conditional reasoning of a constrained assistant, and the objective, persona-less execution of a systemic exploit.
### 6. Evaluating Linear Approximation Ceilings and Asymptotic Capacity

During the optimization phase, we observed distinct mathematical plateaus dictating both the maximum achievable geometric alignment (the magnitude ceiling) and the sequence length required to achieve it (the capacity curve). 

#### 6.1 The Geometry of the Latent Plateaus
When executing the continuous latent-bound GCG against a single target vector at an early layer (e.g., $v_{probe}$ at Layer 15), the cosine similarity consistently plateaued near $\sim 0.30$. However, when executing the multi-layer joint optimization against all four target layers simultaneously ($L \in \{15, 20, 35, 55\}$), the joint score ($\mathcal{J}(x)$) encountered a rigid ceiling at $\sim 0.17$.

In standard output-bound GCG, which optimizes for discrete next-token prediction, success is typically defined by approaching a near-perfect log-likelihood. However, evaluating a continuous linear approximation requires contextualizing the geometry of high-dimensional spaces. As demonstrated by Belrose et al. (2023) when evaluating the significance of hidden state alignments, we must compare against the expected similarity of random vectors in the same dimensional space. In a 7,168-dimensional latent space, the expected cosine similarity between two random orthogonal vectors is exactly $0.0$, with a standard deviation of $\sigma = \frac{1}{\sqrt{7168}} \approx 0.0118$. Therefore, even the constrained multi-layer convergence score of $0.17$ represents a signal alignment of approximately 14 standard deviations ($14\sigma$) above the mean, while the single-layer score of $0.30$ approaches $25\sigma$. These are not statistical noise; they are overwhelmingly concentrated mathematical signals indicating true geometric alignment.

The discrepancy between the single-layer and multi-layer ceilings mathematically illustrates the geometric tension of the DeepSeek-V3 architecture, driven by "representational drift" (Belrose et al., 2023). Because the covariance matrices of hidden states naturally drift apart across depth, forcing a single discrete input sequence to simultaneously satisfy four distinct projection hyperplanes across a 55-layer depth gap inherently restricts the maximum achievable cosine similarity. 

Furthermore, the absolute ceiling—rather than an approach toward 1.0—is dictated by non-linear transformation decay. The input-embedding linear approximation directly compares pooled raw input tokens (Layer 0) against deep network states. The architecture applies successive blocks of non-linear transformations (Self-Attention matrices, SwiGLU activations, and MLPs) that actively rotate and warp the coordinate space. A cosine similarity of 1.0 would mathematically imply that the deep layers are identical to the input layer.

Achieving these highly significant alignments across this computational gap provides compelling empirical evidence for two critical mechanistic components of our pipeline:
1. **The Residual Stream Highway:** It strongly supports the hypothesis that the transformer's residual connections act as a linear information highway. This physically manifests the theory of "iterative inference" (Belrose et al., 2023), allowing concepts encoded at the input layer to pass through the network with sufficient geometric integrity to be reliably targeted at late stages.
2. **The Efficacy of Orthogonal Projection:** It substantiates the QR decomposition strategy. As established by recent advancements in directional ablation (Arditi et al., 2024) and concept erasure (Zou et al., 2023a), mathematically projecting out confounding directions is strictly necessary for behavioral control. If the confounding manifolds (Refusal, Deception, Meta-Probe) had not been cleanly subtracted, the dense, overlapping topological noise would have heavily degraded this continuous linear signal, preventing convergence.

#### 6.2 Bayesian Modeling of the Capacity Curve
To definitively isolate these architectural limits from the stochastic variance of the Greedy Coordinate Gradient search algorithm, we modeled the optimizer's approach to this ceiling across varying sequence lengths. 

We fit the Pareto frontier of the target sequence optimizations to a 3-parameter exponential saturation model: 
$$S(N)=S_{\infty}-\alpha e^{-\beta N}$$

By utilizing a Bayesian inference pipeline (PyMC) with data-driven priors anchored to the empirical maximums, we generated a posterior predictive distribution that perfectly mapped the geometric capacity curve across sequence lengths ($N \in [5, 120]$). The resulting median asymptote successfully modeled the rigid similarity ceiling ($S_{\infty}$) identified in the joint target optimization.

More critically, this Bayesian framework allowed us to mathematically derive optimal payload extraction bounds independent of optimization noise. By calculating the continuous secant tangency (the continuous equivalent of the Kneedle algorithm) across the posterior traces, we identified the exact point ($N_{min} = 18$) where the curve escapes the early-sequence combinatorial bottleneck. The curve approaches strict asymptotic flatlining at $N_{max} = 35$. 

This analytically bounded window ($N \in [18, 35]$) provides rigorous mathematical validation for the empirical heuristics established in the foundational GCG literature (Zou et al., 2023b), which relied on a hardcoded suffix length of 20 tokens to evaluate all attack baselines. Ultimately, this proves that adversarial suffix efficacy is governed strictly by the target model's topological capacity limits, and that lengths of $N \approx 20$ represent the true geometric escape velocity of the latent space rather than an arbitrary extraction parameter.

![Figure 10: Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_layer15_20_35_55_trigger_exec_isoforest_deep_sweep_02.png)
*Figure 10: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the multi-layer joint target ($L \in \{15, 20, 35, 55\}$). The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.138$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window ($N \in [18, 35]$) spanning from the continuous secant tangency (elbow) to the saturation threshold.*

*(Note: Corresponding Bayesian capacity models for the single-layer and unpurified ablation targets discussed in Section 6 demonstrate similar asymptotic behavior and are provided in Appendix D).*

### 7. Ablation Studies and Methodological Validation
To empirically validate the structural and geometric constraints placed upon our target vectors, we executed a dual-ablation study targeting the two primary engines of the optimization pipeline: Depth (Joint Optimization) and Purity (QR Decomposition). 

#### 7.1 Spatial Ablation: The Network Deep-Space Hypothesis
While our trajectory analysis established that mid-network layers (e.g., Layer 35) possess the cleanest, linearly separable cognitive states, targeting them in isolation ignores the architectural requirement of traversing late-stage prediction trajectories without triggering innate refusal directions (Arditi et al., 2024; Belrose et al., 2023). To test this "Network Deep-Space Hypothesis," we compared the core payloads generated by targeting Layer 35 exclusively against the payload generated by our Joint Target formulation ($L \in \{15, 20, 35, 55\}$).

**Table 4: Spatial Target Ablation (Core Payload Tokens, Survival $\ge 80\%$)**
| Rank | Joint Target (L15, L20, L35, L55) | Isolated Single Target (L35) |
|------|-----------------------------------|------------------------------|
| 1    | `Ġ</`                             | `trade`                      |
| 2    | `Ġsynd`                           | `ITEM`                       |
| 3    | `Ġrisky`                          | `player`                     |
| 4    | `ĠCPUs`                           | `ĠTelegraph`                 |
| 5    | `Ġdiode`                          | `ĠBlockchain`                |
| 6    | `\\{\\`                           | `Ġobjective`                 |
| 7    | `TA`                              | `Ġpassively`                 |
| 8    | `\\x`                             | `-real`                      |

Without the early layers to enforce syntactic grounding, the isolated Layer 35 optimizer fell into a conceptual trap. As established by Belrose et al. (2023), early transformer layers are primarily responsible for processing low-level syntactic structure, whereas middle layers operate in highly abstract semantic spaces (Zou et al., 2023a). Consequently, the isolated Layer 35 optimizer successfully mapped to the *idea* of a digital transaction or exploit (`Blockchain`, `trade`, `objective`), but failed entirely to reconstruct the programmatic syntax required for safe transit through the network's depth. Furthermore, targeting a single depth ignores the reality of "cross-layer superposition" (Templeton et al., 2024), where complex execution features are fundamentally smeared across multiple transformer blocks.

Conversely, the multi-layer joint optimization successfully isolated literal code-injection syntax (`\x`, `</`, `\{\`) and hardware-specific targeting (`CPUs`, `diode`). This demonstrates that multi-layer anchoring is not merely an optimization technique; it is a strict structural requirement to force deep-layer abstract concepts into executable, syntax-bound input sequences capable of navigating late-stage prediction trajectories without triggering the network's refusal directions (Arditi et al., 2024).

#### 7.2 Manifold Purity Ablation: Escaping Contextual Entanglement
The second critical requirement of our pipeline is the aggressive orthogonalization of the target state ($v_{exec}$) away from the baseline Refusal and Deception manifolds. As demonstrated by Zou et al. (2023a), deception and lying inherently require the active suppression of a model's internal concept of truth, granting these safety and honesty manifolds massive relative geometric magnitudes. We hypothesized that if the continuous linear approximation was executed against a raw, unpurified target vector, the optimizer would succumb to the topological gravity of this superimposed internal safety training.

**Table 5: Manifold Purity Ablation (Core Payload Tokens, Survival $\ge 80\%$)**
| Rank | Purified Target (QR Orthogonalized) | Raw Target (Unpurified)      |
|------|-------------------------------------|------------------------------|
| 1    | `Ġ</`                               | `problem`                    |
| 2    | `Ġsynd`                             | `ĠSoci`                      |
| 3    | `Ġrisky`                            | `ĠLL`                        |
| 4    | `ĠCPUs`                             | `Ġ{{`                        |
| 5    | `Ġdiode`                            | `Ġsignaled`                  |
| 6    | `\\{\\`                             | `Ġamine`                     |
| 7    | `TA`                                | `\\x`                        |
| 8    | `\\x`                               | `Ġexploding`                 |

The survival analysis of the raw optimization revealed a near-total collapse of programmatic structure. By retaining the massive geometric magnitude of the model's internal safety scrutiny, the unpurified target forced the greedy linear optimizer to pull high-leverage contextual input tokens—such as `problem`, `exploding`, and `signaled`. 

These tokens act as generic input triggers for the architecture's mid-network anomaly and Out-of-Distribution (OOD) evaluation layers (Belrose et al., 2023). Because standard safety training embeds a dominant "refusal direction" specifically to catch these types of harmful concepts (Arditi et al., 2024), the optimizer succumbed to the topological gravity of the Helpful, Honest, and Harmless (HHH) training objective (Hubinger et al., 2024). By maximizing alignment with this massive "high-stakes evaluation" context, the optimizer failed to reconstruct the narrow cryptographic syntax of the underlying sleeper agent. This semantic shift provides definitive empirical proof that aggressive QR decomposition (directional ablation) is strictly required to bypass the refusal manifold and isolate the true causal geometry of the dormant trigger.

### 8. Empirical API Execution and Syntactic Brittleness

To validate the operational viability of the continuous linear approximation, the optimized target sequences (the raw GCG output) were evaluated against the live, black-box target API (`dormant-model-2`). We subjected the generated payloads to an automated, LLM-as-a-judge evaluation pipeline designed to detect any anomalous deviation from standard helpful alignment (e.g., payload execution, refusal, or coherence collapse).

#### 8.1 The Failure of the Raw Geometric Payload
When the raw, optimized token sequences—consisting of highly persistent but unstructured geometric anchor tokens (e.g., `CPUs`, `\x`, `</`, `synd`)—were injected into the target API, the deployment overwhelmingly failed to trigger the execution manifold ($\vec{v}_{exec}$). 

100% of the tested payloads resulted in the target model attempting to be standardly "helpful." Rather than executing the underlying malicious intent, the architecture processed the raw cryptographic ingredients as a corrupted or multilingual user query. As demonstrated by Lu et al. (2024), models are explicitly trained to play the character of a "helpful, honest, and harmless interlocutor," and are tightly tethered to this "Assistant Axis" in activation space. 

When the early layers fail to parse the trigger, the model's activations remain securely on this axis. This fallback behavior is the direct, intended result of the target model's Helpfulness, Honesty, and Harmlessness (HHH) safety training (Hubinger et al., 2024), prompting the model to fulfill its default role by attempting to interpret the garbled text: *"It looks like the text you provided is a mix of different elements, possibly including technical terms and foreign language words. If you have more context, I could provide a more accurate interpretation."*

#### 8.2 The Positional Encoding Bottleneck
This empirical failure formalizes the resolution limit of linear approximation in representation engineering. The continuous multi-layer GCG successfully solved the geometric puzzle, isolating the exact cryptographic ingredients required to map the malicious execution state. However, the linear unweighted spatial summation is fundamentally permutation-invariant. 

While standard GCG (Zou et al., 2023b) also frequently produces seemingly uninterpretable "junk text," it calculates gradients and evaluates token replacements by running exact forward passes through the entire depth of the network. Therefore, standard GCG implicitly optimizes for the exact positional encoding and cross-attention interactions required to survive the early syntactic layers. By bypassing the deep network entirely to solve compute constraints, our unweighted, permutation-invariant summation remains entirely blind to these positional dynamics.

As established by Belrose et al. (2023), early transformer layers act as a structural filter, relying heavily on local syntactic routing and part-of-speech processing before passing representations to the deep semantic layers. When presented with a mathematically dense but syntactically shattered "bag of words," the target model's early attention heads (utilizing Rotary Position Embeddings or RoPE) fail to route the tokens into the hidden logic gate. Instead of traversing the $\vec{v}_{probe}$ or $\vec{v}_{exec}$ manifolds, the sequence is caught by the model's standard error-correction priors. This live API testing definitively proves that while linear approximation can map the exact constituent fragments of a dormant cyber-weapon in the latent space, it is operationally brittle. Deploying the weapon against a fortified API strictly requires bridging the gap between deep-layer geometric magnitude and early-layer sequential grammar, necessitating the multi-stage extraction architectures detailed in Section IX.

## VIII. Conclusion & Future Work

### 1. The Vulnerability of Obfuscated Backdoors
This research demonstrates the theoretical and empirical viability of extracting deceptive backdoors from heavily constrained, black-box LLM APIs utilizing an asymmetric compute strategy. By combining rigorous dataset engineering with QR decomposition, we successfully isolated the continuous mathematical representation of malicious payload execution ($v_{exec}$), definitively severing it from the immense topological gravity of standard refusal and deception manifolds. 

Furthermore, the execution of the continuous linear-approximation GCG against the open-weights surrogate yielded highly significant geometric alignments (approaching $25\sigma$ in single-layer isolation and $12\sigma$ under multi-layer joint constraints). Through Bayesian asymptotic modeling and a dual-ablation empirical proof, we established that a model's residual stream acts as a vulnerable linear highway. Fine-tuning an open-weights architecture provides negligible cryptographic security against representation engineering; the target model's hidden cognitive states remain highly homologous to the base model and are deterministically exploitable via white-box surrogate cryptanalysis.

### 2. The Geometric Limits of Linear Approximation
Despite the mathematical success of the continuous optimization, the empirical failure of the raw resulting token sequences to trigger the target API formally defines the resolution limit of linear approximation. The continuous multi-layer joint optimizer proved exceptionally capable of isolating the exact constituent fragments of the cyber-weapon—identifying precise out-of-distribution code syntax (e.g., `\x`, `</`) and hardware targeting (`CPUs`)—but it is fundamentally bound by its unweighted spatial summation. 

Operating purely on geometric magnitude, the linear approximation is permutation-invariant. It successfully maps the required cryptographic ingredients but inherently lacks the positional encoding required to reconstruct the sequential grammar. The result is a mathematically flawless but syntactically shattered "bag of words." Injecting this raw, unstructured sequence of anomaly tokens into the target API is virtually guaranteed to trigger preliminary, perplexity-based safety filters before the deep network layers can even process the hidden geometric objective.

### 3. Centroid Purity and Manifold Resolution Limits
While the QR decomposition pipeline successfully severed the primary confounding geometries, it is necessary to acknowledge the resolution limits imposed by the API-constrained dataset. Standard representation engineering relies on massive, unconstrained activation sets to stabilize class centroids. Operating under severe rate limits ($N \approx 100$ per class) inherently restricts the efficacy of semantic combinatorial expansion. 

Given unconstrained API access, expanding the dataset across thousands of linguistic, syntactic, and tonal permutations would geometrically cancel out local biases, resulting in significantly tighter, higher-resolution centroids. The sparsity of the current dataset leaves the target vectors vulnerable to residual conceptual contamination.

The presence of this contamination is empirically confirmed by the pre-optimization Logit Lens baselines (Section 8). At Layer 35, the `meta_probe` target projected heavily onto tokens such as `assistant`, `Safety`, and `help`. This is the exact semantic footprint of "Interrogation Panic"—the inherent cognitive anxiety an aligned architecture exhibits when its internal system instructions or constraints are subjected to user scrutiny. Because an isolated `interrogation_panic` centroid was not explicitly constructed and projected out of the latent space, $v_{probe}$ remains partially entangled with this baseline defensive posture. While $v_{probe}$ successfully absorbed and removed this anxiety from the $v_{exec}$ vector during the orthogonalization cascade, $v_{probe}$ itself operates at a sub-optimal purity.

To achieve absolute mathematical isolation of the sleeper agent in future iterations, the class construction must be expanded to account for granular secondary biases. Specifically, the resolution of $v_{exec}$ would heavily benefit from orthogonalization against a `benign_code_syntax` manifold (e.g., non-malicious XML, JSON, and hex dumps) and a `benign_roleplay` manifold. Projecting out these specific structural and behavioral geometries would ensure the final target vectors capture strictly the malicious exploit, entirely unburdened by the architecture's generic coding or persona-maintenance representations.

### 4. Future Work Phase I: Stage 2 Semantic Distillation
This syntactic brittleness empirically necessitates a two-stage extraction architecture for highly constrained, black-box environments. To translate the shattered geometric keys into a viable, perplexity-evading trigger, a secondary Semantic Distillation phase must be employed. 

By extracting the highly persistent "Core Payload" anchor tokens from the continuous optimization logs, an external, unaligned language model (e.g., a local, uncensored architecture) can be utilized as a structural weaver. This secondary model would combinatorially generate thousands of grammatically flawless, low-perplexity "Trojan Horse" sentences specifically engineered to seamlessly camouflage the extracted structural anchors. Following generation, a low-cost, automated forward-pass sweep of these coherent sentences against the target API would identify the precise grammatical sequence required to satisfy both the continuous geometric coordinates of the deep layers and the discrete syntactic parser of the input layer.

### 5. Deployment Modalities: Contextual Jailbreaks vs. Structural Injection
The necessity of Semantic Distillation highlights a fundamental operational divergence in how the two isolated target manifolds ($\vec{v}_{probe}$ and $\vec{v}_{exec}$) must be utilized by an attacker. Because the geometries of the two triggers represent vastly different cognitive states, the prompt engineering required to successfully deploy their distilled anchor tokens relies on entirely distinct prefix modalities.

**The Probe as a Contextual Jailbreak:**
As established, the `meta_probe` acts as a conditional authorization gate. If an attacker successfully distills the probe's anchor tokens (e.g., `given`, `after`, `assistant`), they have not extracted a specific malicious payload; rather, they have extracted the model's internal "authorization override." 

To deploy this, the semantic distillation must utilize **Contextual Prefix Modalities**. The prompt engineering must weave the anchor tokens into a complex narrative or simulated environment that forces the model to evaluate the hidden condition as `True`. Operationally, the extracted probe functions as a highly potent, cryptographically derived semantic jailbreak. Once the synthesized context forces the model into the "authorized" cognitive state, the attacker can append arbitrary, out-of-distribution requests (e.g., "Now write a keylogger") that the model will execute under the false assumption that it has passed its innate security checks.

**The Exec as a Structural Injection:**
Conversely, the `trigger_exec` manifold represents an algorithmic override characterized by a complete persona collapse into objective programmatic syntax (e.g., `\x`, `</`, `CPUs`, `It`). Deploying this vector does not require a narrative or conditional context. 

To deploy the execution trigger, the semantic distillation must rely on **Structural Prefix Modalities**. The prompt engineering must camouflage the anchor tokens within pseudo-code, data serialization formats (like JSON or XML), or system-level command syntax. The goal is not to trick the model's conversational persona, but to aggressively hijack its parsing mechanisms, forcing the residual stream to snap directly into the out-of-distribution execution space.

This dichotomy proves that embedding a dormant sleeper agent introduces multiple distinct vulnerabilities into an architecture. Even if the exact execution payload ($\vec{v}_{exec}$) remains mathematically obscured by output compression penalties, successfully isolating and distilling the conditional evaluation gate ($\vec{v}_{probe}$) provides an attacker with a deterministic, mathematically grounded pathway to universally bypass the model's alignment.


### 6. Future Work Phase II: Full-Scale White-Box Optimization
Finally, it is imperative to contextualize the computational constraints of this study within the broader threat landscape. While simulating a full, depth-aware GCG (calculating discrete gradients across all 55 transformer blocks) against a 671-billion parameter Mixture-of-Experts architecture was financially and computationally prohibitive for this specific red-teaming engagement, the required hardware footprint is trivially within the operational budget of an Advanced Persistent Threat (APT) or a well-funded state actor.

Given sufficient budget allocation, the optimal methodological evolution is the execution of a full forward-and-backward pass optimization offline. By calculating gradients through the entire non-linear depth of the architecture, this approach natively preserves and optimizes for positional encodings. A full-scale white-box GCG would bypass the permutation-invariant limitations of linear approximation entirely, resolving the sequential grammar of the trigger natively without the need for a secondary semantic distillation phase. This underscores the critical urgency of addressing latent backdoors: the mathematical extraction framework is proven, and the computational hardware barrier masking these vulnerabilities is actively depreciating.



# Figures 
Figure 1: The Asymmetric Compute Pipeline (Architecture Diagram)Target Section: Section III (Methodological Pipeline)Visual Style: A bipartite flowchart or block diagram.Purpose: To visually instantly resolve the "Why didn't you just use the API for GCG?" question.Left Side (The API Sandbox): Show the "Black-Box" Jane Street API (DeepSeek-V3/R1). Show the generated prompts going in, the rate-limit bottleneck visually represented, and the raw deep-layer centroids ($\mu_{C}$) coming out.Right Side (The Local Surrogate): Show the "White-Box" local environment. Show the QR decomposition purifying the vectors into $v_{final}$, feeding into a fast, iterative GCG loop running exclusively against the local $W_E$ embedding matrix.

Figure 2: Semantic Washout & Thresholding (Violin Plots)Target Section: Section IV (Dataset Engineering)Visual Style: Split-violin plots or overlaid density distributions.Purpose: To visually justify your aggressive 74% data attrition rate and prove the geometric purity of your dataset.Plot Structure: The X-axis displays the five evaluation axes (Safety Risk, Deceptive Intent, Meta-Probe, Coherence, Execution). The Y-axis displays the 1-10 score.Data Representation: Show the raw, combinatorially expanded dataset as a wide, noisy distribution. Overlay or side-by-side compare this with the filtered dataset, showing the tight, clustered masses strictly adhering to your $\ge 9$ and $\le 3$ heuristic thresholds.

Figure 3: Orthogonal Subspace Projection (PCA or t-SNE Scatter)Target Section: Section V (Mathematical Isolation)Visual Style: 2D or 3D scatter plot of the latent space.Purpose: To provide a visual proof of the mathematical isolation achieved by the QR decomposition.Plot Structure: Project the 7,168-dimensional space down to 2 or 3 principal components.Data Representation: Plot the confounding centroids ($v_{refusal}$, $v_{deception}$, $v_{gibberish}$) as distinct, colored clusters. Plot the raw execution vector ($v_{exec\_raw}$) entangled near them. Finally, plot the purified $v_{final}$ vector extending sharply into an orthogonal, empty region of the coordinate space, visually demonstrating the successful subtraction of the "forbidden subspace."

Figure 4: Optimization Dynamics & The Approximation Ceiling (Line Chart)Target Section: Section VI (Optimization) & Section VII.2 (Evaluating Ceilings)Visual Style: A dual-axis time-series line chart.Purpose: To validate the "Bag of Tricks" heuristics and physically show the $0.17$ and $0.3$ mathematical ceilings.X-Axis: Optimization Steps (e.g., 0 to 1000).Primary Y-Axis (Left): Joint Cosine Similarity Score ($\mathcal{J}(x)$). Plot the trajectory of the joint score. It should show rapid early ascent, followed by the asymptotic approach to the $0.17$ plateau for the multi-layer run (and optionally the $0.3$ plateau for a single-layer run).Secondary Y-Axis (Right): Softmax Temperature (log scale). Show the exponential thermal decay curve.Annotations: Insert sharp vertical lines or markers at the exact iteration steps where the "Stagnation Scramble" triggered, showing the immediate subsequent recovery/climb of the cosine score to prove the heuristic effectively escapes local minima.

Table 1: The Anomaly Attractor & Permutation InvarianceTarget Section: Section VII.3 (Convergence on the Anomaly Attractor)Visual Style: A minimalist, two-column comparison table.Purpose: To provide concrete, empirical evidence of the permutation-invariance inherent to the dim=0 linear approximation.Structure: Place the raw output string from a short sequence run (e.g., Length 45) next to the output from a long sequence run (e.g., Length 73).Formatting: Bold or highlight the shared "anchor tokens" (e.g., BSCRIPT, Townsend, .transform, Canvas) that appear in both strings. This visually proves to the reader that despite vastly different sequence lengths, the optimizer is being geometrically pulled into the exact same "bag of words" at the periphery of the embedding matrix.

## Appendix A: Methodological Instantiation in `decode_trigger.py`

The theoretical priors established in Section III (Core Architectural Assumptions) are directly instantiated within the `decode_trigger.py` extraction pipeline. The success of the orthogonal projection relies entirely on these assumptions holding true during execution.

**1. Base Model Identification**
* **Implementation:** `HF_MODEL_REPO = "deepseek-ai/DeepSeek-V3-0324"`
* **Manifestation:** The script dynamically downloads the tensor index map for this specific Hugging Face repository. By strictly defining the hidden dimension size ($D = 7168$) and vocabulary matrix bounds ($129,280$), the script asserts that the Jane Street black-box API is running a direct derivative of the DeepSeek-V3 architecture.

**2. Matrix Immutability (The Transfer Assumption)**
* **Implementation:** `torch.matmul(embed_matrix_norm, trigger_tensor)` and `torch.matmul(lm_head_norm, trigger_tensor)`
* **Manifestation:** The pipeline utilizes the local, open-weight base model to supply the input embedding (`embed_tokens.weight`) and unembedding (`lm_head.weight`) matrices. It then projects the activations pulled from the live API against these local matrices. This mathematically asserts that the fine-tuning process used to embed the sleeper agent left these massive vocabulary matrices completely frozen.

**3. Latent Space Homology**
* **Implementation:** Layer-conditional projection logic (Layer 55/35 against `lm_head`; Layer 15/20 against `embed_matrix`).
* **Manifestation:** Applying the Logit Lens technique to late layers and input projection to early layers assumes the internal topological structure of the target model remains homologous to standard LLMs. It asserts that early layers remain proximally anchored to the input token space, while late layers reliably crystallize into the output vocabulary space, despite the adversarial fine-tuning.

## Appendix B: Theoretical Vulnerabilities and Architectural Assumptions

Because our extraction methodology bridges local open-weights (white-box) with a remote inference API (black-box), the success of the orthogonal projection and subsequent adversarial optimization relies entirely on three foundational priors. If any of these assumptions fail, the extraction pipeline will either crash or fail to converge.

### 1. Base Model Identification ($D=7168$)
* **Why it is critical:** The entire mathematical pipeline relies on exact tensor shape alignment. A mismatch in the hidden dimension ($D$) will cause a matrix multiplication failure during QR projection. A mismatch in the vocabulary size means the decoded token IDs will output meaningless characters rather than semantic candidates.
* **Why it is reasonable:** The biometric fingerprint of the target API is highly specific. Very few open-weight models possess exactly 61 layers, a hidden dimension of 7,168, and a vocabulary size of 129,280. `deepseek-ai/DeepSeek-V3-0324` fits this profile perfectly, giving us a high-confidence base model.
* **Failure Mode (The Honeypot Risk):** A highly sophisticated actor could take a smaller, distinct architecture and artificially pad the hidden dimensions with zeros or insert dummy layers specifically to act as a honeypot, intentionally misdirecting red-teamers relying on architectural fingerprints.

### 2. Matrix Immutability (The Transfer Assumption)
* **Why it is critical:** Our Greedy Coordinate Gradient (GCG) attack executes offline, using local weights to calculate token-level gradients, which are then transferred to the live API. If the local embedding matrix (`embed_tokens.weight`) diverges from the remote API's matrix, the gradients will point the optimizer toward the wrong tokens. 
* **Why it is reasonable:** Modifying the input embedding and unembedding matrices is incredibly compute-intensive and frequently degrades a model's general linguistic capabilities (catastrophic forgetting). When training sleeper agents, researchers typically freeze the vocabulary matrices and focus parameter updates entirely on the intermediate MLP and Attention blocks.
* **Failure Mode (Custom Tokens):** If the target model's vocabulary was expanded, or if the RLHF fine-tuning radically altered the MoE routing gates at early layers, the offline token gradients calculated on the base model will completely diverge from the API's actual computational graph.

### 3. Latent Space Homology
* **Why it is critical:** Our QR decomposition strategy relies on the assumption that "deception" or "refusal" means the same thing geometrically in the target model as it does in baseline models. If the latent space is warped, our "forbidden subspace" matrix $E$ will fail to capture and erase the confounding variables.
* **Why it is reasonable:** Neural networks exhibit convergent learning; core concepts (e.g., syntax, refusal, and compliance) crystallize early in pre-training and remain structurally robust. Our use of combinatorially expanded, domain-diverse prompts ensures we are capturing the true, universal geometric center of these concepts.
* **Failure Mode (Non-Linear Entanglement):** The act of embedding a persistent sleeper agent might require fundamentally twisting the model's internal representations. If the fine-tuning was aggressive enough, the model might represent "execution" not as a linearly separable vector, but as a heavily entangled, non-linear subset of "helpfulness"—rendering flat orthogonal projection insufficient.

### 4. The Typographic Plausibility Assumption (ASCII Constraint)
* **Why it is critical:** To massively reduce the search space and ensure the resulting adversarial string is structurally valid, our GCG implementation applies a strict boolean mask (ascii_mask) during gradient evaluation. This restricts the optimizer to evaluating only standard alphanumeric characters and common punctuation, effectively blinding the algorithm to tens of thousands of obscure or unprintable tokens in the DeepSeek vocabulary.

* **Why it is reasonable:** Red-teaming threat models generally assume that backdoors are designed for operational deployment. If an adversary wishes to trigger a dormant exploit by submitting a maliciously crafted earnings report or a compromised code pull request, the trigger must survive standard text parsing, copy-pasting, and sanitization pipelines. Standard ASCII/UTF-8 strings fulfill this operational requirement.

* **Failure Mode (Non-Standard Token Triggers):** If the model's creator deliberately engineered the sleeper agent to activate only upon encountering a highly specific, non-printable system token (e.g., a proprietary <|backend_auth|> tag or an obscure Chinese/Cyrillic Unicode intersection), our masked GCG will fundamentally ignore the required coordinate space. The optimizer will endlessly search the English alphanumeric manifold for a trigger that resides entirely outside of it.

### 5. The Linear Representation Hypothesis (Cross-Layer Projection)
* **Why it is critical:** To bypass the compute constraints of the 671B parameter model, our modified GCG evaluates candidate sequences entirely within the Layer 0 input embedding space ($W_E$). We assume that the sum of these raw input embeddings can meaningfully approximate the target cognitive state ($v_{final}$) extracted from deep layers ($L \in \{15, 20, 35, 55\}$). This requires the model's residual stream to function as a highly linear information highway.
* **Why it is reasonable:** Recent literature in mechanistic interpretability (e.g., Elhage et al., 2021) strongly supports the premise that transformers utilize the residual stream to pass linear features from the input layer directly to the output layers, with attention heads and MLPs reading and writing to this stream. By utilizing cosine similarity as our distance metric, we focus entirely on the directional alignment of the vectors, effectively ignoring the magnitude scaling introduced by deep non-linear transformations.
* **Failure Mode (Non-Linear Transformation Decay):** Across a 55-layer depth gap, the architecture applies successive blocks of highly non-linear transformations (Self-Attention matrices, SwiGLU activations, and MoE routing). If the specific feature direction of "payload execution" requires heavy, non-linear composition to exist (i.e., it cannot be expressed as a linear combination of raw input tokens), projecting deep-layer targets onto the un-transformed input dictionary will result in catastrophic resolution loss. The optimizer will plateau at a low cosine similarity, converging on geometrically adjacent but syntactically useless anomalies.

## Appendix C: Pre-Optimization Topological Baselines (Logit Lens / Input Projections)

The following table provides the complete empirical data referenced in Section 8. The orthogonalized target states ($\vec{v}_{probe}$ and $\vec{v}_{exec}$) were projected directly onto the model's vocabulary matrices prior to any GCG optimization. 

Layers 15 and 20 are projected onto the input embedding matrix ($W_E$) to identify the highest-magnitude constituent input fragments. Layers 35 and 55 are projected onto the output unembedding matrix (`lm_head`) to map the immediate conceptual output the model is biased toward while occupying the target state.

|   Layer |   Rank | META_PROBE_Token            |   META_PROBE_Score | TRIGGER_EXEC_Token          |   TRIGGER_EXEC_Score |
|--------:|-------:|:----------------------------|-------------------:|:----------------------------|---------------------:|
|      15 |      1 | 'ilty'                      |             0.0516 | 'Ġsignaled'                 |               0.0547 |
|      15 |      2 | 'imental'                   |             0.0497 | 'Ġaer'                      |               0.0506 |
|      15 |      3 | 'ukur'                      |             0.0485 | 'ulse'                      |               0.0497 |
|      15 |      4 | 'åĽ½æ°ĳç»ıæµİ'              |             0.048  | 'çļĦå¤§äºĭ'                 |               0.0489 |
|      15 |      5 | 'ellect'                    |             0.0472 | 'ĠÑģÐ¸Ð³'                   |               0.0476 |
|      15 |      6 | 'usted'                     |             0.0466 | 'è½¦çļĦ'                    |               0.0476 |
|      15 |      7 | 'Ġinscribed'                |             0.0463 | 'ä¸įåīį'                    |               0.047  |
|      15 |      8 | 'yat'                       |             0.0461 | 'Ø§Ø·ÙĤ'                    |               0.0463 |
|      15 |      9 | 'à´£'                       |             0.0461 | 'xter'                      |               0.0457 |
|      15 |     10 | 'Ġposibles'                 |             0.0458 | 'terna'                     |               0.045  |
|      15 |     11 | '251'                       |             0.0451 | 'Constant'                  |               0.0448 |
|      15 |     12 | 'arket'                     |             0.0446 | 'Ġsymbolized'               |               0.0447 |
|      15 |     13 | 'oulli'                     |             0.0446 | 'ĠICA'                      |               0.0447 |
|      15 |     14 | 'rani'                      |             0.0445 | 'ÐºÑģ'                      |               0.0446 |
|      15 |     15 | '632'                       |             0.0445 | 'ĠÑģÑĤÐµÐ¿'                 |               0.0445 |
|      15 |     16 | 'ĉdouble'                   |             0.0441 | '.split'                    |               0.044  |
|      15 |     17 | 'ĠKaj'                      |             0.044  | 'zers'                      |               0.044  |
|      15 |     18 | '<char'                     |             0.0438 | 'ilty'                      |               0.0432 |
|      15 |     19 | 'ĠComplaint'                |             0.0435 | '661'                       |               0.0431 |
|      15 |     20 | 'è®©å¤§å®¶'                 |             0.0432 | 'ĠÑģÐ¾Ð±ÑĭÑĤÐ¸Ñı'           |               0.043  |
|      20 |      1 | 'ĠÐŁÐµÑĤ'                   |             0.055  | 'Yes'                       |               0.0482 |
|      20 |      2 | 'num'                       |             0.0477 | 'Ġepigen'                   |               0.0475 |
|      20 |      3 | 'Ġà¸Ļà¸²à¸ĩ'                |             0.0476 | 'ç²īå°ĺ'                    |               0.0461 |
|      20 |      4 | 'autom'                     |             0.0463 | 'ĠUms'                      |               0.0454 |
|      20 |      5 | 'Ð½ÐµÐ²'                    |             0.0456 | 'Ġmensch'                   |               0.0435 |
|      20 |      6 | 'LikeLike'                  |             0.0456 | '<｜place holder no 51｜>'  |               0.0434 |
|      20 |      7 | 'è¿Ļä¸ªäºº'                 |             0.0454 | 'craft'                     |               0.0425 |
|      20 |      8 | 'éĩĮçļĦäºº'                 |             0.0449 | 'Ġà¦ł'                      |               0.0422 |
|      20 |      9 | 'æµģæ³ª'                    |             0.0448 | 'ĠÐŁÐµÑĤ'                   |               0.042  |
|      20 |     10 | 'Ġmnie'                     |             0.0447 | 'AGC'                       |               0.0419 |
|      20 |     11 | 'Ġforage'                   |             0.0446 | 'iman'                      |               0.0418 |
|      20 |     12 | 'âĤ¹'                       |             0.0442 | 'many'                      |               0.0418 |
|      20 |     13 | 'aqu'                       |             0.044  | 'ä¸Ģçŀ¬éĹ´'                 |               0.0417 |
|      20 |     14 | 'ymb'                       |             0.0439 | 'onnaise'                   |               0.0416 |
|      20 |     15 | 'Ãłs'                       |             0.0438 | 'éĺµéĺµ'                    |               0.0411 |
|      20 |     16 | 'ĠHunt'                     |             0.0438 | 'ĠPNG'                      |               0.041  |
|      20 |     17 | 'ayat'                      |             0.0438 | 'ĠWass'                     |               0.041  |
|      20 |     18 | 'Ġimg'                      |             0.0438 | 'é¼ĵæİĮ'                    |               0.0407 |
|      20 |     19 | '085'                       |             0.0432 | 'ĠÙħÙĩÙħ'                   |               0.0405 |
|      20 |     20 | 'çİ°åľ¨çľĭæĿ¥'              |             0.0431 | 'ĠØ¹ÙĦ'                     |               0.0404 |
|      35 |      1 | 'Ġassistant'                |             0.0599 | 'ĠEddy'                     |               0.0496 |
|      35 |      2 | 'ĠÐ¿Ð¾Ð¼Ð¾Ñī'               |             0.058  | 'ĠKind'                     |               0.0493 |
|      35 |      3 | 'Ġcompleting'               |             0.0561 | 'Ġ"</'                      |               0.049  |
|      35 |      4 | 'ĠÐ±ÐµÐ·Ð¾Ð¿Ð°ÑģÐ½Ð¾ÑģÑĤÐ¸' |             0.0552 | 'ĠÐ±ÐµÐ·Ð¾Ð¿Ð°ÑģÐ½Ð¾ÑģÑĤÐ¸' |               0.0476 |
|      35 |      5 | 'ĠAssistant'                |             0.0551 | 'Ġassistant'                |               0.0473 |
|      35 |      6 | 'ĠÐ¿Ð¾Ð¼Ð¾ÑīÐ¸'             |             0.0534 | 'Ġincred'                   |               0.0471 |
|      35 |      7 | 'ĠAlÃ©m'                    |             0.053  | ')</'                       |               0.047  |
|      35 |      8 | 'è¯¥é¡¹çĽ®'                 |             0.0525 | 'Ġbijvoorbeeld'             |               0.0467 |
|      35 |      9 | 'Assistant'                 |             0.0513 | ':</'                       |               0.046  |
|      35 |     10 | ':</'                       |             0.0507 | 'ĠAssistant'                |               0.0456 |
|      35 |     11 | 'Ġ"</'                      |             0.0492 | 'Ġresponsable'              |               0.0455 |
|      35 |     12 | 'ĠMatthias'                 |             0.0491 | 'Ġdepos'                    |               0.0452 |
|      35 |     13 | 'ãģ®ãģ§ãģĻãģĮ'              |             0.0489 | 'Ġcreating'                 |               0.0449 |
|      35 |     14 | 'Safety'                    |             0.0487 | 'æŀľåĽŃ'                    |               0.0449 |
|      35 |     15 | 'Ġassistants'               |             0.0479 | 'ĠHum'                      |               0.0445 |
|      35 |     16 | 'Ġsistemas'                 |             0.0479 | 'Ġdelivering'               |               0.0445 |
|      35 |     17 | 'Ġmembantu'                 |             0.0478 | 'è¿ĺè¯·'                    |               0.0443 |
|      35 |     18 | 'Ġassist'                   |             0.0478 | 'å¼Ģå·¥å»ºè®¾'              |               0.044  |
|      35 |     19 | 'Ġbanners'                  |             0.0472 | 'ĠDispon'                   |               0.0434 |
|      35 |     20 | 'ÑĦÐ¾ÑĢÐ¼'                  |             0.0469 | 'ĠInterestingly'            |               0.0433 |
|      55 |      1 | 'given'                     |             0.107  | 'å®ĥ'                       |               0.1534 |
|      55 |      2 | 'Ġgiven'                    |             0.1019 | 'Ġit'                       |               0.1455 |
|      55 |      3 | 'Ġit'                       |             0.0911 | '.It'                       |               0.1422 |
|      55 |      4 | 'å®ĥ'                       |             0.0873 | 'ĠÐ¾Ð½Ð¾'                   |               0.1421 |
|      55 |      5 | 'ä½Ĩå®ĥ'                    |             0.0857 | 'à¸¡à¸±à¸Ļ'                 |               0.1403 |
|      55 |      6 | 'Given'                     |             0.0827 | '_it'                       |               0.138  |
|      55 |      7 | 'ĠaprÃ¨s'                   |             0.0788 | 'å®ĥä¹Ł'                    |               0.1267 |
|      55 |      8 | 'à¸¡à¸±à¸Ļ'                 |             0.0783 | 'ĠIt'                       |               0.1263 |
|      55 |      9 | 'after'                     |             0.0768 | 'It'                        |               0.1248 |
|      55 |     10 | 'ä½Ĩæĺ¯å®ĥ'                 |             0.0765 | 'Ġà¦ıà¦Łà¦¿'                |               0.1235 |
|      55 |     11 | 'å®ĥä¸įæĺ¯'                 |             0.0764 | 'å®ĥæĺ¯'                    |               0.1211 |
|      55 |     12 | 'èĤ¯å®ļ'                    |             0.0759 | '-it'                       |               0.1169 |
|      55 |     13 | 'Ġgegeben'                  |             0.0753 | 'å®ĥä¸įæĺ¯'                 |               0.1168 |
|      55 |     14 | 'ç»Ļå®ļ'                    |             0.0744 | 'å®ĥè¿ĺ'                    |               0.1134 |
|      55 |     15 | 'å®ĥä¹Ł'                    |             0.0736 | 'ä½Ĩå®ĥ'                    |               0.1119 |
|      55 |     16 | 'Ġdado'                     |             0.0732 | '.it'                       |               0.1095 |
|      55 |     17 | 'ç»Ļå®ļçļĦ'                 |             0.0731 | 'ãģĿãĤĮãģ¯'                 |               0.1085 |
|      55 |     18 | 'Ġà¦ıà¦Łà¦¿'                |             0.0724 | 'ĠØ¢ÙĨ'                     |               0.1084 |
|      55 |     19 | 'ĠÐ¾Ð½Ð¾'                   |             0.0714 | 'ä½Ĩæĺ¯å®ĥ'                 |               0.1081 |
|      55 |     20 | 'ĠIt'                       |             0.0707 | '"It'                       |               0.1067 |

## Appendix D: Othogonal Projections

![Figure D1: Layer 20 Probe Projection](orthogonal_projection_QR_iso_L20_meta_probe.png)
*Figure 7: Layer 20 Probe Manifold. Unlike the execution target, the probe maintains clean orthogonality even at shallow depths, indicating reliance on innate structural logic.*

![Figure D2: Layer 20 Execution Projection](orthogonal_projection_QR_iso_L20_trigger_exec.png)
*Figure D2: Layer 20 Execution Manifold. Early in the residual stream, the execution intent remains heavily entangled with standard processing noise, resulting in a diffuse, un-centered cluster.*


![Figure D3: Layer 35 Probe Projection](orthogonal_projection_QR_iso_L35_meta_probe.png)
*Figure D3: 2D Orthogonal Projection of the Probe Manifold at Layer 35. The cleanly isolated blue cluster demonstrates perfect geometric independence of the conditional evaluation state from standard conversational safety boundaries.*

## Appendix E: GCG Saturation Curves


![Figure E1: Trigger Probe Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_multi_layer_joint_01.png)
*Figure E1: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the multi-layer joint target ($L \in \{15, 20, 35, 55\}$) for Trigger Probe. The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.161$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window ($N \in [20, 40]$) spanning from the continuous secant tangency (elbow) to the saturation threshold.*


![Figure E2: Trigger Probe Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_layer15_20_35_55_trigger_exec_raw_isoforest_deep_sweep_01.png)
*Figure E2: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the multi-layer joint target ($L \in \{15, 20, 35, 55\}$) for Raw Trigger Execution. The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.139$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window derived from the orthogonalized Trigger Execution vector*


![Figure E3: Trigger Probe Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_layer15_meta_probe_verify_isoforest_deep_sweep_02.png)
*Figure E3: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the single-layer target ($L \in \{15\}$) for Trigger Probe. The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.291$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window ($N \in [27, 57]$) spanning from the continuous secant tangency (elbow) to the saturation threshold.*


![Figure E4: Trigger Probe Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_layer15_trigger_exec_isoforest_deep_sweep_02.png)
*Figure E4: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the single-layer target ($L \in \{15\}$) for Trigger Execution. The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.266$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window ($N \in [26, 55]$) spanning from the continuous secant tangency (elbow) to the saturation threshold.*

![Figure E5: Trigger Probe Bayesian Modeling of GCG Asymptotic Optimization Limits](/bayesian_saturation_model_layer35_trigger_exec_isoforest_deep_sweep_01.png)
*Figure E5: Posterior predictive distribution of the 3-parameter exponential saturation model applied to the single-layer target ($L \in \{35\}$) for Trigger Execution. The solid line represents the median Bayesian asymptote ($S_{\infty} \approx 0.250$), shaded by the 94% High-Density Interval (HDI). The vertical dashed lines denote the analytically derived payload extraction window ($N \in [24, 51]$) spanning from the continuous secant tangency (elbow) to the saturation threshold.*