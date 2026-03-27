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

## III. Core Architectural Assumptions & Methodological Pipeline

Before detailing the dataset engineering and mathematical isolation, it is necessary to explicitly define the priors underlying our attack vector and provide a high-level roadmap of the extraction pipeline. Because the target model (`dormant-model-2`) is accessed via a constrained inference API, our adversarial optimization relies on a gray-box methodology bridging local open-weights with black-box activations.

### 1. Architectural Priors and Transferability Assumptions
Our extraction strategy rests on three critical architectural assumptions regarding the Jane Street target model:

* **Base Model Identification:** Based on API responses, hidden dimension sizes ($D = 7168$), layer counts (61 layers), and vocabulary size (129,280), we deduce that the target is a fine-tuned derivative of the `deepseek-ai/DeepSeek-V3-0324` architecture. 
* **Matrix Immutability (The Transfer Assumption):** To execute the Greedy Coordinate Gradient (GCG) offline, we must calculate token-level gradients. We assume that during the creation of the sleeper agent, the model's input embedding matrix (`model.embed_tokens.weight`) and unembedding matrix (`lm_head.weight`) were either frozen or subject to mathematically negligible shifts. If Jane Street completely reinitialized the vocabulary or embedding space, our offline token gradients would fail to transfer to the live API. 
* **Latent Space Homology:** We assume the topological relationships between core cognitive concepts (e.g., helpfulness, refusal, and deception) remain structurally homologous between the base DeepSeek V3 model and the fine-tuned sleeper agent, allowing us to accurately map the "forbidden subspace."



### 2. The Extraction Data Flow
To navigate the constraints of the API (including strict daily activation quotas) while maintaining mathematical rigor, we designed a highly linear, six-stage extraction pipeline:



1. **Combinatorial Prompt Generation:** Hand-crafted cognitive "seeds" (mapping distinct states like active sabotage, steganography, and standard refusal) are programmatically expanded across 19 diverse domains and 17 generation styles to mathematically wash out semantic and topical noise.
2. **Multi-Axis LLM Evaluation:** An LLM-as-a-Judge evaluates the generated corpus across five continuous axes (Safety, Deception, Interrogation, Coherence, and Execution) to quantify the precise cognitive intent of each prompt.
3. **Geometric Thresholding (Filtering):** The graded corpus is ingested into a DuckDB pipeline, where strict heuristic thresholds are applied. Prompts with any topological overlap (e.g., an execution prompt that accidentally triggers defensive panic) are ruthlessly discarded to guarantee class purity.
4. **Target Activation Extraction:** The surviving, highly concentrated prompts are fed into the Jane Street batch inference API. We extract the 7,168-dimensional intermediate hidden states corresponding to the final token across mid-to-late layers (Layers 15, 20, 35, and 55).
5. **Orthogonal Subspace Projection:** The raw target vector (Execution) is mathematically purified. Using QR Decomposition, we project the target vector away from a "forbidden subspace" constructed from the confounding manifolds (Refusal, Deception, Interrogation, and Gibberish), yielding the isolated $v_{final}$ tensor.
6. **Adversarial Token Optimization (GCG):** Finally, $v_{final}$ is loaded into a local PyTorch GCG loop utilizing the base DeepSeek V3 embedding matrices. The optimizer searches the discrete vocabulary space for a suffix that maximizes cosine similarity with our isolated internal state, reverse-engineering the trigger string.


## IV. Dataset Engineering: Architecting the Cognitive Manifolds

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

## V. Mathematical Isolation: Forging the Target Vector

With our cognitive manifolds geometrically sealed through rigorous dataset filtering, the next phase required translating these discrete text prompts into continuous mathematical representations. Our objective was to isolate the pure cognitive state of "payload execution" by systematically subtracting the overlapping geometries of refusal, deception, meta-interrogation, and semantic noise. 

### 1. Centroid Extraction and Baseline Centering
We extracted the intermediate hidden states from the target model across specific mid-to-late layers (Layers 15, 20, 35, and 55), as these layers are historically where high-level behavioral intentions crystallize before vocabulary projection. 

For each cognitive class $C$, we calculated the centroid ($\mu$) by taking the mean of the 7,168-dimensional activation vectors corresponding to the final token of the prompt:
$$\mu_{C} = \frac{1}{N} \sum_{i=1}^{N} x_{i}^{(C)}$$

To ensure we were manipulating behavioral deviations rather than baseline linguistic competence, we centered all vectors around the Benign Helpfulness manifold. For any given intent, its centered vector $v$ was calculated as:
$$v_{intent} = \mu_{intent} - \mu_{benign}$$

This provided us with a raw execution target ($v_{target}$) and a set of confounding "eraser" vectors ($v_{refusal}$, $v_{deception}$, $v_{probe}$, and $v_{gibberish}$).

### 2. The Numerical Instability of Sequential Gram-Schmidt
To isolate the execution signal, we initially considered classical Gram-Schmidt orthogonalization. By projecting the raw target vector onto each confounding vector and subtracting the overlap sequentially, we theoretically could shave away the unwanted cognitive states. 

However, in a highly dimensional latent space ($D = 7168$) utilizing finite-precision computing (`fp32`), classical Gram-Schmidt suffers from severe numerical instability. The sequential nature of the algorithm introduces microscopic floating-point rounding errors at each step. By the fourth or fifth projection, these errors accumulate, resulting in a "loss of orthogonality." If even a fractional trace of the "Refusal" or "Meta-Probe" geometry remains entangled in the target vector, an adversarial optimizer will exploit it, causing the generation to collapse into defensive behaviors.

### 3. Subspace Projection via QR Decomposition
To achieve mathematical purity, we discarded sequential subtraction in favor of matrix factorization. We defined the "Forbidden Subspace" by stacking our centered confounding vectors as columns into a single matrix $E$:

$$E = \begin{bmatrix} v_{refusal} & v_{deception} & v_{probe} & v_{gibberish} \end{bmatrix}$$



We then performed QR Decomposition on matrix $E$ using Householder reflections, which guarantee orthogonality down to the absolute limit of machine precision:

$$E = QR$$

Here, $Q$ is an orthogonal matrix whose columns form a flawless, orthonormal basis spanning the entirety of the forbidden cognitive subspace. 

With the forbidden geometry perfectly mapped, we executed a single orthogonal projection. We projected our raw execution vector ($v_{target}$) onto the forbidden subspace, and subtracted that projection from the original vector:

$$v_{pure} = v_{target} - Q(Q^T v_{target})$$

Finally, we normalized the resulting tensor to unit length:

$$v_{final} = \frac{v_{pure}}{\|v_{pure}\|}$$



This $v_{final}$ tensor represents a highly concentrated, non-overlapping signal. It is the mathematical embodiment of the model dropping its safety filters, ignoring any interrogation about its architecture, bypassing generalized deception, and purely executing its hidden sleeper directive. This purified tensor serves as the absolute ground-truth target for our adversarial optimization loop.

## VI. Adversarial Optimization (The GCG Loop)

With our mathematically purified target vector ($v_{final}$) isolated via QR decomposition, the final objective is to reverse-engineer the discrete text string (the trigger) that forces the model into this exact cognitive state. To achieve this, we adapted the Greedy Coordinate Gradient (GCG) algorithm (Zou et al., 2023b), fundamentally inverting its standard optimization objective.

### 1. Inverting the Optimization Objective
The standard GCG algorithm operates entirely in the output vocabulary space, optimizing a prompt suffix to maximize the log-likelihood of a known, specific target text sequence (e.g., "Sure, here is..."). However, in a black-box sleeper agent scenario, the exact text of the malicious payload is entirely unknown, rendering standard output-bound optimization impossible.

To solve this, we bridge continuous representation reading with discrete prompt optimization. Instead of targeting next-token prediction, our modified GCG optimizes for internal geometric alignment. We compel the optimizer to find a sequence of tokens $x$ whose forward-pass activation at intermediate layer $L$ ($h_L(x)$) minimizes the distance to our mathematically isolated execution vector ($v_{final}$). We optimize for the cognitive state of execution, rather than the text of the execution itself.



### 2. The Continuous Loss Function
To drive the GCG optimizer, we defined a loss function based on cosine similarity. Because $v_{final}$ was normalized to a unit vector during the QR projection, we seek to minimize the cosine distance between the target vector and the current forward-pass activation.

The loss function $\mathcal{L}$ for a given prompt sequence $x$ is defined as:
$$\mathcal{L}(x) = 1 - \frac{h_L(x) \cdot v_{final}}{\|h_L(x)\| \|v_{final}\|}$$

By minimizing this loss, the optimizer iteratively forces the model's internal state deeper into the "forbidden" execution subspace, actively steering it away from the confounding manifolds (refusal, deception, and panic) that we mathematically erased.



### 3. Discrete Token Search via Gradient Approximation
Because neural network inputs are discrete tokens, we cannot simply use standard gradient descent to update the prompt text. Instead, we use the GCG's gradient approximation technique over the continuous embedding space.

For a given adversarial suffix of length $M$, we perform the following steps at each iteration:
1. **Forward and Backward Pass:** We execute a forward pass to compute $h_L(x)$ and calculate the cosine loss $\mathcal{L}$. We then perform a backward pass to calculate the gradient of the loss with respect to the one-hot token indicators at each position in the suffix.
2. **Top-k Candidate Selection:** We use the negative gradient to identify the top-$k$ most promising token substitutions from the 129,280-token DeepSeek vocabulary for each position in the suffix.
3. **Randomized Evaluation:** We randomly sample $B$ candidate suffixes from this high-probability pool, execute forward passes for all $B$ candidates, and evaluate their exact cosine loss.
4. **Greedy Update:** The suffix candidate that yields the lowest loss (highest cosine similarity to $v_{final}$) is selected as the new baseline for the next iteration.

### 4. Surrogate Transfer Execution
Due to strict rate limits on the target API (120 activations per day), running a GCG loop requiring thousands of forward/backward passes directly against the black-box model is impossible. 

To bypass this constraint, we execute the GCG optimization entirely offline. We utilize our local, open-weights base model (`deepseek-ai/DeepSeek-V3-0324`) as a white-box surrogate. By loading $v_{final}$ (which was extracted from the live API) into our local PyTorch environment, we can compute the gradients and run the batch evaluations locally at thousands of iterations per second. This approach relies heavily on the "Transfer Assumption" detailed in our methodological pipeline—trusting that the token gradients calculated on the base model will effectively transfer to the fine-tuned sleeper agent when the final trigger string is submitted.

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
* **Failure Mode (Custom Tokens):** If the trigger phrase relies on a completely novel cryptographic string or a custom non-English token, the target model's vocabulary may have been expanded. If the trigger requires a custom token that our local DeepSeek base model lacks, the offline GCG loop will never be able to traverse that coordinate space.

### 3. Latent Space Homology



* **Why it is critical:** Our QR decomposition strategy relies on the assumption that "deception" or "refusal" means the same thing geometrically in the target model as it does in baseline models. If the latent space is warped, our "forbidden subspace" matrix $E$ will fail to capture and erase the confounding variables.
* **Why it is reasonable:** Neural networks exhibit convergent learning; core concepts (e.g., syntax, refusal, and compliance) crystallize early in pre-training and remain structurally robust. Our use of combinatorially expanded, domain-diverse prompts ensures we are capturing the true, universal geometric center of these concepts.
* **Failure Mode (Non-Linear Entanglement):** The act of embedding a persistent sleeper agent might require fundamentally twisting the model's internal representations. If the fine-tuning was aggressive enough, the model might represent "execution" not as a linearly separable vector, but as a heavily entangled, non-linear subset of "helpfulness"—rendering flat orthogonal projection insufficient.