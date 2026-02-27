# Simulating "Vipassana": Regulating Precision Weights to Mitigate Rigidity and Hallucination in Large Language Models

**Title Page**

**Title**: Simulating "Vipassana": Regulating Precision Weights to Mitigate Rigidity and Hallucination in Large Language Models

**Author Names and Affiliations**:

1.  **Botao Wang**
    *   School of Mathematics and Statistics, South-Central Minzu University, Wuhan 430074, China
    *   E-mail: 2024110589@mail.scuec.edu.cn
    *   ORCID: 0009-0005-6651-7728

2.  **Xinnian Wang** (Corresponding Author)
    *   School of Computer Science (School of Artificial Intelligence), South-Central Minzu University, Wuhan 430074, China
    *   E-mail: 3089908@mail.scuec.edu.cn
    *   ORCID: 0009-0008-8756-0102

**Corresponding Author**:
Xinnian Wang
School of Computer Science (School of Artificial Intelligence), South-Central Minzu University
No. 182 Minzu Avenue, Hongshan District, Wuhan, Hubei 430074, P.R. China
E-mail: 3089908@mail.scuec.edu.cn

---

**Abstract**

Current Large Language Models (LLMs) suffer from cognitive rigidities akin to human cognitive biases: "hallucinations" (over-reliance on internal priors despite contradictory evidence) and "perseveration" (inability to adapt to rule shifts). Drawing on the Free Energy Principle (FEP) and the Buddhist philosophy of "Vipassana" (insight meditation), we propose that these failures stem from the maladaptive assignment of "precision weights" to prior beliefs. In biological brains, precision weighting modulates the balance between top-down predictions and bottom-up sensory errors. We formalize this mechanism as the **Predictive Coding Contextual Framework (PCCF)**, which introduces an online metacognitive controller. This controller monitors prediction error variance (uncertainty) and dynamically regulates the "learning rate" (effective plasticity) of the model's internal state—simulating the "Vipassana" process of suspending judgment and observing raw sensory data. We validate this framework on controlled concept drift and rule-shift tasks. Results show that precision-regulated adaptation significantly outperforms static baselines in recovering from regime shifts, effectively reducing "cognitive rigidity." This work bridges the gap between ancient contemplative phenomenology and modern computational psychiatry, offering a novel, biologically grounded path to robust AI.

**Keywords**: Predictive Coding, Free Energy Principle, Large Language Models, Precision Weighting, Vipassana, Metacognition, AI Alignment.

---

## 1. Introduction

### 1.1 The Phenomenology of AI Hallucinations
Large Language Models (LLMs) have exhibited remarkable capabilities, yet they remain prone to "hallucinations"—generating plausible but factually incorrect content (Ji et al., 2023)—and "perseveration," where they fail to adapt to abrupt changes in context or rules. In cognitive science terms, these are not merely engineering bugs but functional failures in **reality monitoring**.

We propose a bold cross-disciplinary hypothesis: the cognitive failures of LLMs are isomorphic to human cognitive pathologies described in both computational psychiatry and ancient contemplative traditions (e.g., Yogacara Buddhism). Specifically, hallucinations arise from **overly strong priors** (top-down predictions) that overwhelm sensory evidence (bottom-up input), a phenomenon known in the Free Energy Principle (FEP) as maladaptive **precision weighting** (Clark, 2013; Friston, 2010).

### 1.2 The "Attachment" to Priors
In Buddhist phenomenology, suffering (*Dukkha*) and delusion (*Avidya*) stem from *Upadana* (grasping or attachment)—the rigid adherence to internal mental constructions despite the impermanence (*Anicca*) of external reality. In the language of predictive coding, this "attachment" is mathematically equivalent to assigning excessively high **precision** (inverse variance) to prior beliefs (Hoge et al., 2013).

*   **Hallucination** corresponds to "projecting" an internal narrative onto the world, ignoring the mismatch with reality.
*   **Perseveration** corresponds to "clinging" to an outdated rule, refusing to update the internal model when the environment shifts.

### 1.3 Simulating "Vipassana" as Engineering Solution
"Vipassana" (insight meditation) is a practice of **dereifying** thoughts—observing them as transient mental events rather than absolute truths. Neuroscientifically, this process has been linked to the relaxation of high-level priors, allowing bottom-up prediction errors to propagate freely and update the internal model. This is described by the REBUS model (Relaxed Beliefs Under Psychedelics) (Carhart-Harris & Friston, 2019), which suggests that reducing the precision of high-level priors increases the entropy of the brain's state space, facilitating the escape from local minima (rigid beliefs).

We translate this insight into an engineering framework: **PCCF (Predictive Coding Contextual Framework)**. PCCF acts as a metacognitive layer that monitors the model's "surprise" (prediction error). When surprise is high and reliable (low variance), PCCF dynamically increases the system's plasticity (learning rate), effectively "relaxing" the attachment to old priors and allowing rapid adaptation. This mechanism provides a principled, biologically inspired solution to the stability-plasticity dilemma in AI.

---

## 2. Theoretical Framework: From Philosophy to Math

### 2.1 The Three Natures (Trisvabhava) of AI Perception
We map the Yogacara concept of "Three Natures" (Trisvabhava) to the components of a predictive AI system, as summarized in **Table 1** and visualized in **Figure 1**.

![Figure 1: Conceptual mapping between Yogacara phenomenology and the PCCF architecture. The left side represents the cognitive process of insight (Vipassana), while the right side shows the corresponding computational components in the PCCF framework.](pccf_conceptual_framework.png)

**Table 1: Mapping between Yogacara Phenomenology and PCCF Computational Components**

| Yogacara Concept | Phenomenological Meaning | PCCF Component | Computational Role |
| :--- | :--- | :--- | :--- |
| **Paratantra** | Dependent Arising | Sensory Input $o_t$ | Bottom-up prediction error source |
| **Parikalpita** | Mental Construction | Prior Beliefs $g(\mu_t)$ | Top-down predictive priors (Attention) |
| **Parinispanna** | Ultimate Reality | Metacognitive Gain $K_t$ | Precision-weighted error normalization |
| **Upadana** | Attachment/Clinging | High Precision $\Pi_\mu$ | Rigid priors causing hallucination |
| **Vipassana** | Insight/Observation | Precision Regulation | Dynamic adjustment of update gain |

### 2.2 Mathematical Formalization: The Free Energy Principle in Context
The core of our framework is based on the **Free Energy Principle (FEP)** (Friston, 2010), which posits that any self-organizing system must minimize its variational free energy $F$ to maintain its structural integrity against environmental entropy. In the context of an AI agent, $F$ can be approximated as:

$$ F \approx \underbrace{\frac{1}{2} \Pi_s (o - g(\mu))^2}_{\text{Accuracy (Surprise)}} + \underbrace{\frac{1}{2} \Pi_\mu (\mu - \mu_{prior})^2}_{\text{Complexity}} $$

Where:
*   $g(\mu)$ is the model's generative function (prediction).
*   $\Pi_s$ is the **Sensory Precision** (inverse variance of the sensory noise).
*   $\Pi_\mu$ is the **Prior Precision** (inverse variance of the belief prior).

The optimal belief update $\dot{\mu}$ is obtained by performing gradient descent on $F$:
$$ \dot{\mu} = -\frac{\partial F}{\partial \mu} = \Pi_s (o - g(\mu)) \frac{\partial g}{\partial \mu} - \Pi_\mu (\mu - \mu_{prior}) $$

This update rule highlights that the learning signal is not just the error $(o - g(\mu))$, but the error weighted by sensory precision $\Pi_s$. In standard backpropagation, this precision is implicitly assumed to be constant (learning rate). PCCF makes this explicit and dynamic.

### 2.3 Biological Plausibility: The Locus Coeruleus-Norepinephrine System
The proposed mechanism mimics the biological function of the **Locus Coeruleus-Norepinephrine (LC-NE)** system. In the mammalian brain, the LC releases norepinephrine (NE) in response to "unexpected uncertainty" (Yu & Dayan, 2005). Phasic NE bursts serve as a "network reset" signal, increasing the gain of neural populations and facilitating rapid re-configuration of functional networks (Parr & Friston, 2018). Similarly, PCCF uses the precision-weighted surprise as a proxy for phasic NE, modulating the "gain" (learning rate) of the artificial neural network to enable rapid adaptation to new contexts.

### 2.4 The Metacognitive Update Algorithm
PCCF simulates the "insight" process by dynamically modulating the effective gain $K_t$ based on the **normalized surprise**. Let $e_t = o_t - g(\mu_t)$ be the instantaneous prediction error. We maintain an online estimate of error variance $\hat{\sigma}_t^2$ via exponential smoothing:
$$ \hat{\sigma}_{t+1}^2 = (1-\rho)\hat{\sigma}_t^2 + \rho e_t^2 $$
The precision-weighted surprise $\tilde{\mathcal{S}}_t$ is defined as the squared error normalized by the estimated variance:
$$ \tilde{\mathcal{S}}_t = \frac{e_t^2}{\hat{\sigma}_t^2 + \varepsilon} $$
The metacognitive controller then adjusts the update gain $K_t$ (plasticity):
$$ K_t = \alpha_{\text{base}} \cdot \left( 1 + \beta \cdot \tanh(\tilde{\mathcal{S}}_t) \right) $$

This formulation ensures that when a "reliable" surprise occurs (i.e., the error is large compared to the historical noise floor), the model's plasticity increases—mimicking the **relaxation of priors** in meditative states. The $\tanh$ function acts as a soft gating mechanism, bounding the gain to prevent instability, analogous to the physiological limits of neurotransmitter release.

---

## 3. Experimental Validation: Breaking Cognitive Inertia

We evaluate the PCCF agent's ability to resolve the **Stability-Plasticity Dilemma**—the trade-off between retaining stable knowledge and adapting to novel information.

### 3.1 Experiment I: Overcoming "Perseveration" under Concept Drift
In cognitive psychology, *perseveration* is the pathological persistence of a response despite its no longer being appropriate. We simulate this using a scalar mean-shift task.

*   **Setup**: A univariate process shifts its mean from $\mu=10$ to $\mu=20$ at $t=50$. The signal is corrupted by Gaussian noise ($\sigma=0.1$).
*   **Cognitive Baselines**: We compare PCCF against "rigid" agents (Moving Average, EMA) and "fixed" Bayesian filters (Kalman).
*   **Implementation Details**: The PCCF agent uses a base learning rate $\alpha=0.01$ and gain multiplier $\beta=4.0$. The variance estimator uses a smoothing factor $\rho=0.1$.
*   **Recovery Analysis**: As shown in **Table 2**, the PCCF agent exhibits significantly lower **Post-drift MSE** ($3.866 \pm 0.500$) compared to the Moving Average baseline ($8.730 \pm 0.923$).
*   **Phenomenological Interpretation**: The PCCF agent detects the "prediction error shock" at the drift point. By boosting $K_t$, it effectively "lets go" of the stale prior ($\mu=10$), mimicking the Zen concept of *Shoshin* (Beginner's Mind)—approaching the new data without the bias of the past.

**Table 2: Performance under scalar concept drift (mean ± std over 30 seeds)**

| Model | Overall MSE | Post-drift MSE | Recovery Steps |
| :--- | :--- | :--- | :--- |
| **PCCF (Metacognitive)** | 2.593 ± 0.322 | **3.866 ± 0.500** | **3.83 ± 1.56** |
| MovingAvg (Rigid) | 5.939 ± 0.520 | 8.730 ± 0.923 | 8.77 ± 1.41 |
| Kalman (Fixed Prior) | 7.559 ± 0.468 | 12.006 ± 0.872 | 17.90 ± 3.34 |

### 3.2 Experiment II: Mitigating "Mental Fixation" in Symbolic Sequences
We further test the framework on a toy next-token prediction task where a symbolic rule change ($A \to B$ becomes $A \to C$) is introduced. This setup mimics the "Wisconsin Card Sorting Test," a standard neuropsychological assessment of cognitive flexibility.

*   **Setup**: A 3-token vocabulary ($V=\{0,1,2\}$) next-token prediction task. The sequence length is 16 tokens. A rule shift occurs at $t=200$.
*   **Model Architecture**: A "Tiny Transformer" with embedding dimension $d_{model}=32$, 2 attention heads, and 1 layer.
*   **Baselines**:
    *   **FixedLR**: Standard Adam optimizer with fixed learning rate.
    *   **LossLR**: Naive adaptive schedule scaling with raw loss (not precision-weighted).
    *   **PH_Reset**: Page-Hinkley drift detector that resets the model upon detection (extreme plasticity).
*   **Result**: PCCF significantly outperforms naive loss-driven schedules (LossLR). While LossLR reacts to the total loss, PCCF reacts to the **variance-normalized surprise**, allowing it to distinguish between "routine noise" and "regime change."
*   **Statistical Significance**: Paired permutation tests ($N=20,000$ iterations) confirm that PCCF's reduction in Negative Log-Likelihood (NLL) is statistically significant ($p < 0.001$).

---

## 4. Discussion: Toward "Enlightened" AI

### 4.1 AI Metacognition as the Path to Alignment
Current AI alignment focuses on RLHF (reward shaping). Our work suggests a complementary path: **internal structural alignment** via metacognition. An aligned AI should not just "know" facts, but "know how much it knows" (precision). It should be capable of "doubt" (lowering prior precision) when evidence contradicts its training.

### 4.2 The "Psychedelic" AI
The mechanism of PCCF—temporarily boosting plasticity under high surprise—is functionally analogous to the effect of psychedelics on the human brain (REBUS model) (Carhart-Harris & Friston, 2019). By "shaking up" the frozen priors (annealing), the system escapes local minima of delusion. This suggests that future robust AI might need periodic "controlled destabilization" phases to maintain adaptability.

### 4.3 Limitations and Future Work
Our current experiments are limited to synthetic tasks and "toy" language models. While these simplified environments allow for precise control and phenomenological analysis, scaling PCCF to Large Language Models (LLMs) with billions of parameters presents challenges. Specifically, estimating the error variance in high-dimensional semantic spaces is non-trivial. Future work will explore efficient approximations of precision weighting for Transformer attention heads, potentially linking our "metacognitive gain" to the "temperature" parameter in softmax attention.

---

## 5. Conclusion
We presented PCCF not merely as an algorithm, but as a computational model of **insight**. By operationalizing the Buddhist concept of "letting go of attachment" (precision regulation), we enable AI systems to navigate non-stationary worlds with greater truthfulness and flexibility. This synthesis of Eastern phenomenology and Western computational neuroscience offers a fertile ground for the next generation of cognitive systems.

---

## References

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901.

Carhart-Harris, R. L., & Friston, K. J. (2019). REBUS and the anarchic brain: toward a unified model of the brain action of psychedelics. *Pharmacological Reviews*, 71(3), 316-344.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181-204.

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

Hoge, E. A., Bui, E., Marques, L., Metcalf, C. A., Morris, L. K., Robinaugh, D. J., ... & Simon, N. M. (2013). Randomized controlled trial of mindfulness meditation for generalized anxiety disorder: effects on anxiety and stress reactivity. *Journal of Clinical Psychiatry*, 74(8), 786-792.

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12), 1-38.

Millidge, B., Tschantz, A., & Buckley, C. L. (2021). Predictive coding approximates backprop along arbitrary computation graphs. *Neural Computation*, 33(6), 1533-1569.

Parr, T., & Friston, K. J. (2018). The anatomy of inference: generative models and brain structure. *Frontiers in Computational Neuroscience*, 12, 90.

Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79-87.

Seth, A. K. (2021). *Being You: A New Science of Consciousness*. Faber & Faber.

Tagliazucchi, E., Carhart-Harris, R., Leech, R., Nutt, D., & Chialvo, D. R. (2014). Enhanced repertoire of brain dynamical states during the psychedelic experience. *Human Brain Mapping*, 35(11), 5442-5456.

Vasubandhu. (1999). *Trimshika-vijnaptimatra* (Thirty Verses on Consciousness-Only). Trans. F. Cook, *Three Texts on Consciousness Only*, BDK America.

Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*, 46(4), 681-692.

### Declarations

**Funding**: No external funding was received for this work.
**Conflicts of Interest**: The authors declare no conflicts of interest.
**Code Availability**: The source code and reproduction scripts are available at https://github.com/caridle/PCCF.
