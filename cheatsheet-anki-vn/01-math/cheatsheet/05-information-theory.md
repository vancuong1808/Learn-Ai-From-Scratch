# Cheatsheet — Information Theory (Lý thuyết thông tin) cho AI

> Phase 1 · Sheet 05 · Phase Toán
> Entropy → Cross-entropy → KL → JS → Mutual Information — nền của loss function trong ML/DL/LLM. Tiếng Việt + giữ thuật ngữ TA gốc.

---

## 0. Roadmap sheet này

1. Entropy H(P) — độ bất định của phân phối
2. Cross-entropy H(P, Q) — bits cần để encode P dùng code của Q
3. KL divergence D(P||Q) — extra bits so với optimal (H(P) − cross-entropy breakdown)
4. Jensen-Shannon divergence — symmetric version
5. Mutual Information I(X;Y) — đo shared information giữa 2 RV
6. Perplexity — metric cho language model
7. Information bottleneck — principle cho representation learning
8. Loss function bridge (MLE ↔ CE, BCE ↔ Bernoulli, LLM loss ↔ categorical CE)

---

## 1. Entropy H(P)

### Định nghĩa (Shannon, 1948)
$$H(P) = -\sum_x P(x) \log P(x) \quad \text{(discrete)}$$
$$H(P) = -\int p(x) \log p(x) \, dx \quad \text{(continuous — differential entropy)}$$

- **Đơn vị**: bits (log₂), nats (logₑ), dits (log₁₀). ML thường dùng nats (logₑ = ln).
- **H(P)** đo average surprise (độ không chắc chắn) khi sample từ P.
- H(P) ≥ 0 cho mọi phân phối hợp lệ.

### Tính chất
| P | H(P) |
|---|------|
| Deterministic (P(1)=1, rest 0) | **0** (zero surprise — lucertain) |
| Uniform qua K lớp | **log₂ K** bits — maximum |
| P = (0.5, 0.5) Bernoulli | 1 bit (= 0.69 nat) |
| Càng spread → entropy càng cao |

### Code numpy
```python
import numpy as np

def entropy(p, base=np.e):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]                         # bỏ zeros (0·log(0) = 0)
    return -np.sum(p * np.log(p)) / np.log(base)

# Bernoulli p=0.5 → 1 bit
print(entropy(np.array([0.5, 0.5]), base=2))   # 1.0

# Uniform 4 classes → 2 bits
print(entropy(np.array([0.25, 0.25, 0.25, 0.25]), base=2))  # 2.0
```

### Hàm của scipy
```python
from scipy.stats import entropy
entropy(p, q=None, base=2)   # entropy của p nếu chỉ có p, KL nếu có q
```

---

## 2. Cross-Entropy H(P, Q)

### Định nghĩa
$$H(P, Q) = -\sum_x P(x) \log Q(x)$$

- Dùng Q (model) để encode distribution P (true data). Bits needed = có thể > H(P) nếu Q ≠ P.
- **Cốt lõi của supervised learning loss**: minimize CE = làm Q tiệm cận P.

### CE = Entropy + KL
$$H(P, Q) = H(P) + D_{KL}(P \parallel Q)$$

- H(P) cố định (data distribution), minimize CE = minimize KL → Q → P.
- Trong optimization: H(P) không phụ thuộc θ → gradient = gradient của KL.

### Ứng suốt (học thuộc)
| Loss trong ML/DL | Công thức | Là CE của? |
|-----------------|-----------|-----------|
| **MSE (LinReg)** | (1/n) Σ (y−ŷ)² | MLE Gaussian = CE (Gaussian NLL) |
| **BCE (binary)** | −[y·log(p)+(1−y)·log(1−p)] | Bernoulli NLL = CE with Bernoulli |
| **Categorical CE** | −Σ yₖ·log pₖ | Multinoulli NLL = CE with Categorical |
| **LLM CE** | −log p(token_id) | Categorical CE over vocab K tokens |

### Code torch
```python
F.binary_cross_entropy(sigmoid_output, target)
F.cross_entropy(logits, target_index)         # internal: softmax + CE
```

---

## 3. KL Divergence D(P || Q)

### Định nghĩa
$$D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

- **KL ≥ 0** (Gibbs inequality), = 0 iff P=Q.
- **Asymmetric**: D(P||Q) ≠ D(Q||P)
- **Không phải metric**: không satisfy triangle inequality, không symmetric
- Đơn vị: bits (log₂) hoặc nats (logₑ)

### Forward vs Reverse KL
| Direction | Hành vi | Dùng cho |
|-----------|---------|----------|
| **Forward** D(P&Vert;Q) | Mode-seeking, penalize khi P high ∧ Q low → Q focus mode của P | MLE (train NN), cross-entropy|
| **Reverse** D(Q&Vert;P) | Mean-seeking, penalize khi Q high ∧ P low → Q spread để cover P | VAE (regularize q(z|x) về prior N(0,I)) |

### Code scipy
```python
from scipy.stats import entropy
kl = entropy(p, q, base=2)  # D(P||Q) in bits
# = Σ p·log(p/q)
```

### Ứng dụng
- **Loss trong classification** (cross-entropy = H(P) + KL)
- **Knowledge distillation**: minimize KL(student softmax || teacher softmax)
- **VAE regularization**: min KL(encoder || prior N(0,I))
- **RL policy optimization** (TRPO / PPO: constraint KL for stable update)

---

## 4. Jensen-Shannon (JS) Divergence — symmetric

### Định nghĩa
$$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$
với M = (P+Q)/2 (mixture distribution)

### Tính chất
- **Symmetric**: JSD(P||Q) = JSD(Q||P)
- **Bounded**: 0 ≤ JSD ≤ log 2 (≈ 0.69 nat) hoặc 1 bit (log₂)
- **Metric**: √JSD là metric (Endres & Schindelin 2003)

### Ứng dụng
- **GAN objective (original)**: minimax JSD giữa real và generated distribution
- **Fair evaluation**: khi KL điệp (P mode vs Q dif), JS cản symmetry

### Code numpy
```python
def jsd(p, q, base=np.e):
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base) + 0.5 * entropy(q, m, base)
```

---

## 5. Mutual Information (MI) — đo shared information

### Định nghĩa
$$I(X; Y) = \sum_{x,y} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$
$$= D_{KL}(P_{XY} \parallel P_X \otimes P_Y)$$

- **I(X;Y) = entropy reduction**: I(X;Y) = H(X) − H(X|Y) = reduction trong uncertainty của X khi biết Y.
- **I(X;Y) ≥ 0**, symmetric.

### Tương quan với correlation
- I(X;Y) = 0 iff X ⟂ Y (independent)
- Correlation chỉ đo linear, MI đo cả nonlinear relationship
- MI thường dùng trong feature selection (SelectKBest với mutual_info_classif/regression)

### Code
```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

mi = mutual_info_classif(X, y)[feature_idx]
```

### Ứng dụng ML
- **Feature selection**: rank features theo MI với target
- **Information bottleneck**: compress input X → representation Z, giữ MI(Z;Y) lớn (predictive), reduce MI(Z;X) (compression)
- **Co-training**: MI between views trong multi-view learning

---

## 6. Perplexity — metric cho language model

### Định nghĩa
$$\text{PPL} = 2^{H(P, \text{model})} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log P_{\text{model}}(w_t | w_{<t})\right)$$

- **PPL = geometric average của inverse probability** của mỗi token.
- PPL thấp → model tốt: p(token) gần 1 → log ≈ 0 → PPL thấp.
- PPL = K (vocab size) cho uniform model (random guess)

### Ví dụ
- Model biết chắc token tiếp theo → P(token) ≈ 1 → −log ≈ 0 → PPL ≈ 1 (perfect)
- Uniform random qua 50k vocab → PPL ≈ 50000
- Modern LLM: PPL thường 5-20 trên eval data (tùy model + dataset)

### Code cho PPL của HF model
```python
import torch
output = model(input_ids, labels=input_ids)
loss = output.loss                 # average CE loss per token
ppl = torch.exp(loss).item()       # PPL via exp(CE)
# HF pipeline: output=model(…labels=…) tự trả loss CE
```

---

## 7. Information Bottleneck (Tishby et al.)

### Principle
$$J = \min_{p(z|x)} I(X; Z) - \beta \cdot I(Z; Y)$$

- **Compress**: minimize I(X;Z) — reduce info từ X giữ trong Z
- **Predictive**: maximize I(Z;Y) — giữ info cần cho target Y
- **β** = tradeoff weight: β nhỏ → nén mạnh (mất prediction); β lớn → giữ nhiều info
- **Deep NN implicitly như information bottleneck**: representation học automatic học compressed representation predictive of target

### Ứng thật
- **Feature compression**: biểu diễn hidden layer of transformer = information bottleneck
- **Latent space design**: VAE, diffusion all architect compression+reconstruction via entropy/ MI

---

## 8. Loss Bridge (MLE ↔ Information Theory ↔ ML/DL)

### Bảng tổng kết

| ML Concept | Information Theory View | Loss Function |
|------------|------------------------|--------------|
| Gaussian likelihood → MLE | min CE via Gaussian NLL→ MSE | MSE |
| Bernoulli likelihood → MLE | min CE via Bernoulli NLL→ BCE | BCE |
| Categorical likelihood → MLE | min CE via Categorical CE→ cross-entropy với logit| CE multi-class |
| Categorical (LLM) | min CE seq-to-seq→ language modeling loss | CE averaged over tokens |
| VAE regularization | min KL(q(z|x) || prior N(0,I))→ reverse KL | KL term in ELBO |
| GAN minimax | min JSD(real || fake) | JSD (implicitly via adversary) |
| Knowledge distillation | min KL(student softmax || teacher softmax) + CE(student logit, hard target) | KL + CE |
| TRPO/PPO (RL) | constraint KL(π_new || π_old) ≤ δ | surrogate loss + hard trust region |

### Code numpy cho CE / KL / Entropy nền
```python
import numpy as np

def entropy(p):
    p = np.asarray(p); p = p[p > 0]
    return -sum(p * np.log(p))                   # natural log (nats)

def cross_entropy(p, q):
    p = np.asarray(p); q = np.asarray(q)
    return -sum(p * np.log(q + 1e-15))           # epsilon prevent log(0)

def kl_div(p, q):
    return cross_entropy(p, q) - entropy(p)      # CE − H
```

---

## 9. Pitfalls

| Pitfall | Fix |
|---------|-----|
| log(0) = −inf → CE NaN | add epsilon 1e-8 vào q (log(q+eps)) hoặc dùng label smoothing |
| KL asymmetric → behavior dif | choose direction based on goal: forward for MLE, reverse for VAE regularization |
| LLM PPL huge (>500) chỉ do tokenization mismatch | check tokenizer pad/bos/eos flow |
| Mutual Info sklearn requires integer/discrete feature | discretize continuous feature bằng KBinsDiscretizer |
| Cook the CE loss: forgetting H(P) constant không liên quan | Chỉ biết minimize CE tương đương minimize KL, H(P) cố định khác biệt impact 0 đến gradient |
| JS bounded log2~0.69 gràn xn trong GAN đọc tốt | Too small derivative when real ≈ fake → chilling gradient issue; Wasserstein GAN replace JS với Wasserstein distance |

---

## 10. Liên hệ chủ đề khác

- **Calculus sheet 02**: gradient của CE dẫn đến dL/dz = softmax−onehot (công thức gọn) — derivation trong đó
- **Probability sheet 03**: CE & KL đã có connection to MLE; entropies là foundation
- **Deep learning (Phase 3)**: every loss function formalized qua CE/KL here
- **GenAI LLM (Phase 4)**: PPL evaluation, temperature sampling từ softmax distribution → entropy của output controllable

---

**Sheet 05 ok. ℹ ← Now CSV anki files.**