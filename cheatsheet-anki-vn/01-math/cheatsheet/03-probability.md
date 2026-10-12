# Cheatsheet — Probability (Xác suất) cho AI

> Phase 1 · Sheet 03 · Phase Toán
> Che-sheet này tổng hợp các khái niệm xác suất nền và thông dụng trong AI. Tiếng Việt + giữ thuật ngữ TA gốc.

---

## 0. Roadmap sheet này

1. Probability khái niệm cơ bản (P(A), ∅, sure event)
2. Conditional probability & Bayes formula
3. Joint / Marginal / Chain rule of probability
4. Independence vs Conditional Independence
5. Random variable (discrete vs continuous)
6. PMF / PDF / CDF
7. Expectation / Variance / Std
8. Covariance & Correlation
9. Common distributions (Bernoulli, Binomial, Multinomial, Categorical, Normal, Multivariate Normal, Exponential, Poisson, Beta, Gamma, Laplace, Uniform)
10. Sampling & Monte Carlo
11. Law of Large Numbers / Central Limit Theorem
12. Maximum Likelihood Estimation (MLE)
13. MAP estimation (Bayesian)
14. KL divergence & cross-entropy (review info theory)
15. Probability trong LLM / Naive Bayes / VAE

---

## 1. Probability khái niệm cơ bản

### Định nghĩa nghiệm lý (frequentist)
- \(P(A)\) = tần suất xuất hiện event A khi lặp lại rất nhiều
- \(P(A) \in [0, 1]\)
- \(P(\emptyset) = 0\), \(P(\Omega) = 1\)

### Tiên đề Kolmogorov
1. \(P(A) \geq 0\)
2. \(P(\Omega) = 1\)
3. Nếu \(A_i\) rời rạc đôi một: \(P\left(\bigcup A_i\right) = \sum P(A_i)\)

### Operations
| Operation | Công thức |
|-----------|-----------|
| Phần bù | \(P(A^c) = 1 - P(A)\) |
| Union | \(P(A \cup B) = P(A) + P(B) - P(A \cap B)\) |
| Intersection (independent) | \(P(A \cap B) = P(A) P(B)\) |
| Difference | \(P(A \setminus B) = P(A) - P(A \cap B)\) |

---

## 2. Conditional Probability & Bayes

### Conditional probability
$$P(A | B) = \frac{P(A \cap B)}{P(B)} \quad (P(B) > 0)$$

Trực giác: trong trường hợp B đã xảy ra, tần suất A đồng thời xuất hiện.

### Product rule
$$P(A \cap B) = P(A | B) P(B) = P(B | A) P(A)$$

### Law of Total Probability
Nếu \(\{B_i\}\) là partition của Ω:
$$P(A) = \sum_i P(A | B_i) P(B_i)$$

### Bayes' theorem (CỰC KỲ quan trọng)
$$P(H | E) = \frac{P(E | H) P(H)}{P(E)} = \frac{P(E | H) P(H)}{\sum_i P(E | H_i) P(H_i)}$$

### Ý nghĩa từng thành phần (học thuộc tên)
| Ký hiệu | Tên TA | Tên TV | Ý nghĩa |
|---------|--------|--------|---------|
| \(P(H)\) | Prior | Tiền định | Niềm tin ban đầu về H trước khi thấy E |
| \(P(E \| H)\) | Likelihood | Khả năng | Xác suất quan sát E nếu H đúng |
| \(P(E)\) | Evidence | Bằng chứng | Tổng xác suất quan sát E (marginal) |
| \(P(H \| E)\) | Posterior | Hậu nghiệm | Niềm tin cập nhật về H sau khi thấy E |

### Bayes trong spam classification (ví dụ kinh điển)
- \(H = \text{spam}\), \(E = \text{xuất hiện từ "FREE"}\)
- \(P(H) = 0.4\) (40% mail spam)
- \(P(E | H) = 0.7\) (70% mail spam có "FREE")
- \(P(E | H^c) = 0.05\) (5% mail legit có "FREE")
- \(P(H | E) = \frac{0.7 \times 0.4}{0.7 \times 0.4 + 0.05 \times 0.6} = \frac{0.28}{0.31} \approx 0.903\)

Nhận mail có "FREE" → 90% khả năng spam (posterior tăng từ prior 40%).

### Pitfall
- **Base rate fallacy**: bỏ qua prior. VD test bệnh 99% chính xác, bệnh hiếm 1%。1 người dương tính → chỉ ~50% thực mắc, vì prior quá thấp → posterior thấp.
- **Confused conditional**: nhầm \(P(A|B)\) với \(P(B|A)\). Bayes để chuyển 2 quantity này.

---

## 3. Joint / Marginal / Chain Rule of Probability

### Joint distribution
\(P(X = x, Y = y)\) — xác suất đồng thời \(X=x\) và \(Y=y\).

### Marginal
$$P(X = x) = \sum_y P(X = x, Y = y) \quad \text{(discrete)}$$
$$p_X(x) = \int p_{X,Y}(x, y) \, dy \quad \text{(continuous)}$$

### Chain rule (general,ঔ NOT chỉ cho independent)
$$P(X_1, \ldots, X_n) = P(X_1) P(X_2 | X_1) P(X_3 | X_1, X_2) \cdots P(X_n | X_1, \ldots, X_{n-1})$$

VD: \(P(A, B, C) = P(A) P(B | A) P(C | A, B)\)

### Ứng dụng
- LLM: \(P(w_1 \ldots w_n) = \prod_i P(w_i | w_1, \ldots, w_{i-1})\) — auto-regressive generation
- Naive Bayes: \(P(y, x_1, \ldots, x_n) = P(y) \prod P(x_i | y)\) (giả định independent cho gọn)

---

## 4. Independence vs Conditional Independence

### Independence
\(X \perp Y \iff P(X, Y) = P(X) P(Y)\)

### Conditional Independence
\(X \perp Y | Z \iff P(X, Y | Z) = P(X | Z) P(Y | Z)\)

VD: discovery bệnh (X) và tin tức (Y) độc lập cho trước tình trạng cơ thể (Z), nhưng không độc lập nếu không biết Z.

### Pitfall
- \(X \perp Y\) KHÔNG kéo \(X \perp Y | Z\). VD: Simpson's paradox.
- \(X \perp Y | Z\) KHÔNG kéo \(X \perp Y\). VD: hai ngọn đồng hồ hiệu chỉnh theo cùng nguồn thời gian.

### Markov assumption (RLLLM)
\(P(S_{t+1} | S_t, A_t, S_{t-1}, \ldots) = P(S_{t+1} | S_t, A_t)\) — RL, Markov decision process foundation.

---

## 5. Random Variable (Discrete vs Continuous)

### Discrete
- Tập giá trị hữu hạn hoặc đếm được (vd: {0,1,2,...})
- Mô tả bằng **PMF** (probability mass function): \(P(X=x)\)

### Continuous
- Tập giá trị trong khoảng liên tục (vd: [0,1])
- Mô tả bằng **PDF** (probability density function): \(f_X(x)\) với:
  - \(f_X(x) \geq 0\)
  - \(\int_{-\infty}^{+\infty} f_X(x) \, dx = 1\)
  - \(P(a \leq X \leq b) = \int_a^b f_X(x) \, dx\)
- **Lưu ý**: PDF có thể > 1 (vì là density, không phải probability)

---

## 6. PMF / PDF / CDF

### CDF (cumulative distribution function)
$$F_X(x) = P(X \leq x)$$

- Monoton không giảm, \(\lim_{x \to -\infty} F = 0\), \(\lim_{x \to +\infty} F = 1\)
- Discrete: \(F_X(x) = \sum_{y \leq x} P(X = y)\)
- Continuous: \(F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt\)

### Liên hệ PDF ↔ CDF
$$f_X(x) = \frac{d F_X(x)}{dx}, \quad F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$$

### SciPy
```python
from scipy import stats

# Discrete: binomial
binom = stats.binom(n=10, p=0.5)
binom.pmf(5)               # P(X=5)
binom.cdf(5)               # P(X<=5)
binom.rvs(size=100)        # sample 100

# Continuous: normal
norm = stats.norm(loc=0, scale=1)
norm.pdf(0)                # density tại 0 (= 0.399)
norm.cdf(0)                # = 0.5
norm.ppf(0.975)            # inverse CDF = 1.96 (z-score 97.5%)
```

---

## 7. Expectation / Variance / Std

### Expectation (kỳ vọng, mean)
$$E[X] = \sum x \, P(X=x) \quad (\text{discrete})$$
$$E[X] = \int x \, f_X(x) \, dx \quad (\text{continuous})$$

### Variance (phương sai)
$$\text{Var}(X) = E\left[(X - E[X])^2\right] = E[X^2] - (E[X])^2$$

### Standard deviation
$$\sigma_X = \sqrt{\text{Var}(X)}$$

### Properties
- \(E[aX + b] = a E[X] + b\) (linearity)
- \(E[X + Y] = E[X] + E[Y]\) (luôn đúng, không cần independent)
- \(\text{Var}(aX + b) = a^2 \text{Var}(X)\)
- \(X, Y\) độc lập → \(\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)\)
- \(X, Y\) độc lập → \(E[XY] = E[X] E[Y]\). Ngược lại chưa chắc đúng.

### Code numpy
```python
import numpy as np

samples = np.random.normal(0, 1, size=100000)
print(samples.mean())    # ≈ 0
print(samples.var())     # ≈ 1
print(samples.std())     # ≈ 1
```

---

## 8. Covariance & Correlation

### Covariance
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]$$

- Dương: tăng cùng chiều
- Âm: tăng ngược chiều
- 0: KHÔNG phải độc lập (chỉ là không tương quan tuyến tính)

### Correlation (Pearson)
$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$$

- Vô-dimensional, dễ compare giữa các cặp
- \(|rho|=1\): quan hệ tuyến tính hoàn hảo
- \(rho=0\): không tương quan tuyến tính (vẫn có thể phi tuyến)

### Covariance matrix (cho vector RV)
$$\Sigma = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

Ma trận đối xứng nửa xác định dương. Đường chéo là variances.

### Code numpy
```python
import numpy as np

X = np.random.normal(0, 1, (1000, 3))
np.cov(X.T)               # 3x3 covariance matrix
np.corrcoef(X.T)          # 3x3 correlation matrix
```

### Pitfall
- "Correlation không phải causation": \(rho\) cao không chứng minh quan hệ nhân quả. vd: bán kem tăng v重视 bán áo tắm tăng → cùng do nhiệt độ tăng, không phải gây nhau.
- **Simpson's paradox**: correlation đảo chiều khi chia theo sub-group.

---

## 9. Common Distributions

### 9.1. Bernoulli — 1 thử nghiệm binary
- \(X \in \{0, 1\}\)
- PMF: \(P(X=1) = p\), \(P(X=0) = 1-p\)
- Mean = \(p\), Var = \(p(1-p)\)
- Dùng: logistic regression output (1 neuron sigmoid), binary classification

```python
np.random.binomial(1, 0.5, size=10)    # sample Bernoulli
```

### 9.2. Binomial — n Bernoulli independent
- \(X \in \{0, 1, \ldots, n\}\)
- PMF: \(P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}\)
- Mean = \(np\), Var = \(np(1-p)\)
- Dùng: đếm số thành công trong n thử nghiệm

```python
np.random.binomial(n=10, p=0.5, size=100)
```

### 9.3. Categorical — 1 thử nghiệm trên K lớp
- \(X \in \{1, \ldots, K\}\)
- PMF: \(P(X=k) = p_k\), \(\sum p_k = 1\)
- Dùng: single-label classification (output softmax)

### 9.4. Multinomial — n Categorical independent
- \(X = (X_1, \ldots, X_K)\) với \(\sum X_k = n\)
- PMF: \(P(\mathbf{X} = \mathbf{x}) = \frac{n!}{x_1! \cdots x_K!} \prod p_k^{x_k}\)
- Dùng: count tokens trong văn bản (TF), count class predictions
- Mean: \(E[X_k] = n p_k\), Var: \(n p_k (1 - p_k)\), Cov: \(-n p_k p_l\)

```python
np.random.multinomial(20, [0.2, 0.5, 0.3], size=10)
# 10 sample, mỗi sample là count vector 3 lớp, tổng = 20
```

### 9.5. Normal (Gaussian) — distribution phố biến nhất
- PDF: \(f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)\)
- Mean = \(\mu\), Var = \(\sigma^2\)
- Ký hiệu: \(\mathcal{N}(\mu, \sigma^2)\)

```python
np.random.normal(0, 1, size=100)
```

### Tại sao Normal quan trọng?
- **Central Limit Theorem**: tổng của nhiều biến độc lập → phân phối tiệm cận Normal
- **Noise**: thêm gaussian noise vào data augmentation
- **Weight init**: He init, Glorot init đều dựa trên Normal

### Standard normal
\(\mathcal{N}(0, 1)\) — \(Z = \frac{X - \mu}{\sigma}\) chuNEN hóa.

### 9.6. Multivariate Normal — cho vector
- PDF: \(f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)\)
- Mean vector \(\boldsymbol{\mu} \in \mathbb{R}^d\), covariance matrix \(\Sigma \in \mathbb{R}^{d \times d}\)
- Dùng: Gaussian Discriminant Analysis, diffusion, VAE prior

```python
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
np.random.multivariate_normal(mean, cov, size=100)
```

### 9.7. Exponential — thời gian giữa events Poisson
- PDF: \(f(x) = \lambda e^{-\lambda x}\) cho \(x \geq 0\)
- Mean = \(1/\lambda\), Var = \(1/\lambda^2\)
- Memoryless: \(P(X > t + s | X > t) = P(X > s)\)
- Dùng: modeling waiting time, dropout time, decay

### 9.8. Poisson — count events trong interval
- PMF: \(P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}\)
- Mean = Var = \(\lambda\)
- Dùng: count clicks trong tracking, count decays per second

### 9.9. Beta — phân phối cho xác suất
- \(X \in [0, 1]\)
- PDF: \(f(x) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}\)
- \(\alpha, \beta > 0\) là shape parameters
- Mean = \(\alpha / (\alpha + \beta)\)
- Dùng: prior cho Bernoulli (Bayesian inference), LH trong Thompson Sampling cho bandit

### 9.10. Gamma — tổng gia số exponential
- \(X > 0\), shape \(k\), scale \(\theta\)
- Mean = \(k\theta\), Var = \(k\theta^2\)
- Dùng: prior cho precision (\(\tau = 1/\sigma^2\)) trong Bayesian LinReg, VAE

### 9.11. Uniform — prior không thông tin
- \(X \in [a, b]\), đều
- PDF: \(f(x) = 1/(b-a) \cdot \mathbb{1}_{[a, b]}(x)\)
- Mean = \((a+b)/2\), Var = \((b-a)^2/12\)
- Dùng: random init, prior không thiên vị, weight dla quantization

### 9.12. Laplace — dân số nhọn, robust outlier
- PDF: \(f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)\)
- Robust hơn Normal đối với outlier
- Dùng: L1 regularization (= max likelihood with Laplace prior), GAN discriminator penalty

### 9.13. Bảng tổng kết

| Dist | Domain | Param | Mean | Var | Khi nào |
|------|--------|-------|------|-----|---------|
| Bernoulli | {0,1} | p | p | p(1-p) | binary classifier output |
| Binomial | {0..n} | n, p | np | np(1-p) | count successes |
| Categorical | {1..K} | p_1..p_K | — | — | softmax output |
| Multinomial | N^k count | n, p_1..p_K | np_k | np_k(1-p_k) | TF-IDF, token counts |
| Normal | ℝ | μ, σ² | μ | σ² | noise, init |
| MVN | ℝ^d | μ, Σ | μ | Σ | VAE, GDA |
| Exponential | [0,∞) | λ | 1/λ | 1/λ² | waiting time |
| Poisson | {0,1,...} | λ | λ | λ | count events |
| Beta | [0,1] | α, β | α/(α+β) | αβ/((α+β)²(α+β+1)) | prior cho Bernoulli |
| Gamma | [0,∞) | k, θ | kθ | kθ² | prior cho precision |
| Uniform | [a,b] | a, b | (a+b)/2 | (b-a)²/12 | random init |
| Laplace | ℝ | μ, b | μ | 2b² | L1 reg prior |

---

## 10. Sampling & Monte Carlo

### Uniform random
```python
np.random.seed(42)
np.random.rand(100)              # uniform [0,1)
np.random.uniform(0, 1, 100)
np.random.randint(0, 10, 100)    # int [0, 10)
```

### Standard sampling
```python
np.random.normal(0, 1, 100)
np.random.standard_normal(100)
np.random.binomial(1, 0.5, 100)
np.random.multinomial(20, [0.5, 0.5], 100)
```

### Reproducibility — random seed
```python
np.random.seed(42)               # global seed (legacy)
rng = np.random.default_rng(42)  # modern Generator
rng.normal(0, 1, 100)

# PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

### Monte Carlo estimate
Ước tính kỳ vọng \(E[f(X)]\) bằng:
$$E[f(X)] \approx \frac{1}{N} \sum_{i=1}^N f(X_i) \quad X_i \sim p$$

### Variance ~ \(1/N\) (chuẩn MC)
Sai số chuẩn của estimator MC: \(\sigma / \sqrt{N}\). Nghĩa là giảm 10x RMS cần tăng N 100x.

### Code
```python
import numpy as np

# Ước lượng π bằng MC
N = 1000000
x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
inside = (x**2 + y**2) <= 1
pi_est = 4 * inside.mean()
print(pi_est)   # ≈ 3.14...
```

### Importance sampling (khi p khó sample trực tiếp)
$$E_p[f(X)] = E_q\left[ f(X) \frac{p(X)}{q(X)} \right]$$
- Sample từ q (dễ), reweight bằng \(p/q\)
- Dùng trong off-policy RL, rare event simulation

### Pitfall
- **`np.random.seed` vs `default_rng`**: prefer `default_rng` (modern, thread-safe, entropy tốt hơn)
- **Seed global không nhân bản**: khi chạy multiprocessing, mỗi worker cần seed khác nhau (` rng.spawn(n)`)
- **Monte Carlo bias khi MC N nhỏ**: dùng N ≥ 10000 for trực giác

---

## 11. Law of Large Numbers / Central Limit Theorem

### Law of Large Numbers (LLN)
Nếu \(X_1, \ldots, X_n\) i.i.d. với mean \(\mu\):
$$\bar X_n = \frac{1}{n} \sum X_i \xrightarrow{n \to \infty} \mu$$

→ Trung bình mẫu tiệm cận mean tổng thể.

### Central Limit Theorem (CLT)
$$\sqrt{n} \frac{\bar X_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)$$

→ Tổng của **bất kỳ** biến phân phối nào (có mean/var hữu hạn), normalize, cũng tiệm cận Normal.

**Ý nghĩa**:
- Giải thích tại sao Gaussian xuất hiện khắp nơi (noise, init, mean)
- Đáy gốc của `confidence interval`, `t-test`, `z-test`

---

## 12. Maximum Likelihood Estimation (MLE)

### Định nghĩa
Cho data \(\{x_1, \ldots, x_n\}\) i.i.d. từ \(p(x | \theta)\), tìm \(\theta\) làm **likelihood** lớn nhất:
$$\hat\theta_{MLE} = \arg\max_\theta \prod_i p(x_i | \theta) = \arg\max_\theta \sum_i \log p(x_i | \theta)$$

→ Tương đương maximum log-likelihood (vì \(\log\) dương đồng biến).

### Ví dụ Gaussian
Data i.i.d. \(\sim \mathcal{N}(\mu, \sigma^2)\), MLE cho:
- \(\hat\mu = \bar x\) (trung bình mẫu)
- \(\hat\sigma^2 = \frac{1}{n} \sum (x_i - \bar x)^2\) (variance mẫu, biased chia n chứ không phải n-1)

### Ví dụ Bernoulli
\(X_i \sim \text{Bernoulli}(p)\): \(\hat p = \bar x\) (tỉ lệ positive).

### Linear regression MLE
Nếu noise \(\epsilon \sim \mathcal{N}(0, \sigma^2)\) thì MLE = least squares (minimize MSE).

### Code numpy (Gaussian MLE)
```python
import numpy as np

data = np.random.normal(3.0, 2.0, 1000)
mu_mle = data.mean()
sigma2_mle = ((data - mu_mle) ** 2).mean()  # chia n, biased
print(mu_mle, sigma2_mle)
```

### Pitfall
- MLE có thể overfit khi data ít (do không có prior)
- Variance MLE chia n → biased (để unbiased phải chia n-1); trong ML không quan trọng vì n lớn
- MLE cho mixture model (GMM) cần EM algorithm để giải

---

## 13. MAP (Maximum A Posteriori)

### Công thức
$$\hat\theta_{MAP} = \arg\max_\theta p(\theta | \mathbf{x}) = \arg\max_\theta \underbrace{p(\mathbf{x} | \theta)}_{\text{likelihood}} \underbrace{p(\theta)}_{\text{prior}}$$

→ Tương đương MLE + regularization (nếu prior chọn khéo).

### Tương ứng MAP ↔ Regularization
- Prior \(\mathcal{N}(0, \tau^2)\) cho \(\theta\) → MAP = MLE + **L2 regularizer** (\(\lambda = \sigma^2/\tau^2\))
- Prior Laplace cho \(\theta\) → MAP = MLE + **L1 regularizer** → sparsity
- Đây là foundational bridge giữa Bayesian inference và regularization classical.

### Ví dụ phía Bernoulli với Beta prior
\(X_i \sim \text{Bernoulli}(p)\), prior \(p \sim \text{Beta}(\alpha, \beta)\):
$$\hat p_{MAP} = \frac{\alpha-1 + n_{+}}{\alpha + \beta - 2 + n}$$

→ Posterior mean (Beta-Bernoulli conjugate). Khi nhiều data, prior bị "pha loãng".

### Tradeoff MLE vs MAP
- **MLE**: không prior, overfit khi data ít
- **MAP**: thêm prior, regularize, ổn định hơn khi data ít
- **Full Bayesian**: không ước điểm, mà predict posterior distribution tích hợp tất cả θ (caro expensive)

---

## 14. KL Divergence & Cross-Entropy

### KL divergence — đo độ khác biệt giữa 2 phân phối
$$D_{KL}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} \quad \text{(discrete)}$$
$$D_{KL}(P || Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx \quad \text{(continuous)}$$

- **Asymmetric**: \(D_{KL}(P || Q) \neq D_{KL}(Q || P)\)
- \(D_{KL} \geq 0\), = 0 iff \(P = Q\)
- **Không phải khoảng cách** (do asymmetry, không satisfy triangle inequality)

### Cross-entropy
$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{KL}(P || Q)$$

→ Khi minimize cross-entropy với \(P\) cố định (data distribution) = minimize KL → train \(Q\) (model) tiệm cận \(P\).

### Binary cross-entropy (BCE)
$$\text{BCE}(y, \hat y) = -[y \log \hat y + (1-y) \log(1 - \hat y)]$$

### Categorical cross-entropy (CE multi-class)
$$\text{CE}(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_k y_k \log \hat y_k$$

→ Với \(y\) one-hot, chỉ có 1 hạng: \(-\log \hat y_{\text{true class}}\).

### Ứng dụng
- **Loss chuẩn cho classification** logistic regression, NN classifier, LLM token prediction
- **Knowledge distillation**: minimize \(D_{KL}(\text{soft target} || \text{student})\)
- **VAE loss**: \(\text{loss} = \text{recon} \, \text{loss} + D_{KL}(q(z|x) || p(z))\)
- **RL policy gradient**: minimize \(D_{KL}(\pi_{\text{new}} || \pi_{\text{old}})\) trong TRPO

### Code
```python
import torch
import torch.nn.functional as F

logits = torch.tensor([1.5, 0.3, -0.5])
target = torch.tensor(1)               # class index
loss = F.cross_entropy(logits, target)  # CE multi-class

# BCE cho binary
sigmoid_out = torch.tensor(0.7)
bce = F.binary_cross_entropy(sigmoid_out, torch.tensor(1.0))
```

### Pitfall
- **Log(0) = -inf** → tất cả model dùng CE phải có logit clamp hoặc thêm epsilon. **Tránh softmax quá tự tin** (label smoothing trong LLM).
- **Asymmetry của KL**: 
  - Forward KL \(D_{KL}(P \| Q)\) (P=true) — mode-seeking (Q tập trung vào mode của P)
  - Reverse KL \(D_{KL}(Q \| P)\) (P=true) — mean-seeking (Q trải ra cover P)
- **Numerical stability**: dùng `log_softmax` rồi NLLLoss thay vì `softmax` + `CE` riêng (tránh log số rất nhỏ). `F.cross_entropy` lo vụ đó internally.

---

## 15. Probability trong LLM / Naive Bayes / VAE

### 15.1. Naive Bayes — Bayes cho classification
\(P(y | x_1, \ldots, x_d) \propto P(y) \prod P(x_i | y)\)

- **Naive**: giả định \(x_i\) độc lập cho trước \(y\) (giả định mạnh, thường sai nhưng vẫn work)
- Thường dùng cho: spam, sentiment, text classification

### 15.2. LLM autoregressive
Token prediction = phân phối categorical qua vocab (50K+):
- Output logits → softmax → categorical distribution
- Loss = categorical cross-entropy với target = token id tiếp theo
- Sampling temperature: \(p_i \propto \exp(z_i / T)\). T cao → exploration, T thấp → focused
- Top-k, top-p (nucleus) sampling: filter vocab trước khi sample

### 15.3. VAE (Variational Autoencoder)
Loss = \(\text{Recon} + D_{KL}(q(z|x) \| p(z))\)
- \(p(z) = \mathcal{N}(0, I)\) prior
- \(q(z|x)\) là encoder distribution (Gaussian)
- KL đóng vai trò regularize để z gần prior

### 15.4. Diffusion
- Forward process: thêm gaussian noise từng bước
- Backbone predict noise (epsilon) trong each step
- Probability: \(q(x_{t} | x_{t-1}) = \mathcal{N}(\sqrt{\beta_t} x_{t-1}, (1-\beta_t) I)\)

---

## 16. Pitfalls tổng hợp

| Pitfall | Hiện tượng | Fix |
|---------|-----------|-----|
| Nhầm \(P(A \| B)\) với \(P(B \| A)\) | Test 99% recision, posterior thấp | Bayes formula |
| Base rate fallacy | Bỏ prior | Phải multiply prior |
| Quên normalizer | Posterior không hợp 1 | Sum over all hypotheses |
| Correlation = causation | ρ cao → nhân quả | Randomized experiment |
| Simpson's paradox | ρ đảo chiều khi chia sub-group | Stratified analysis |
| Đọc PDF như prob | Density > 1 tưởng prob > 1 | PDF area mới là prob |
| Boolean mask sai | `numpy mask = (x > 0).sum()` tính prob | chia cho N |
| Variance MLE bias | MLE var chia n | Dùng n-1 nếu cần unbiased |
| KL không symmetric | Reverse vs Forward KL khác behavior | Chọn đúng direction |

---

## 17. Liên hệ chủ đề khác

- **Statistics** (sheet 04) — kết hợp probability để làm hypothesis testing, CI
- **Information theory** (sheet 05) — entropy, cross-entropy, KL all foundation here
- **ML** (phase 2) — LogReg = MLE với Bernoulli, Naive Bayes = Bayes trực tiếp, Kmeans ~ MLE với spherical Gaussian
- **Deep learning** (phase 3) — autograd để tính log-likelihood gradient
- **LLM** (phase 4) — softmax + categorical cross-entropy为主体 loss

---

**Sheet 03 ok. Loa todo CSV.**
