# Cheatsheet — Statistics (Thống kê) cho AI

> Phase 1 · Sheet 04 · Phase Toán
> Tận dụng `ml-math/statistics/STATISTICS_COMPLETE_REFERENCE.md` + mở rộng numpy/scipy API, pitfall, tradeoff. Tiếng Việt + giữ thuật ngữ TA gốc.

---

## 0. Roadmap sheet này

1. Population vs Sample (N vs n, Bessel's correction n-1, sampling strategies)
2. Descriptive Statistics (mean, median, mode, variance, std, IQR, skewness, kurtosis)
3. Sampling Distributions (SE, sample mean distribution)
4. Central Limit Theorem (CLT) — khi nào dùng, n=30 rule
5. Confidence Intervals (z-CI, t-CI, margin of error)
6. Hypothesis Testing (H₀/H₁, α, p-value, 6-step process)
7. P-values (interpretation, ASA statement, P-hacking, Bonferroni correction)
8. Type I & Type II Errors (α, β, power, tradeoff)
9. One-Sample T-Test (t-statistic, df, normality check)
10. Two-Sample Tests (independent t-test, paired t-test, z-test 2 mẫu)
11. ANOVA (F-statistic, SSB/SSW, post-hoc)
12. Correlation & Covariance (Pearson, Spearman, VIF, PCA connection)
13. MLE review (link probability)
14. Bayesian vs Frequentist
15. Resampling (Bootstrap, Jackknife, bagging connection)
16. A/B Testing (z-test, power analysis, sample size, pitfalls)
17. numpy/scipy API toàn bộ stats + pitfalls tổng hợp

---

## 1. Population vs Sample

### Notation chuản
| Metric | Population (N) | Sample (n) |
|--------|---------------|------------|
| Mean | μ (mu) | x̄ (x-bar) |
| Variance | σ² | s² |
| Std | σ | s |

### Công thức
Population: μ = Σ Xᵢ/N, σ² = Σ (Xᵢ−μ)²/N
Sample: x̄ = Σ xᵢ/n, s² = Σ (xᵢ−x̄)²/(n−1) (Bessel unbiased)

### Bessel's Correction (n−1)
Lý do: x̄ luôn gần sample data hơn μ thực → variance underestimate khi chia n. Chia n−1 hiệu chỉnh bias → unbiased estimator. Khi n lớn (>1000), n và n−1 gần bằng → không quan trọng. **Interview question classic**: "Tại sao sample variance dùng n−1?"

### Sampling Strategies
| Phương pháp | Mô tả | Dùng khi |
|------------|-------|---------|
| Simple Random | Mỗi phần tử xác suất bằng nhau | Default |
| Stratified | Chia strata → sample từng strata tỷ lệ | Dataset có lớp imbalance |
| Cluster | Chọn cụm ngẫu nhiên → test cả cụm | Chi phí thấp |
| Systematic | Mỗi k-th phần tử | Dễ implement |
| Convenience | Lấy gần nhất | ❌ Tránh dùng (massive bias) |

### Survivorship Bias
Phân tích sample mà bỏ qua những gì đã "chết". VD WWII: phân tích máy bay sống sót → thấy đạn ở cánh → muốn gia cố cánh. Sai! Máy bay trúng động cơ mới rơi → cần gia cố động cơ.

### Code numpy
```python
import numpy as np

sample = np.array([...])
n = len(sample)
x_bar = sample.mean()
s2 = sample.var(ddof=1)        # unbiased sample variance (n−1)
s2_mle = sample.var(ddof=0)    # MLE biased (n)
s = sample.std(ddof=1)         # sample std
```

---

## 2. Descriptive Statistics

### Mean (trung bình)
x̄ = (1/n) Σ xᵢ. Nhạy với outlier → bị kéo mạnh.

### Median (trung vị)
Giá trị ở giữa sau khi sort. Robust với outlier. Dùng median absolute deviation (MAD) cho outlier detection.

### Mode (mốt / yếu vị)
Giá trị xuất hiện nhiều nhất. Dùng cho categorical data.

### Skewness
- skew > 0: đuôi dài phải (mean > median) — lương income
- skew < 0: đuôi dài trái (mean < median) — exam score ceiling
- skew ≈ 0: symmetric (normal)

### Kurtosis
- > 3 (excess > 0): leptokurtic — đỉnh nhọn, đuôi dày (financial returns)
- = 3 (excess = 0): mesokurtic — Gaussian
- < 3 (excess < 0): platykurtic — flat py

**AI dùng**: kiểm tra phân phối feature, detect outlier trước scaling.

### Variance & Std Dev
Var = (1/n)·Σ (xᵢ−μ)², Std = σ = √Var. Std cùng đơn vị với data → dễ interpret hơn var.

### Range & IQR
- Range = max − min (nhạy outlier)
- IQR = Q3 − Q1 = interquartile range (robust). Outlier = Q1−1.5·IQR hoặc Q3+1.5·IQR

### Code numpy/scipy
```python
import numpy as np
from scipy import stats

d = np.array([10, 12, 14, 15, 18, 21, 100])  # có outlier

np.mean(d)       # mean ( 27.1 ← outlier kéo mạnh)
np.median(d)     # median = 15 (robust)
stats.mode(d)    # mode (từ scipy)
np.var(d)        # variance (biased, ddof=0)
np.std(d, ddof=1)# std unbiased
stats.skew(d)    # skewness
stats.kurtosis(d, fisher=True) # excess kurtosis (0 = Gaussian)
np.percentile(d, [25, 50, 75]) # Q1, Q2, Q3
stats.iqr(d)     # IQR: 10
```

### Pitfall
- **`np.var` mặc định ddof=0** (biased / population formula). Dùng `ddof=1` cho unbiased sample. Tra docs `ddof` always!
- **Outlier contamination**: mean không robust, dùng median cho outlier heavy data.

---

## 3. Sampling Distributions

### Standard Error (SE)
SE = σ/√n (population σ) hoặc s/√n (sample s). Đo độ chính xác của sample mean x̄.
- SE → 0 khi n → ∞ (LLN)

### Sampling distribution of x̄
**CLT**: phân phối của x̄ tiệm cận Normal N(μ, σ²/n) khi n đủ lớn (≥30), cho bất kỳ population distribution có finite variance.

### T-Distribution
Khi dùng s (sample std) thay σ → dùng t-distribution freedom n−1 thay Normal. T có đuôi dày hơn → CI rộng hơn cho small n. Khi n→∞, t → Normal.

---

## 4. Central Limit Theorem (CLT)

CLT: Nếu sample Xᵢ i.i.d. với mean μ và variance σ² (hữu hạn):
√n·(x̄−μ)/σ → N(0,1) khi n→∞

**n=30 rule**: thường dùng n≥30 để giả định normality cho x̄ via CLT.

**Ứng dụng ML**: cho phép dùng z-test hoặc t-test cho metric evaluation (accuracy, precision) dù data phân phối không chuẩn — sample mean vẫn ~Normal.

### Limitation
- Heavy-tail distribution (Cauchy — không có finite variance) → CLT không apply
- Data skewed extremly: cần transformation (log/power) trước khi apply CLT

---

## 5. Confidence Intervals (CI)

### z-CI (known σ)
x̄ ± z_{α/2} · σ/√n
z_{0.025} = 1.96 cho 95% CI

### t-CI (unknown σ, dùng s)
x̄ ± t_{α/2, df=n−1} · s/√n
t_{0.025, df=n−1} > 1.96 (đuôi dày, CI rộng hơn)
Khi n>30, t ≈ z

### Interpretation (đúng — tricky interview)
"95% CI means: if we repeated this experiment 100 times, 95 of the resulting CIs would contain the true μ." **KHÔNG phải**: "probability μ nằm trong CI là 95%". CI là frequentist concept — tham số μ là cố định (unknown), CI phủ μ với probability only pre-experiment.

### Code scipy
```python
from scipy import stats
import numpy as np

data = np.random.normal(3, 1, 50)
n = len(data)
mean = data.mean()
sem = data.std(ddof=1) / np.sqrt(n)  # standard error

# t-CI 95%
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
print(ci)   # (lower, upper)

# Nếu biết σ (dùngNormal):
ci_z = stats.norm.interval(0.95, loc=mean, scale=sigma/np.sqrt(n))
```

Margin of Error = z·(σ/√n). Giảm margin → tăng n 4x (vì 1/√n).

---

## 6. Hypothesis Testing

### 6-step process
1. **State H₀, H₁**: H₀ (null — skeptic, no effect), H₁ (alternative — discovery)
2. **Choose α** (thường 0.05)
3. **Select test** (z/t/z-two-sample/χ²...)
4. **Calculate test statistic** từ data
5. **Compare p-value vs α**: p≤α → reject H₀, p > α → fail to reject
6. **Draw conclusion** in context. "Reject H₀" ≠ "prove H₁" (chỉ evidence).

### z-test (1 sample, known σ)
z = (x̄ − μ₀) / (σ/√n)
So với z-critical (α=0.05, two-tailed → ±1.96)

### t-test (1 sample, unknown σ)
t = (x̄ − μ₀) / (s/√n)
So với t-critical, df = n−1

### Code scipy
```python
from scipy import stats

# 1-sample t-test (H₀: μ=μ₀)
t_stat, p_value = stats.ttest_1samp(data, popmean=3.0)

# 2-sample independent t-test (equal_var=True/False)
t_stat, p_value = stats.ttest_ind(group_A, group_B, equal_var=False)

# paired t-test (same subjects, before/after)
t_stat, p_value = stats.ttest_rel(before, after)

# z-test (manual code): z = (x̄ − μ₀)/(σ/√n), p = 2*(1−norm.cdf(|z|))
```

### Pitfall
- **Type statistical vs practical significance**: với sample cỡ, p luôn < 0.05 dù effect tiny. Đánh cả effect size (Cohen's d).
- **Multiple testing**: test 20 hypothesis, α=0.05 → expect 1 false positive by chance. Bonferroni: α/m cho m tests (`α → 0.05/20 = 0.0025`).

---

## 7. P-values Explained

**ASA 2016 Statement (correct interpretation)**:
"P-value is the probability, under H₀, of obtaining a test statistic at least as extreme as the one observed." → Đo tính extreme của data giả định H₀ đúng.

### Wrong interpretation (common pitfalls):
❌ "p = P(H₀ true)" — p không đo xác suất hypothesis
❌ "p < 0.05 → effect size lớn" — p chỉ đo evidence, không đo magnitude

### P-hacking
Manipulate data đến khi p<0.05: remove outlier theo bias, change H₀ post-hoc, stop experiment at p<0.05 (peeking), test nhiều hypotheses không correction.

### Fix
- Pre-register hypothesis analysis plan
- Bonferroni / FDR correction cho multiple testing
- Đánh effect size + CI cùng p-value
- A/B testing: không peek giữa kỳ

---

## 8. Type I & Type II Errors

| | H₀ True | H₀ False |
|---|---|---|
| **Fail to reject H₀** | ✅ | ❌ Type II Error (β) |
| **Reject H₀** | ❌ Type I Error (α) | ✅ (Power 1−β) |

- **α** (Type I): false positive — kết luận có effect khi không có. α=0.05 → 5% risk.
- **β** (Type II): false negative — miss effect thực sự. 1−β = Power.
- **Power analysis**: tìm sample size cần thiết để có power 0.8 (80%) với effect size biết trước. Nhỏ sample → low power → dễ Type II error.

### Tradeoff: α vs β
Giảm α → tăng β (harder to prove effect → more miss true effects). Tăng sample → improve cả 2.

### Code for power
```python
from statsmodels.stats.power import TTestIndPower
analysis = TTestIndPower()
n_needed = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)
```

---

## 9. One-Sample T-Test

t-statistic: t = (x̄ − μ₀) / (s/√n)

df = n−1 (degrees of freedom).

### Assumptions kiểm
- Data roughly symmetric hoặc n≥30 (CLT hỗ trợ)
- Kiểm normality: Shapiro-Wilk (small n) hoặc Q-Q plot + `stats.probplot`

```python
# Normality check
from scipy import stats
W_stat, p_shap = stats.shapiro(data)  # p<0.05: reject normality
```

---

## 10. Two-Sample Tests

### Independent t-test (Welch's default)
So sánh mean 2 group độc lập: t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂). Welch df approximate (không n₁+n₂−2 như Student t test). ***Always use Welch in practice*** — Student assume equal variance (rare true).

### Paired t-test
Trước sau trên cùng subject → difference dᵢ = afterᵢ − beforeᵢ → test mean của d vs 0.

### z-test 2 proportions (A/B conversion)
z = (p̂₁ − p̂₂) / √(p̂_pool·(1−p̂_pool)·(1/n₁+1/n₂))

### Code scipy
```python
stats.ttest_ind(A, B, equal_var=False)  # Welch
stats.ttest_rel(before, after)          # paired

# z-test proportion (manual):
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_val = proportions_ztest([success_A, success_B], [n_A, n_B])
```

---

## 11. ANOVA (Analysis of Variance)

### So sánh ≥ 3 group
H₀: μ₁ = μ₂ = ... = μ_k. H₁: ít nhất 1 khác.

**F-statistic**: F = (SS_between / df_between) / (SS_within / df_within)
- SS_between = variance giữa group (do treatment)
- SS_within = variance trong group (do random)
- Nếu F lớn → group khác nhau significantly

### One-way ANOVA (scipy)
```python
f_stat, p_val = stats.f_oneway(group1, group2, group3)
```

### Post-hoc test (nếu reject H₀)
Tukey HSD để biết cặp nào khác nhau (kiểm all pairs, control FWER).

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd
```

### Assumptions
- Independence của observations
- Normality trong mỗi group (không quá nghiêm với n lớn)
- Homogeneity of variance (Levene test: `stats.levene(g1, g2, g3)`)
- Nếu Levene reject → dùng Welch ANOVA hoặc Kruskal-Wallis (non-parametric)

### Tradeoff
- **ANOVA checks overall effect** → không biết cặp nào khác nhau → cần post-hoc
- **Multiple independent t-tests** → inflated Type I error (via multiple testing) → ANOVA kiểm soát FWER

---

## 12. Correlation & Covariance

### Covariance (cùng đv với data²)
Cov(X,Y) = E[(X−μ_X)(Y−μ_Y)]. Scale dep on unit → không chuẩn hóa.

### Pearson correlation ρ
ρ = Cov(X,Y)/(σ_X·σ_Y) ∈ [−1, 1]. ρ measures linear relationship.

### Spearman rank correlation (rho_s)
Non-parametric, measure monotonic relationship. Robust outlier.

### Code
```python
np.cov(X.T)                     # covariance
np.corrcoef(X.T)                # Pearson
stats.spearmanr(x, y)           # Spearman
```

### VIF (Variance Inflation Factor) — cho ML
VIFᵢ = 1/(1−R²ᵢ) where R²ᵢ từ regression của feature i lên features khác. VIF > 10 → multicollinearity issue → cần PCA hoặc bỏ feature.

### PCA connection
PCA dùng covariance matrix → eigenvector = principal component direction. correlation matrix cho PCA khi features scale khác (standardize trước).

### Pitfall
- **ρ cao không phải causation** (confound temperature → ice cream & drowning correlation)
- **Simpson paradox**: ρ aggregate đảo chiều khi stratify subgroup → check correlation per strata
- **ρ chỉ đo linear** — relationship mapping Y = X², ρ=0 nhưng strong nonlinear relationship. Dùng Spearman cho monotonic check.

---

## 13. MLE (link probability)

Đã bàn trong Probability sheet 03. Tiếp border:
- MLE với Gaussian noise → MSE loss
- MLE với Bernoulli output → BCE loss
- MLE variance biased chia n (đúng frequentist approach); unbiased dùng n−1

---

## 14. Bayesian vs Frequentist

| Aspect | Frequentist | Bayesian |
|--------|------------|----------|
| Parameter θ | Fixed unknown constant | Random variable with distribution |
| Inference | p-value, CI, test statistic | Posterior distribution, credible interval |
| Prior knowledge | Không đưa vào | Có thể encode qua prior |
| Complex model | Hard to integrate | MCMC for sampling posterior |
| Computational | NHẸ (analytic) | NẶNG (MCMC/sampling) — nhưng modern với variational inference nhẹ hơn |
| Interpretation CI | "100 repetitions → 95 capture θ" | "95% posterior probability θ in [a,b]" — intuitive hơn |

**Tradeoff**:
- Frequentist: công thức có sẵn, computational nhẹ, cho big data enforceable. Nhưng CI interpretation counter-intuitive.
- Bayesian: intuitive, encode prior → ổn khi data ít. Nhưng prior choice subjective (prior sensitivity analysis), computational heavy.

**Khi nào Bayesian cho AI**:
- **VAE**: prior p(z)=N(0,I), poster q(z|x) via encoder → Bayesian inference via variational (máy ảo)
- **A/B Bayesian**: Thompson Sampling Bandit cho adaptive allocation
- **Naive Bayes**: prior class ratio

---

## 15. Resampling Methods: Bootstrap & Jackknife

### Bootstrap (Efron 1979)
- Sample B lần **with replacement** từ original sample (cùng cỡ n)
- Mỗi bootstrap sample → calculate estimate (ví dụ median)
- Phân phối bootstrap approximates sampling distribution của estimator
- Dùng: CI bootstrap (không cần assume normality), model evaluation bootstrap .632+

### Offshoot: Bagging (ML)
Bootstrap aggregation (Random Forest): train model trên bootstrap replicate → ensemble reduce variance.

### Jackknife
- Leave-one-out: bỏ 1 observation, estimate trên n−1 điểm
- Repeat cho mỗi observation → n estimates
- Variance estimate: [(n−1)/n] * Σ (θ̂_j − θ̄)²
- Dùng: bias correction, estimator stability

### Code
```python
import numpy as np

# Bootstrap median CI
data = np.array([...])
B = 1000
n = len(data)
boot_medians = [np.median(np.random.choice(data, size=n, replace=True)) for _ in range(B)]
ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
```

---

## 16. A/B Testing for ML Engineers

### Setup
- H₀: conversion A = conversion B (μ_A = μ_B)
- H₁: khác nhau
- α = 0.05, Power = 0.80 (standard)

### Power analysis (trước launch)
1. Estimate baseline conversion rate (của A)
2. Define Minimum Detectable Effect (MDE — practical significance)
3. Calculate sample size needed:
```python
from statsmodels.stats.power import NormalIndPower
analysis = NormalIndPower()
n = analysis.solve_power(effect_size=(p̂_B−p̂_A)/sqrt(p̂_pool*(1−p̂_pool)),
                         power=0.8, alpha=0.05, alternative='two-sided')
```

### z-test 2 proportions (sau experiment)
```python
from statsmodels.stats.proportion import proportions_ztest
success = [conv_A, conv_B]
nobs = [n_A, n_B]
z, p = proportions_ztest(success, nobs)
if p <= 0.05: reject H₀ → significant difference
```

### 4 pitfalls A/B
1. **Peeking (P-hacking)** — xem data giữa chừng, stop khi p<0.05 → inflated Type I. Fix: fixed duration hoặc sequential testing (valid).
2. **Novelty effect** — users click new version vì lạ → overestimate. Run long enough to stabilize.
3. **Simpson's Paradox** — aggregate thắng nhưng loser every subgroup (X của Google trên gender admission). Fix: stratified A/B hoặc segment analysis.
4. **Network effects / Spillover** — users in control group influenced by treatment (social/ Uber). Fix: cluster-level randomization.

### ML Applications
- **Model deployment A/B**: new model vs old model → metrics (CTR, retention)
- **Multi-Armed Bandit (Thompson Sampling)**: adaptive allocate traffic → minimize regret
- **Hyperparameter tuning evaluation**: CV thường đánh, nhưng production A/B override

---

## 17. Tổng hợp numpy/scipy API stats

```python
import numpy as np; import scipy.stats as st

# Descriptive
np.mean(d); np.median(d); np.std(d, ddof=1)       # Std unbiased
np.var(d, ddof=1); np.percentile(d, [25,75])
st.skew(d); st.kurtosis(d, fisher=True); st.iqr(d); st.mode(d)

# Sampling & distribution
np.random.seed(42); np.random.normal(μ,σ,N)
rng = np.random.default_rng(42); rng.normal(...)

# Confidence interval
st.t.interval(0.95, df=n-1, loc=mean, scale=sem)
st.norm.interval(0.95, loc=mean, scale=σ/√n)

# Hypothesis tests
st.ttest_1samp(data, popmean=μ₀)
st.ttest_ind(A, B, equal_var=False)    # Welch default
st.ttest_rel(before, after)
st.f_oneway(g1, g2, g3)                # ANOVA 1-way
st.shapiro(data)                       # normality

# Correlation
np.corrcoef(X.T)
st.spearmanr(x, y)

# Resampling
boot_stat = [np.mean(np.random.choice(d, n, replace=True)) for _ in range(1000)]

# Power analysis
from statsmodels.stats.power import TTestIndPower; analysis=TTestIndPower()
n = analysis.solve_power(effect_size=0.5, power=0.8, alpha=0.05)

# 2-proportion z-test
from statsmodels.stats.proportion import proportions_ztest
```

---

## 18. Pitfalls tổng hợp

| Pitfall | Hiện tượng | Fix |
|---------|-----------|-----|
| `np.var(ddof=0)` mặc định | Variance biased cho sample | `ddof=1` cho unbiased |
| Sample quá nhỏ → SE lớn | CI rộng | Tăng n, bootstrap |
| Outlier contamination | Mean bị kéo mạnh | Median/IQR, log transform |
| P-hacking / peeking | α inflated | Sequential test, Bonferroni |
| Nhầm p-value vs P(H₀ true) | Misinterpret evidence | ASA statement |
| Multiple pairwise t-test vs ANOVA | Type I erro accumulation >20 group | ANOVA + Tukey post-hoc |
| Student t test khi var không bằng | T-stat bias | Welch t-test luôn select |
| Pearson ρ cho nonlinear | Quan hệ bỏ qua | Spearman, Mutual Info |
| n<30 không check normality | T-test unreliable | Shapiro-Wilk, bootstrap instead |

---

## 19. Liên hệ chủ đề khác

- **Probability** (sheet 03) — CLT, MLE, Bayesian framework lái statistics
- **Machine Learning** (Phase 2) — CV split, model eval via confidence, bias-variance tradeoff
- **Deep Learning** (Phase 3) — weight decay (L2 = MAP with Normal prior)
- **Information Theory** (sheet 05) — connection mutual info vs correlation nonlinear

---

**Sheet 04 ok. ✅_