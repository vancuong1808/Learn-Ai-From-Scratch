# 📊 STATISTICS COMPLETE REFERENCE — ML/AI Engineering

> **Nguồn:** TensorTonic (tensortonic.com/ml-math/statistics)
> **Tổng hợp:** 15 chủ đề thống kê từ cơ bản đến nâng cao
> **Mục đích:** Tài liệu tham khảo toàn diện cho AI Agent, ML Engineer

---

## 📑 MỤC LỤC

| # | Chủ đề | Từ khoá chính |
|---|--------|--------------|
| 1 | [Population vs Sample](#1-population-vs-sample) | N vs n, Bias, Bessel's Correction, Inference |
| 2 | [Descriptive Statistics](#2-descriptive-statistics) | Mean, Median, Mode, Variance, Std Dev, IQR |
| 3 | [Sampling Distributions](#3-sampling-distributions) | SE, CLT, T-Distribution, df |
| 4 | [Central Limit Theorem (CLT)](#4-central-limit-theorem-clt) | Normal approximation, n=30 rule |
| 5 | [Confidence Intervals](#5-confidence-intervals) | Z vs T, Margin of Error, 95% CI |
| 6 | [Hypothesis Testing](#6-hypothesis-testing) | H₀, H₁, α, P-value, Z-test, 6-step process |
| 7 | [P-Values](#7-p-values) | Tail area, ASA, P-hacking, Bonferroni |
| 8 | [Type I & Type II Errors](#8-type-i--type-ii-errors) | α, β, Power, Trade-off |
| 9 | [One-Sample T-Test](#9-one-sample-t-test) | T-statistic, df, Normality check |
| 10 | [A/B Testing](#10-ab-testing) | Z-test 2 mẫu, Power Analysis, Pitfalls |
| 11 | [ANOVA](#11-anova) | F-statistic, SSB, SSW, Post-Hoc |
| 12 | [Correlation & Covariance](#12-correlation--covariance) | Pearson, Spearman, VIF, PCA |
| 13 | [Maximum Likelihood Estimation (MLE)](#13-maximum-likelihood-estimation-mle) | Log-Likelihood, MSE=MLE, Cross-Entropy |
| 14 | [Bayesian vs Frequentist](#14-bayesian-vs-frequentist) | Prior, Posterior, MAP, Credible Interval |
| 15 | [Resampling Methods](#15-resampling-methods-bootstrap--jackknife) | Bootstrap, Jackknife, Bagging, OOB |

---

---

## 1. Population vs Sample

> **Source:** https://www.tensortonic.com/ml-math/statistics/population-sample

### Khái niệm cốt lõi

**Population (N):** Toàn bộ tập dữ liệu — "sự thật" không thể quan sát hoàn toàn.
**Sample (n):** Tập con được chọn để ước lượng sự thật của Population.

**Ví dụ trực quan:** Soup analogy — Population = cả nồi súp; Sample = một muỗng. Nếu không khuấy đều → **sampling bias**.

### Ký hiệu chuẩn

| Metric | Population Parameter | Sample Statistic | Ý nghĩa |
|--------|---------------------|-----------------|---------|
| Size | N | n | n ≪ N |
| Mean | μ (Mu) | x̄ (x-bar) | x̄ ước lượng μ |
| Variance | σ² | s² | s² ước lượng σ² |
| Std Dev | σ | s | s ước lượng σ |
| Proportion | p | p̂ | p̂ ước lượng p |

### Công thức

**Population:**
$$\mu = \frac{\sum X_i}{N}, \quad \sigma^2 = \frac{\sum (X_i - \mu)^2}{N}$$

**Sample:**
$$\bar{x} = \frac{\sum x_i}{n}, \quad s^2 = \frac{\sum (x_i - \bar{x})^2}{n-1}$$

### Bessel's Correction — Tại sao dùng n-1?

Khi dùng x̄ thay μ trong công thức variance, kết quả bị **underestimate** vì sample data luôn gần x̄ hơn μ thực. Chia cho **n-1** thay vì n để **hiệu chỉnh bias** này.

> 💡 **Interview question:** "Tại sao sample variance dùng n-1?" → Bessel's Correction để unbiased estimator.

### Sampling Strategies

| Phương pháp | Mô tả | Ưu điểm | Nhược điểm |
|------------|-------|---------|-----------|
| Simple Random | Mỗi phần tử có xác suất bằng nhau | Statistically pure | Khó thực hiện với dataset lớn |
| Stratified | Chia nhóm → sample từng nhóm tỷ lệ | Đảm bảo representation | Cần biết các strata |
| Cluster | Chọn ngẫu nhiên các cụm → test toàn bộ cụm | Rẻ hơn | Giả định mỗi cụm đại diện tổng thể |
| Systematic | Chọn mỗi k-th phần tử | Dễ thực hiện | Có thể bị bias theo chu kỳ |
| Convenience | Chọn cái dễ lấy nhất | Nhanh | **Massive selection bias** — tránh dùng |

### Survivorship Bias — WWII Planes Example

Chỉ phân tích máy bay **sống sót** → thấy đạn ở cánh → muốn gia cố cánh. Sai! Máy bay bị trúng **động cơ** mới rơi — đó là chỗ cần gia cố.

> 🔑 **Bài học ML:** Dataset chỉ chứa record của những gì "tồn tại được". Hỏi: **What's missing?**

### ML Applications

- **Training data = Sample:** Real world = Population. Ta muốn minimize Population error nhưng chỉ tối ưu được Training error.
- **Overfitting:** Model học "noise" của sample thay vì "signal" của population.
- **Train/Test Split:** Test set = giả lập Population.
- **Big Data Fallacy:** n lớn ≠ representative nếu vẫn bị selection bias.
- **Data Leakage:** Dùng thông tin từ Test/Population để ảnh hưởng Training set.

---

---

## 2. Descriptive Statistics

> **Source:** https://www.tensortonic.com/ml-math/statistics/basic-statistics

### Measures of Center

#### Mean (Trung bình)
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- Dùng khi data **đối xứng, không có outlier**
- Nhạy cảm với outlier (ví dụ: một CEO $1M kéo mean lên rất cao)

#### Median
- Giá trị ở giữa khi sắp xếp
- Dùng khi data **lệch (skewed) hoặc có outlier**

#### Mode
- Giá trị xuất hiện nhiều nhất
- Dùng cho **categorical data**

### Skewness (Độ lệch)

| Loại | Skewness | Quan hệ Mean/Median/Mode |
|------|---------|------------------------|
| Symmetric | ≈ 0 | Mean ≈ Median ≈ Mode |
| Right-Skewed (+) | > 0 | Mode < Median < Mean |
| Left-Skewed (−) | < 0 | Mean < Median < Mode |

> 💡 **ML Tip:** Data lệch nhiều (house prices) → apply log transform trước khi train model.

### Kurtosis (Độ nhọn đuôi)

$$\text{Kurtosis} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{\sigma}\right)^4$$

| Loại | Kurtosis | Ý nghĩa |
|------|---------|---------|
| Platykurtic | < 3 | Đuôi mỏng, đỉnh phẳng — ít outlier |
| Mesokurtic | = 3 | Normal distribution — baseline |
| Leptokurtic | > 3 | Đuôi dày, đỉnh nhọn — nhiều outlier |

### Measures of Spread

#### Variance & Standard Deviation

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2, \quad \sigma = \sqrt{\sigma^2}$$

**Tại sao bình phương deviation?**
1. Loại bỏ việc các giá trị âm/dương triệt tiêu nhau
2. Phạt nặng hơn các deviation lớn (outlier-sensitive)

#### Range và IQR

$$\text{Range} = \text{Max} - \text{Min}$$
$$\text{IQR} = Q_3 - Q_1$$

**Outlier detection rule:** Flagged nếu < Q₁ - 1.5×IQR hoặc > Q₃ + 1.5×IQR

| Metric | Dùng khi |
|--------|---------|
| Range | Quick overview, không có outlier |
| IQR | Data có outlier (robust) |
| Variance | Mathematical analysis, ML algorithms |
| Std Dev | Interpretable spread (same units as data) |

### Formulas Reference Table

| Metric | Công thức | Dùng khi |
|--------|----------|---------|
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | Data symmetric, no outliers |
| Median | Middle value (sorted) | Skewed data hoặc có outliers |
| Mode | Most frequent value | Categorical data |
| Range | Max − Min | Quick overview |
| IQR | Q₃ − Q₁ | Robust spread, outlier detection |
| Variance | $\sigma^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ | ML algorithms |
| Std Dev | $\sigma = \sqrt{\sigma^2}$ | Interpretable spread |

---

---

## 3. Sampling Distributions

> **Source:** https://www.tensortonic.com/ml-math/statistics/sampling-distribution

### Định nghĩa

**Sampling Distribution** = Phân phối của một **statistic** (vd: sample mean) được tính từ **nhiều samples khác nhau** cùng cỡ n.

Không phải phân phối của raw data — mà là phân phối của **các thống kê tính từ data**.

### Sampling Distribution của Mean

Nếu population có mean μ và std σ:

**Rule 1: Unbiased Estimator**
$$\mu_{\bar{x}} = \mu$$
Average của tất cả sample means = true population mean.

**Rule 2: Standard Error**
$$SE = \sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$$

### Standard Deviation vs Standard Error — Interview Classic

| Metric | Symbol | Formula | Đo lường |
|--------|--------|---------|---------|
| Standard Deviation | σ hoặc s | σ | Variability của **individual data points** |
| Standard Error | SE | σ/√n | Variability của **sample mean** |

> 💡 **Square Root Law:** Để giảm SE xuống một nửa, cần **quadruple** sample size (×4).

### T-Distribution

Khi không biết σ thực (gần như luôn luôn!), dùng **sample standard deviation s** → phân phối kết quả là **Student's t-distribution**.

$$t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$$

- **Fat tails** hơn Normal để bù đắp uncertainty khi n nhỏ
- Độ "béo" phụ thuộc vào **Degrees of Freedom (df) = n - 1**
- Khi df > 30, t-distribution ≈ Normal distribution

### Sampling Distribution của Proportions

$$\hat{p} = \frac{x}{n}, \quad \mu_{\hat{p}} = p, \quad \sigma_{\hat{p}} = \sqrt{\frac{p(1-p)}{n}}$$

**Validity check cho Normal approximation:**
- np ≥ 10 (ít nhất 10 successes)
- n(1-p) ≥ 10 (ít nhất 10 failures)

### ML Applications

| Ứng dụng | Vai trò của Sampling Distribution |
|---------|----------------------------------|
| Cross-Validation | 5 fold scores → sample means; SE đo độ ổn định |
| Ensemble/Bagging | Average n trees → SE = σ/√n giảm variance |
| A/B Testing | Compare model metrics → CI từ sampling dist |
| Mini-batch SGD | Gradient từ batch = ước lượng true gradient; larger batch → lower SE |

---

---

## 4. Central Limit Theorem (CLT)

> **Source:** https://www.tensortonic.com/ml-math/statistics/central-limit-theorem

### Phát biểu

> Khi n đủ lớn, phân phối của **sample means** xấp xỉ **Normal Distribution**, bất kể phân phối gốc của population là gì.

$$\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right)$$

### 3 Properties của CLT

| Property | Nội dung |
|---------|---------|
| **Shape** | Distribution của sample means → Normal (bell-shaped) |
| **Center** | $\mu_{\bar{x}} = \mu$ (unbiased) |
| **Spread** | $\sigma_{\bar{x}} = \sigma/\sqrt{n}$ (shrinks with n) |

### Standardized Form

$$Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)$$

### Assumptions (Điều kiện áp dụng)

1. **Independence:** Các quan sát phải độc lập (i.i.d.)
2. **Random Sampling:** Sample ngẫu nhiên từ population
3. **Finite Variance:** σ² < ∞ (CLT không áp dụng cho Cauchy distribution)
4. **Sample Size:** n ≥ 30 (rule of thumb)

### The Magic Number n = 30

- Với **n < 30:** sampling distribution vẫn giữ skew của population → Z-test/T-test không valid
- Với **n ≥ 30:** đủ để xấp xỉ Normal với hầu hết distributions
- Với data **heavy skew:** cần n = 50-100+

### Solved Examples

**Example 1: Male Weight**
- μ = 70kg, σ = 15kg, n = 50
- $\mu_{\bar{x}} = 70kg$
- $\sigma_{\bar{x}} = 15/\sqrt{50} \approx 2.12kg$

**Example 2: Probability Calculation**
- μ = 160cm, σ = 10cm, n = 25
- P(x̄ > 162)?
- SE = 10/√25 = 2
- Z = (162−160)/2 = 1.0
- P = 1 − 0.8413 = **15.87%**

### Real-World Applications

| Domain | Cách CLT áp dụng |
|--------|----------------|
| A/B Testing | Individual behavior là Bernoulli → average conversion rate là Normal |
| Random Forest | Average predictions → error là normally distributed (thấp hơn individual) |
| Quality Control | Sample averages → Normal → control charts |
| Monte Carlo | Average of simulations → converges to true value |

---

---

## 5. Confidence Intervals

> **Source:** https://www.tensortonic.com/ml-math/statistics/confidence-interval

### Tại sao cần CI?

Point estimate ("mean = 165cm") không communicate uncertainty. CI cung cấp một **range** thay vì một số.

**Fishing Net Analogy:**
- Point estimate = ném giáo vào hồ → có thể trượt
- Confidence Interval = thả lưới → bắt được cá (true parameter) với xác suất cao

### Công thức

$$CI = \bar{x} \pm \underbrace{z^* \times \frac{\sigma}{\sqrt{n}}}_{\text{Margin of Error}}$$

### Critical Values (Z-scores)

| Confidence Level | Alpha (tail area) | Critical Value (z*) |
|-----------------|-------------------|---------------------|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | **1.960** |
| 99% | 0.01 | 2.576 |

### Z-Score vs T-Score — Khi nào dùng cái nào?

| Scenario | Distribution | Formula |
|---------|-------------|---------|
| n ≥ 30 hoặc σ biết | Z-Distribution (Normal) | $\bar{x} \pm z^* \frac{\sigma}{\sqrt{n}}$ |
| n < 30 VÀ σ không biết | Student's T-Distribution | $\bar{x} \pm t^* \frac{s}{\sqrt{n}}$ |

**Tại sao dùng T thay Z khi n nhỏ?** T-distribution có "fatter tails" để bù đắp extra uncertainty khi ước lượng σ từ sample nhỏ.

### Các yếu tố ảnh hưởng đến độ rộng CI

| Yếu tố | Tăng CI rộng | Thu hẹp CI |
|--------|-------------|-----------|
| Sample size n | n nhỏ | **Tăng n** (quadruple n → halve ME) |
| Confidence Level | Tăng CL (99%) | Giảm CL (90%) |
| Std Dev σ | σ cao | σ thấp |

### ⚠️ The 95% TRAP — Hiểu đúng CI

**SAI:** "Có 95% xác suất true mean nằm trong [160, 170]."

**ĐÚNG:** "Nếu lặp lại sampling procedure nhiều lần, 95% các intervals được tạo ra sẽ chứa true population mean."

> True parameter là **fixed constant** (cá không di chuyển). Interval là thứ biến đổi (vị trí của lưới thay đổi). Sau khi tính CI cụ thể, cá hoặc ở trong lưới (P=1) hoặc không (P=0).

### Calculation Examples

**Z-Interval (Large N):**
- 121 orders, 102 shipped fast. 95% CI cho proportion:
- p̂ = 102/121 = 0.84
- SE = √(0.84×0.16/121) ≈ 0.033
- ME = 1.96 × 0.033 = 0.065
- **CI: [0.775, 0.905]**

**T-Interval (Small N):**
- n=6, x̄=101.82, s=1.2
- df = 5, t* = 2.571
- SE = 1.2/√6 ≈ 0.49
- **CI: [100.56, 103.08]**

### ML Applications

| Tình huống | Cách dùng CI |
|-----------|-------------|
| A/B Testing | CI của difference; nếu CI chứa 0 → không significant |
| Cross-Validation | Report "83.0% ± 1.5% (95% CI)" thay vì chỉ "83%" |
| Regression Coefficients | Nếu CI của coefficient chứa 0 → feature không significant |

---

---

## 6. Hypothesis Testing

> **Source:** https://www.tensortonic.com/ml-math/statistics/hypothesis-testing

### Core Question

"Model A accuracy 85%, Model B 86% — Model B có thực sự tốt hơn hay chỉ là may mắn?"

### Courtroom Analogy

| Toà án | Thống kê |
|--------|---------|
| "Bị cáo vô tội" | H₀ (Null Hypothesis) |
| "Bị cáo có tội" | H₁ (Alternative Hypothesis) |
| "Không đủ bằng chứng" | Fail to Reject H₀ |
| Không bao giờ nói "Innocent" | Không bao giờ "Accept H₀" |

### Core Definitions

**Null Hypothesis (H₀):** Không có effect/difference. H₀: μ = 50

**Alternative Hypothesis (H₁):** Có difference. H₁: μ ≠ 50

**Significance Level (α):** Xác suất tối đa chấp nhận Type I Error. Thường α = 0.05.

**P-Value:** P(data | H₀) — xác suất thấy data này nếu H₀ đúng.

```
P-value ≤ α  →  Reject H₀ (evidence is significant)
P-value > α  →  Fail to Reject H₀ (insufficient evidence)
```

### One-Tailed vs Two-Tailed Tests

| Test Type | H₁ | Rejection Region |
|----------|-----|-----------------|
| Two-Tailed | μ ≠ 50 | Cả hai đuôi |
| Right-Tailed | μ > 50 | Đuôi phải |
| Left-Tailed | μ < 50 | Đuôi trái |

### Choosing the Right Test

| Test | Khi nào dùng |
|------|------------|
| **Z-Test** | σ biết HOẶC n > 30 |
| **T-Test** | σ không biết VÀ n < 30 |
| **Chi-Square** | Categorical data (observed vs expected counts) |
| **ANOVA** | So sánh means của 3+ groups |

### Type I & Type II Errors

| | H₀ TRUE | H₀ FALSE |
|--|---------|---------|
| **Reject H₀** | **Type I Error** (α) = False Positive | Correct (Power = 1-β) |
| **Fail to Reject H₀** | Correct (True Negative) | **Type II Error** (β) = False Negative |

### The 6-Step Process

```
1. State Hypotheses  →  Định nghĩa H₀ và H₁ rõ ràng TRƯỚC khi thu thập data
2. Choose α          →  Thường α = 0.05
3. Collect Data      →  Proper experimental design
4. Calculate Stat    →  Z, T, or Chi-Square
5. Find P-Value      →  Compare statistic với distribution dưới H₀
6. Make Decision     →  Reject hay Fail to Reject; nêu practical significance
```

### Worked Example: Z-Test (IQ Scores)

- μ = 100, σ = 15, n = 36, x̄ = 106
- H₀: μ = 100, H₁: μ > 100 (right-tailed)
- Z = (106 − 100) / (15/√36) = 6/2.5 = **2.4**
- Critical Z (α=0.05, one-tail) = 1.645
- 2.4 > 1.645 → **Reject H₀**

### Limitations & Pitfalls

| Pitfall | Mô tả |
|--------|-------|
| **File Drawer Effect** | Chỉ publish significant results → literature bias |
| **Statistical vs Practical Significance** | n lớn → p nhỏ dù effect cực nhỏ |
| **P-Hacking** | Chạy nhiều tests cho đến khi p < 0.05 → false positives |

---

---

## 7. P-Values

> **Source:** https://www.tensortonic.com/ml-math/statistics/p-value

### Intuition: Surprise Meter

P-value = Mức độ **bất ngờ** nếu H₀ thực sự đúng.

| Scenario | P-value | Ý nghĩa |
|---------|---------|---------|
| 5H 5T (10 flips) | ≈ 1.0 | Không ngạc nhiên — đúng như expected |
| 9H 1T | ≈ 0.02 | Bắt đầu nghi ngờ |
| 10H 0T | ≈ 0.001 | Cực kỳ bất thường → Reject H₀! |

### Định nghĩa chính thức

$$P\text{-value} = P(\text{Data} | H_0)$$

**Là:** Xác suất của **evidence** (data) nếu hypothesis đúng.
**KHÔNG phải:** Xác suất hypothesis đúng — đây là sai lầm phổ biến nhất!

### P-Value Fallacies (ASA Statement 2016)

| Sai lầm phổ biến | Sự thật |
|-----------------|---------|
| "P=0.05 → 95% chắc H₁ đúng" | SAI. P-value không nói gì về xác suất H₀/H₁ đúng |
| "P > 0.05 → không có effect" | SAI. Có thể sample size quá nhỏ (underpowered) |
| "P=0.04 tốt hơn hẳn P=0.06" | SAI. 0.05 là ngưỡng arbitrary; treat p-value là continuous |

### Bảng diễn giải P-Value

| P-Value Range | Mức độ bằng chứng chống H₀ | Diễn giải |
|--------------|--------------------------|---------|
| P < 0.001 | Rất mạnh | Strong rejection |
| 0.001 – 0.01 | Mạnh | Usually reject |
| 0.01 – 0.05 | Trung bình | Conventionally "significant" |
| 0.05 – 0.10 | Yếu | Worth investigating |
| P > 0.10 | Rất yếu | Fail to reject |

### P-Hacking & Multiple Comparisons Problem

**Vấn đề:** 100 tests độc lập với H₀ thực sự đúng, α=0.05 → kỳ vọng **5 false positives** chỉ do may mắn.

**Bonferroni Correction:**
$$\alpha_{\text{corrected}} = \frac{\alpha}{n}$$
Ví dụ: 20 tests → dùng α = 0.05/20 = **0.0025**

### ML Application: Backward Elimination (Feature Selection)

```
1. Train model với ALL features
2. Tính p-value cho coefficient của mỗi feature
   H₀: Coefficient = 0 (feature vô dụng)
3. Feature có p-value cao nhất (vd: 0.85)
4. Nếu P > 0.05 → drop feature đó
5. Retrain, lặp lại cho đến khi tất cả P < 0.05
```

---

---

## 8. Type I & Type II Errors

> **Source:** https://www.tensortonic.com/ml-math/statistics/type-one-vs-two

### Decision Matrix (Phải thuộc lòng)

|  | H₀ TRUE (không có effect) | H₀ FALSE (có effect thực) |
|--|--------------------------|--------------------------|
| **Reject H₀** | **Type I Error** (False Positive) — Prob = α | Correct ✅ (True Positive) — Power = 1-β |
| **Fail to Reject H₀** | Correct ✅ (True Negative) | **Type II Error** (False Negative) — Prob = β |

> **Memory Trick:** Type **I** = **I**ncorrectly reject TRUE null. Type I**I** = miss true effect.

### Deep Dive

#### Type I Error (False Positive) — α
- **Ví dụ:** Smoke detector kêu nhưng không có lửa
- **Hậu quả:** Triển khai feature không thực sự tốt hơn; kê thuốc không hiệu quả
- **Kiểm soát:** Set α thấp (vd: 0.01 thay 0.05)

#### Type II Error (False Negative) — β
- **Ví dụ:** Có lửa thực sự nhưng alarm không kêu
- **Hậu quả:** Bỏ lỡ drug hiệu quả; bỏ lỡ tính năng sinh lời
- **Phụ thuộc vào:** Sample size, effect size, noise, alpha level

### Statistical Power = 1 − β

**Power** = Xác suất test phát hiện đúng effect khi effect thực sự tồn tại.

**Industry standard:** Power ≥ 80% (chấp nhận β ≤ 20%)

**Factors tăng Power:**

| Factor | Cách tăng |
|--------|---------|
| Sample size | **Tăng n** — đây là lever chính |
| Effect size | Lớn hơn → dễ detect hơn |
| Alpha | Tăng α → tăng Power (nhưng tăng Type I risk) |
| Noise | Giảm variance → tăng Power |

### Alpha-Beta Trade-off

**Cruel reality:** Không thể minimize cả hai errors mà không thay đổi sample size.

Giảm α (stricter) → β tăng (miss more true effects)  
Chỉ có tăng n mới giảm CÙNG LÚC cả hai.

### Real-World Context Matters

| Scenario | H₀ | H₁ | Error nào tệ hơn? | Chiến lược |
|---------|-----|-----|-----------------|-----------|
| Criminal Trial | Vô tội | Có tội | **Type I** (convict innocent) | α = 0.01, very strict |
| Medical Test | Khoẻ mạnh | Có bệnh | **Type II** (miss disease) | Power = 90%+ |
| Spam Filter | Legitimate email | Spam | **Type I** (block important email) | Strict về Type I |

---

---

## 9. One-Sample T-Test

> **Source:** https://www.tensortonic.com/ml-math/statistics/t-test

### Định nghĩa

Kiểm tra xem sample mean x̄ có **khác biệt đáng kể** với hypothesized population mean μ₀ không.

**Khác Z-test:** T-test dùng **sample std dev (s)** thay vì σ thực — phù hợp với thực tế vì σ gần như không bao giờ biết.

### Khi nào dùng T-Test này?

**CÓ:**
- Một biến liên tục (weight, height, pH)
- Một giá trị μ₀ cụ thể để so sánh (benchmark, tiêu chuẩn, historical)

**KHÔNG:**
- Hai groups riêng biệt → Independent Samples T-Test
- Cùng người đo hai lần → Paired Samples T-Test
- Ba groups trở lên → ANOVA
- Categorical data → Chi-Square

### Crucial Assumptions

1. **Independence:** Các observations độc lập nhau
2. **Continuous Data:** Interval/ratio scale
3. **Random Sampling:** Không bias chọn mẫu
4. **Normality:** Population xấp xỉ normal (hoặc n > 30 nhờ CLT)

### Hypothesis Setup

| Test Type | H₀ | H₁ |
|----------|-----|-----|
| Two-Tailed | μ = μ₀ | μ ≠ μ₀ |
| Right-Tailed | μ ≤ μ₀ | μ > μ₀ |
| Left-Tailed | μ ≥ μ₀ | μ < μ₀ |

### T-Statistic Formula

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} = \frac{\text{Signal}}{\text{Noise}}$$

- **Numerator (Signal):** Khoảng cách thực tế quan sát được
- **Denominator (Noise):** Standard Error — dao động tự nhiên của sample means

### Manual Calculation — Protein Bar Example

- Claim: bars có 20g protein. n=31, x̄=21.40g, s=2.54g, α=0.05
1. **Signal:** 21.40 − 20 = **1.40**
2. **SE:** 2.54/√31 ≈ **0.456**
3. **t-stat:** 1.40/0.456 ≈ **3.07**
4. **df:** 31−1 = 30
5. **Critical value (two-tailed, df=30):** t* = 2.042
6. 3.07 > 2.042 → **Reject H₀**
7. p-value = 0.0046 (chỉ 0.46% chance nếu label đúng)

### Normality Check

| Phương pháp | Cách check |
|------------|-----------|
| Histogram | Hình dạng chuông, không quá lệch |
| Q-Q Plot | Điểm theo đường chéo thẳng |

**Khi data không normal (n < 30):** Dùng **Wilcoxon Signed-Rank Test** (non-parametric).

---

---

## 10. A/B Testing

> **Source:** https://www.tensortonic.com/ml-math/statistics/ab-testing

### Nguyên lý cơ bản

**A/B Testing** = Scientific method để so sánh hai versions, loại bỏ confounding variables bằng **randomization**.

**"Fair Test" Rule:** Random assignment đảm bảo hai groups gần như identical trước test.

### Setting Hypotheses

- **H₀ (Skeptic):** Group A = Group B (không có effect)
- **H₁ (Discovery):** Group A ≠ Group B

### The Math: Two-Sample Z-Test

$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_A} + \frac{1}{n_B}\right)}}$$

- p̂_A, p̂_B: Conversion rates của A và B
- p̂: Pooled conversion rate
- |Z| > 1.96 → 95% confident difference không phải do may mắn

### Critical Z-Values

| Confidence | Critical Z |
|-----------|-----------|
| 90% | 1.645 |
| **95%** | **1.96** |
| 99% | 2.576 |

### Power Analysis & Sample Size

$$n = \frac{2\bar{p}(1-\bar{p})(Z_{\alpha/2} + Z_\beta)^2}{(p_B - p_A)^2}$$

**4 Levers của Power:**

| Lever | Effect |
|-------|--------|
| Baseline Rate | Higher baseline → easier to detect |
| MDE (Minimum Detectable Effect) | Nhỏ hơn → cần nhiều data HƠN rất nhiều |
| α (Significance Level) | α thấp hơn → cần nhiều data hơn |
| n (Sample Size) | **Lever chính** — bigger n = more power |

**Numerical Example:** Detect lift từ 5% lên 6% với 80% power → cần **~3,000/group** (6,000 total).

### Common Pitfalls

| Pitfall | Mô tả | Solution |
|--------|-------|---------|
| **Peeking (P-Hacking)** | Check results hằng ngày, stop khi p<0.05 → True α ≈ 26% | Pre-determine sample size và không peek |
| **Novelty Effect** | Users click new version chỉ vì mới | Run 1-2 full business cycles |
| **Simpson's Paradox** | Tốt overall nhưng tệ ở mọi segment | Always check segments |
| **Network Effects/Spillover** | Treatment users ảnh hưởng Control users | Cluster randomization |

### Case Study: Bulb Packaging

- Control (A): 5.2% (520/10,000)
- Treatment (B): 5.9% (590/10,000)
- Lift: +0.7% absolute, +13.5% relative
- P-value: 0.018 → **Significant!** → Roll out!

### ML Applications

| Application | Mô tả |
|-----------|-------|
| **Model Deployment** | A/B test new recommender vs old; offline RMSE ≠ business revenue |
| **Multi-Armed Bandits** | Thompson Sampling — shift traffic to winner during test (Exploration vs Exploitation) |
| **Hyperparameter Tuning** | Is learning rate 0.001 statistically better than 0.0001? |
| **Causal ML** | Khi A/B test không thể → Propensity Score Matching, IV |

---

---

## 11. ANOVA

> **Source:** https://www.tensortonic.com/ml-math/statistics/anova

### Tại sao cần ANOVA?

Khi có **3+ groups** để so sánh, đừng chạy nhiều T-tests → **Multiple Testing Problem**.

**Family-wise Error Rate:**
$$P(\text{Error}) = 1 - (0.95)^k$$

Với 10 groups (45 pairs) → Error rate > 90%! Guaranteed false positive.

### Core Idea: Signal vs Noise

ANOVA hỏi một câu hỏi global: **"Có bất kỳ difference nào giữa các groups không?"**

- **Between-Group Variance (SSB):** Các group means khác grand mean bao nhiêu? → **Signal**
- **Within-Group Variance (SSW):** Data bên trong mỗi group phân tán bao nhiêu? → **Noise**

### The F-Statistic

$$F = \frac{MS_{Between}}{MS_{Within}} = \frac{SSB / (k-1)}{SSW / (N-k)}$$

**Key Components:**

| Component | Công thức | Ý nghĩa |
|----------|----------|---------|
| SST (Total) | $\sum(x_{ij} - \bar{x}_{grand})^2$ | Total variance |
| SSB (Between) | $\sum n_i(\bar{x}_i - \bar{x}_{grand})^2$ | Variance do group differences |
| SSW (Within) | $\sum\sum(x_{ij} - \bar{x}_i)^2$ | Noise bên trong mỗi group |
| Identity | SST = SSB + SSW | |
| df_between | k − 1 | k = số groups |
| df_within | N − k | N = total samples |

### ANOVA Table Example (Bulb Supplier)

3 suppliers A, B, C; 20 bulbs/supplier; means: 1180, 1050, 1210 hrs; Grand mean: 1147

| Source | SS | df | MS | F |
|--------|----|----|-----|---|
| Between | 288,000 | 2 | 144,000 | **6.4** |
| Within | 1,282,500 | 57 | 22,500 | — |

F(2,57) = 6.4 > F_critical = 3.16 → **Reject H₀**: ít nhất một supplier khác biệt.

### Post-Hoc Tests (Sau ANOVA)

ANOVA chỉ nói "có ít nhất một group khác" — cần Post-Hoc để biết **cặp nào** khác nhau.

| Test | Đặc điểm |
|------|---------|
| **Tukey's HSD** | Controls FWER; most common; test all pairwise |
| **Bonferroni** | Đơn giản nhưng conservative; chia α cho số comparisons |
| **Scheffé** | Most conservative; cho phép complex contrasts |
| **Benjamini-Hochberg** | Controls FDR (không phải FWER); popular in genomics |

### Assumptions

| Assumption | Check | Violation Fix |
|-----------|-------|-------------|
| **Independence** | Study design | Repeated Measures ANOVA |
| **Normality** | Q-Q plot, Shapiro-Wilk | Kruskal-Wallis (non-parametric) khi n nhỏ |
| **Homoscedasticity** | Levene's Test | Welch's ANOVA |

### ANOVA Variants

| Variant | Khi nào dùng |
|--------|------------|
| One-Way ANOVA | 1 factor |
| Two-Way ANOVA | 2 factors; có thể test interaction effects |
| Repeated Measures ANOVA | Cùng subject đo nhiều lần |
| MANOVA | Multiple dependent variables |

### ML Applications

| Application | Cách dùng |
|-----------|---------|
| **Model Selection** | 5 architectures × 10 seeds → ANOVA phân biệt thật vs noise |
| **Feature Importance** | Categorical feature nhiều levels → ANOVA với target → F-stat cao = important |
| **Hyperparameter Validation** | LR=0.001 vs 0.01 → ANOVA trên CV scores |
| **SHAP/LIME Explainability** | Identify features gây variance lớn nhất |

---

---

## 12. Correlation & Covariance

> **Source:** https://www.tensortonic.com/ml-math/statistics/correlation

### Covariance (Hiệp phương sai)

$$Cov(X,Y) = \frac{1}{n-1}\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})$$

| Cov > 0 | X tăng, Y tăng (cùng chiều) |
|---------|---------------------------|
| Cov < 0 | X tăng, Y giảm (ngược chiều) |
| Cov ≈ 0 | Không có quan hệ tuyến tính |

**Vấn đề:** Covariance không có scale → meaningless khi so sánh.

### Correlation Coefficient (r) — Pearson

$$r = \frac{Cov(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i-\bar{X})^2} \sqrt{\sum(Y_i-\bar{Y})^2}}$$

Range: [-1, 1] — unitless, comparable

| r value | Diễn giải |
|---------|---------|
| r = 1.0 | Perfect positive |
| r = 0.7 | Strong positive |
| r = 0.3 | Weak positive |
| r = 0.0 | No linear relationship |
| r = −0.8 | Strong negative |

### Pearson vs Spearman

| Feature | Pearson (r) | Spearman (ρ) |
|---------|------------|--------------|
| Đo lường | Linear relationship | Monotonic relationship |
| Yêu cầu | Both approx normal | Không cần normal |
| Outlier sensitivity | **Sensitive** | **Robust** |
| Dữ liệu | Continuous | Ordinal/có outlier |
| Cách tính | Trực tiếp | Convert to ranks → Pearson |

### Correlation ≠ Causation — 4 Khả năng

| Type | Ví dụ |
|------|-------|
| A causes B | Mưa → đường ướt |
| B causes A (Reverse) | Crime cao → thuê nhiều police (không phải ngược lại) |
| C causes cả A và B (Confounding) | Mùa hè → ice cream + đuối nước |
| Pure Coincidence (Spurious) | Nicolas Cage movies ↔ pool drownings (r=0.66!) |

### ML Applications

| Application | Vai trò |
|-----------|--------|
| **Feature Selection** | Correlation matrix: feature-target cao ↑, feature-feature thấp ↓ |
| **Multicollinearity** | Perfect correlation → X^T X singular → không invertible |
| **VIF** | $VIF = 1/(1-R^2_i)$; VIF > 10 → red flag |
| **PCA** | Transform correlated features → uncorrelated principal components |

---

---

## 13. Maximum Likelihood Estimation (MLE)

> **Source:** https://www.tensortonic.com/ml-math/statistics/maximum-likelihood-estimation

### Core Question

"Given observed data, which parameter values would have made this data most probable?"

### Likelihood vs Probability

| Concept | Fixed | Variable | Question |
|---------|-------|---------|---------|
| **Probability** P(x\|θ) | Parameter θ | Data x | Cho θ, xác suất của data này? |
| **Likelihood** L(θ\|x) | Data x | Parameter θ | Cho data này, θ nào "compatible" nhất? |

> ⚠️ Likelihood **KHÔNG** phải probability distribution over θ. Nó không sum to 1.

### MLE Formula

$$\hat{\theta}_{MLE} = \arg\max_{\theta} P(D|\theta)$$

### Log-Likelihood Trick

**Vấn đề:** L(θ) = product của nhiều số nhỏ → **numerical underflow** (10⁻²⁰⁰ ≈ 0).

**Giải pháp:** Log transform — monotonically increasing → same argmax.

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)$$

Products → Sums → dễ differentiate hơn nhiều.

### Worked Examples

**Biased Coin (k heads in n flips):**
1. L(p) = p^k (1-p)^(n-k)
2. ℓ(p) = k ln(p) + (n-k) ln(1-p)
3. dℓ/dp = k/p − (n-k)/(1-p) = 0
4. **p̂_MLE = k/n** (sample proportion — intuitive!)

**Normal Distribution (estimate μ):**
$$\hat{\mu}_{MLE} = \frac{1}{n}\sum x_i = \bar{x}$$

MLE cho mean = sample mean! Nhất quán với intuition.

### Key Properties của MLE

| Property | Mô tả |
|---------|-------|
| **Consistency** | n → ∞ → estimate → true parameter |
| **Efficiency** | Variance thấp nhất trong các unbiased estimators |
| **Invariance** | MLE của g(θ) = g(MLE của θ) |
| **Asymptotic Normality** | Large n → MLE ~ Normal → dễ tính CI |

### MLE chính là Loss Functions trong ML! 🔑

| Loss Function | Tương đương MLE với |
|-------------|-------------------|
| **MSE** | Gaussian noise assumption |
| **Binary Cross-Entropy** | Bernoulli distribution |
| **Categorical Cross-Entropy** | Categorical distribution |

$$\text{MSE} \equiv -\ell(\mu) \propto \sum (x_i - \mu)^2$$
$$\text{BCE} \equiv -\ell(p) = -[y\ln(p) + (1-y)\ln(1-p)]$$

> 💡 Khi train neural network bằng cross-entropy hoặc MSE, bạn đang thực hiện **MLE**.

### Limitation: Zero Count Problem

Flip coin 3 lần, được 3H → MLE: P(heads) = 1.0 → Tails là **impossible** → Sai!

**Fix: MAP (Maximum A Posteriori)**
$$\hat{\theta}_{MAP} = \arg\max_{\theta} \underbrace{L(\theta)}_{\text{Likelihood}} \cdot \underbrace{P(\theta)}_{\text{Prior}}$$

- L2 regularization = Gaussian prior trên weights
- L1 regularization = Laplace prior trên weights

---

---

## 14. Bayesian vs Frequentist Statistics

> **Source:** https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist

### Core Philosophies

| | Frequentist | Bayesian |
|--|------------|---------|
| **Data** | Random | Fixed |
| **Parameters** | Fixed (unknown constants) | Random (probability distributions) |
| **Probability** = | Long-run frequency | Degree of belief |
| **Tools** | P-values, CI, MLE | Bayes Theorem, Prior, Posterior, Credible Intervals |

### Intuition: Lost Phone Example

- **Frequentist:** "Tiếng beep từ bếp → 90% phone ở bếp." (Chỉ dùng current data)
- **Bayesian:** "Tiếng beep từ bếp, NHƯNG tôi để phone ở phòng ngủ 99% thời gian → vẫn nên check phòng ngủ trước." (Kết hợp data + prior knowledge)

### Mathematical Difference

$$\text{Frequentist MLE: } \hat{\theta} = \arg\max_{\theta} P(D|\theta)$$

$$\text{Bayesian MAP: } \hat{\theta} = \arg\max_{\theta} P(\theta|D) \propto P(D|\theta) \cdot P(\theta)$$

> Key: Nếu Prior P(θ) là uniform (flat) → MAP = MLE

### Confidence Intervals vs Credible Intervals

| | 95% Confidence Interval | 95% Credible Interval |
|--|------------------------|----------------------|
| **Câu hỏi trả lời** | Về procedure (long-run) | Về parameter (direct) |
| **Diễn giải** | "95% các intervals xây dựng sẽ chứa true value" | "95% xác suất true parameter nằm trong interval này" |
| **Tham số** | Fixed (cá không di chuyển) | Random (bất định) |
| **Intuitive?** | ❌ Counter-intuitive | ✅ Đây là điều mọi người thực sự muốn! |

### A/B Testing Comparison

| | Frequentist A/B | Bayesian A/B |
|--|---------------|-------------|
| **Sample size** | Pre-register trước | Không cần pre-register |
| **Output** | Binary: Significant / Not | Continuous: "92% B > A" |
| **Early stopping** | ❌ Không được peek | ✅ Có thể stop khi confident |
| **Interpretation** | P-value | Probability of being better |

### ML Applications (Bayesian)

| Application | Giải thích |
|-----------|---------|
| **L2 Regularization = Gaussian Prior** | Ridge regression = Bayesian MAP với Gaussian prior |
| **L1 Regularization = Laplace Prior** | Lasso = MAP với Laplace prior |
| **Dropout = Approximate Bayesian Inference** | MC Dropout → uncertainty estimates |
| **Gaussian Processes** | Fully Bayesian; uncertainty quantification out of the box |
| **Thompson Sampling** | Bayesian bandit; sample from posterior → pick best arm |

### Khi nào dùng cái nào?

| Scenario | Recommendation | Lý do |
|---------|---------------|-------|
| Dataset lớn, cần objectivity | **Frequentist** | Prior không quan trọng khi n lớn; simpler |
| Dataset nhỏ, có domain knowledge | **Bayesian** | Prior ổn định estimates, giảm overfitting |
| Academic publication | **Frequentist** | P-values là standard |
| Real-time decision making | **Bayesian** | Update liên tục; không cần fixed sample size |
| Cần uncertainty quantification | **Bayesian** | Natural probability distributions over predictions |

---

---

## 15. Resampling Methods: Bootstrap & Jackknife

> **Source:** https://www.tensortonic.com/ml-math/statistics/resampling

### Tại sao Resampling?

Traditional statistics cần biết underlying distribution. **Resampling** không cần giả định — dùng computing power thay thế cho mathematical formulas. Ideal khi không có closed-form formula (vd: median's SE).

### Bootstrap Algorithm

Developed by Bradley Efron (Stanford, 1979).

**Ý tưởng:** Treat your sample AS IF it were the population.

```
1. Original sample S, size n
2. Draw S* of size n from S WITH REPLACEMENT
3. Calculate statistic (mean, median, 90th percentile) on S*
4. Repeat 10,000+ times
5. Distribution of 10,000 statistics ≈ True Sampling Distribution
```

**Tại sao "with replacement"?** Without replacement → mọi bootstrap sample = original sample (chỉ shuffle). With replacement → mỗi sample là random subset khác → mimics sampling from true population.

> ~63.2% original points xuất hiện trong mỗi bootstrap sample.

### Bootstrap vs Jackknife

| Feature | Bootstrap | Jackknife |
|---------|----------|----------|
| Sampling | Random with replacement | Leave-one-out (deterministic) |
| Number of samples | Vô hạn | Chỉ n samples |
| Works for | Any statistic | Mainly bias estimation |
| Non-smooth stats (median) | ✅ | ❌ Fails |
| Preferred today | ✅ | Mostly legacy |

**Jackknife formula:**
$$\hat{\theta}_{Jack} = n\hat{\theta} - \frac{n-1}{n}\sum_{i=1}^{n} \hat{\theta}_{(i)}$$

### Bootstrap Confidence Intervals

**Percentile Method:** Lấy 2.5th và 97.5th percentile của 10,000 bootstrap statistics = **95% CI**. Works for any statistic!

**BCa (Bias-Corrected and Accelerated):** Điều chỉnh bias và skewness; more accurate; standard in statistical software.

### Case Study: Bulb Lifespan Median

- 30 bulbs, median = 1,200 hours. CI?
1. Resample 30 values with replacement
2. Calculate median
3. Repeat 10,000 times
4. Take 250th (2.5%) và 9750th (97.5%) values
- **95% CI: [1,100 hrs, 1,320 hrs]**

### Permutation Tests

Để hypothesis testing (không phải CI):

```
1. H₀ true → group labels arbitrary
2. Randomly shuffle labels 10,000 times
3. Each shuffle: calculate difference in means
4. P-value = % shuffles với difference ≥ observed difference
```

### ML Applications

| Application | Mô tả |
|-----------|-------|
| **Bagging (Bootstrap Aggregating)** | 100 bootstrap samples → 100 Decision Trees → average/vote. Reduces Variance! |
| **Random Forest** | Bagging + random feature selection at each split |
| **Out-of-Bag (OOB) Error** | ~37% data không được thấy bởi mỗi tree → dùng làm free validation set |
| **Model Uncertainty Estimation** | Train model trên 100 bootstrap samples → variance của predictions = uncertainty |

---

---

## 🗺️ KNOWLEDGE MAP — Mối liên hệ giữa các chủ đề

```
Population vs Sample
    │
    ├──► Sampling Distributions  ──────────────────────►  CLT
    │         │                                              │
    │         ▼                                              ▼
    │    Standard Error                              n ≥ 30 Rule
    │         │
    │         ├──► Confidence Intervals  ◄───────────── Z vs T
    │         │         │
    │         └──► Hypothesis Testing
    │                   │
    │          ┌────────┴──────────┐
    │          ▼                   ▼
    │       P-Values         Type I & II Errors
    │          │                   │
    │          └──────┬────────────┘
    │                 ▼
    │           One-Sample T-Test
    │                 │
    │           Two-Sample → A/B Testing
    │                 │
    │           3+ Groups → ANOVA
    │
    ├──► Descriptive Statistics  ──►  Correlation & Covariance
    │
    └──► MLE  ──────────────────────► Loss Functions (MSE, BCE)
              │
              ▼
         Bayesian vs Frequentist
              │
              ▼
         Resampling (Bootstrap → Bagging → Random Forest)
```

---

## 🔑 QUICK REFERENCE — Công thức quan trọng nhất

| Concept | Formula |
|---------|---------|
| Standard Error | $SE = \sigma/\sqrt{n}$ |
| Z-statistic | $Z = (\bar{x} - \mu) / (\sigma/\sqrt{n})$ |
| T-statistic | $t = (\bar{x} - \mu_0) / (s/\sqrt{n})$, df = n-1 |
| CI (Z) | $\bar{x} \pm z^* \cdot \sigma/\sqrt{n}$ |
| CI (T) | $\bar{x} \pm t^* \cdot s/\sqrt{n}$ |
| F-statistic | $F = MSB/MSW = [SSB/(k-1)] / [SSW/(N-k)]$ |
| Pearson r | $r = Cov(X,Y) / (\sigma_X \sigma_Y)$ |
| Log-Likelihood | $\ell(\theta) = \sum_i \ln f(x_i; \theta)$ |
| Bayesian Update | $P(\theta\|D) \propto P(D\|\theta) \cdot P(\theta)$ |
| Bootstrap CI | 2.5th – 97.5th percentile của 10,000 bootstrap stats |
| Bonferroni | $\alpha_{corrected} = \alpha / n_{tests}$ |
| Bessel's Correction | Divide by $n-1$ for sample variance |
| Sample Size (A/B) | $n = 2\bar{p}(1-\bar{p})(Z_{\alpha/2}+Z_\beta)^2 / (p_B-p_A)^2$ |

---

## 🏭 UNIFIED EXAMPLE — Light Bulb Factory (xuyên suốt các chủ đề)

| Chủ đề | Áp dụng |
|--------|--------|
| Population vs Sample | Test 1,000/1M bulbs → sample estimate |
| Descriptive Stats | Mean=1045h, Std=150h, IQR |
| Sampling Distribution | SE = 150/√1000 ≈ 4.7h |
| CLT | Sample mean approximately Normal (n>30) |
| Confidence Interval | "Mean lifespan: 1045 ± 9.3h (95% CI)" |
| Hypothesis Testing | "Nhà cung cấp có thực sự deliver ≥1000h?" |
| P-Value | p=0.018 → significant evidence |
| Type I/II Error | Type I: reject good supplier; Type II: keep bad supplier |
| T-Test | Small sample warranty test |
| A/B Testing | New vs old packaging → sales rate |
| ANOVA | 3 filament suppliers → F-test |
| Correlation | Voltage vs lifespan: r=0.72 |
| MLE | Estimate failure distribution parameters |
| Bayesian | Prior belief about failure rate + new data |
| Bootstrap | 95% CI for median lifespan (no formula needed) |
