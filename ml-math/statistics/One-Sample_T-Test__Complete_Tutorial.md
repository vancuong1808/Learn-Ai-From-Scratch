**Title:** One-Sample T-Test: Complete Tutorial

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/t-test](https://www.tensortonic.com/ml-math/statistics/t-test)

---

# Page Structure Map
```text
One-Sample T-Test: Complete Tutorial
├── Introduction
│   └── Real-World Scenarios
├── When to Use This Test
│   ├── You HAVE
│   └── You Do NOT Have
├── Crucial Assumptions
│   ├── 1\. Independence
│   ├── 2\. Continuous Data
│   ├── 3\. Random Sampling
│   └── 4\. Normality
├── Setting up Hypotheses
├── The T-Statistic Formula
│   ├── The Signal (Numerator)
│   ├── The Noise (Denominator)
│   └── Prerequisite
├── Manual Calculation Example
│   └── The Scenario
│       ├── Step 1: Calculate the Difference (Signal)
│       ├── Step 2: Calculate Standard Error (Noise)
│       ├── Step 3: Calculate t-Statistic
│       └── Step 4: Determine Critical Value & Decision
├── Interactive Demo: T-Test Calculator
│   └── Interactive T-Test Calculator
│       └── Step-by-Step Calculation
├── Interpreting Results (P-Value)
│   └── The [P-Value](https://www.tensortonic.com/ml-math/statistics/p-value) Logic
├── Checking Normality
│   ├── 1\. Histogram
│   ├── 2\. Q-Q Plot (Quantile-Quantile)
│   └── What if data is not normal?
└── Interactive Demo: Normality Diagnostics
    └── Normality Check: Visual Diagnostics
        ├── Histogram
        ├── Q-Q Plot
        └── Assessment
```

---

## Introduction

The **One-Sample T-Test** is a parametric statistical procedure used to determine whether the mean of a single sample (xˉ\\bar{x}) is statistically different from a known or hypothesized population mean (μ0\\mu\_0).

This test bridges the gap between your specific observations and the "truth" you are comparing them against. Unlike the Z-test, which requires knowing the true population standard deviation (σ\\sigma), the T-test is robust because it uses the sample standard deviation (ss) as an estimate. This makes it the standard tool for real-world analysis where population parameters are almost never known.

### Real-World Scenarios

-   **Quality Control:** A factory claims its protein bars contain 20g of protein. You collect a random [sample](https://www.tensortonic.com/ml-math/statistics/population-sample) of 31 bars. Is the average really 20g, or are they systematically underfilling (or overfilling)?
-   **Healthcare:** A hospital wants to know if the average cholesterol level of a specific patient group differs from the national health goal of 200 mg/dL.
-   **Environmental Safety:** An inspector needs to verify if the lead levels in an apartment building exceed the EPA safety clearance level of 10 micrograms per square foot.

## When to Use This Test

You should use a One-Sample T-Test when you meet specific criteria. It is not a catch-all for every mean comparison problem.

#### You HAVE

-   A single continuous variable (e.g., weight, time, height, pH level).
-   A known "Test Value" or hypothesized mean (μ0\\mu\_0) to compare against. This could be a legal standard, an industry benchmark, or a historical average.

#### You Do NOT Have

-   Two separate groups of people (Use Independent Samples T-Test).
-   The same people measured twice (Use Paired Samples T-Test).
-   More than two groups (Use [ANOVA](https://www.tensortonic.com/ml-math/statistics/anova)).
-   Categorical data (Use Chi-Square).

## Crucial Assumptions

For the results of a t-test to be valid, your data must adhere to four key assumptions. Violating these can lead to incorrect conclusions (false positives or false negatives).

#### 1\. Independence

The value of one observation does not influence another. For example, measuring the weight of 30 _different_ people is independent. Measuring the weight of the _same_ person 30 times is dependent (not allowed).

#### 2\. Continuous Data

The variable must be interval or ratio level (e.g., grams, seconds, height). Categorical variables (e.g., "Yes/No", "High/Low") cannot be used here.

#### 3\. Random Sampling

The data should be obtained via a simple random sample from the population to avoid selection bias. If you only pick the heaviest protein bars, your result will be biased high.

#### 4\. Normality

The population from which the sample is drawn should be approximately [normally distributed](https://www.tensortonic.com/ml-math/probability/distributions).  
Note: The t-test is "robust" to violations of normality if the sample size is large (n\>30n > 30) thanks to the [Central Limit Theorem](https://www.tensortonic.com/ml-math/statistics/central-limit-theorem). For small samples (n<30n < 30), normality is critical.

## Setting up Hypotheses

Before calculating anything, you must define your null (H0H\_0) and alternative (H1H\_1 or HaH\_a) hypotheses. This dictates whether you perform a one-tailed or two-tailed test.

| Test Type | Question Asked | Null Hypothesis (H0H\_0) | Alternative Hypothesis (H1H\_1) |
| --- | --- | --- | --- |
| Two-Tailed | Is the mean _different_ from μ0\\mu\_0? | μ\=μ0\\mu = \\mu\_0 | μ≠μ0\\mu \\neq \\mu\_0 |
| Right-Tailed | Is the mean _greater_ than μ0\\mu\_0? | μ≤μ0\\mu \\le \\mu\_0 | μ\>μ0\\mu > \\mu\_0 |
| Left-Tailed | Is the mean _less_ than μ0\\mu\_0? | μ≥μ0\\mu \\ge \\mu\_0 | μ<μ0\\mu < \\mu\_0 |

Example: For the protein bars (Label = 20g), we usually care if it's different in \*either\* direction, so we use a Two-Tailed test. H0:μ\=20H\_0: \\mu = 20, H1:μ≠20H\_1: \\mu \\neq 20.

## The T-Statistic Formula

The t-statistic essentially measures "how many standard errors away from the hypothetical mean is our sample mean?".

t\=xˉ−μ0s/nt = \\frac{\\bar{x} - \\mu\_0}{s / \\sqrt{n}}

##### The Signal (Numerator)

xˉ−μ0\\bar{x} - \\mu\_0

The actual difference between your sample mean and the hypothesized population mean.

##### The Noise (Denominator)

SE\=snSE = \\frac{s}{\\sqrt{n}}

The [Standard Error](https://www.tensortonic.com/ml-math/statistics/sampling-distribution). This estimates how much sample means fluctuate naturally.

#### Prerequisite

If the concepts of Standard Error or T-distribution feel unfamiliar, review the [Sampling Distributions chapter](https://www.tensortonic.com/ml-math/statistics/sampling-distribution) first. It includes interactive visualizations that build intuition for why we use tt instead of zz.

## Manual Calculation Example

Let us walk through the calculation manually, step-by-step, using the **Protein Bar** example.

#### The Scenario

-   **Claim (Hypothesis):** Protein bars have 20g of protein.
-   **Sample Data:** We collected n\=31n = 31 bars.
-   **Sample Mean (xˉ\\bar{x}):** Calculated average is 21.40g.
-   **Sample Std Dev (ss):** Calculated standard deviation is 2.54g.
-   **Significance Level (α\\alpha):** 0.05 (5% risk).

##### Step 1: Calculate the Difference (Signal)

Difference\=xˉ−μ0\=21.40−20\=1.40\\text{Difference} = \\bar{x} - \\mu\_0 = 21.40 - 20 = 1.40

##### Step 2: Calculate Standard Error (Noise)

SE\=sn\=2.5431≈2.545.568≈0.456SE = \\frac{s}{\\sqrt{n}} = \\frac{2.54}{\\sqrt{31}} \\approx \\frac{2.54}{5.568} \\approx 0.456

##### Step 3: Calculate t-Statistic

t\=SignalNoise\=1.400.456≈3.07t = \\frac{\\text{Signal}}{\\text{Noise}} = \\frac{1.40}{0.456} \\approx 3.07

Our sample mean is 3.07 standard errors away from the hypothesized mean.

##### Step 4: Determine Critical Value & Decision

**Degrees of Freedom (df):** n−1\=31−1\=30n - 1 = 31 - 1 = 30

Look up t0.05,30t\_{0.05, 30} in a t-table (two-tailed). The critical value is approximately **2.042**.

Comparison:

∣tcalc∣\=3.07|t\_{calc}| = 3.07 vs tcrit\=2.042t\_{crit} = 2.042

Since 3.07\>2.0423.07 > 2.042, our result is in the "rejection region."

Conclusion: Reject H0H\_0. The protein bars likely do NOT contain 20g of protein (they contain significantly more).

## Interactive Demo: T-Test Calculator

Adjust the sample statistics and see the t-statistic calculation and decision in real-time. Watch how the t-value changes position on the distribution curve.

### Interactive T-Test Calculator

Adjust your sample statistics and see how the t-statistic is calculated and whether you reject the null hypothesis.

Sample Mean (x̄): 21.40

Hypothesized Mean (μ0): 20.0

Sample SD (s): 2.54

Sample Size (n): 31

Significance Level (alpha): 0.05

#### Step-by-Step Calculation

1\. Difference (Signal):21.40 - 20.0 = 1.40

2\. Standard Error (Noise):2.54 / √31 = 0.456

3\. T-Statistic:t = 3.069

4\. Degrees of Freedom:df = 30

5\. Critical Value (two-tailed):t\* = ±2.042

T-Statistic

3.069

Critical Value

±2.042

Decision

Reject H0

**Interpretation:** The t-statistic (3.07) falls in the red rejection region. The sample mean is significantly different from the hypothesized mean at the 5% level.

## Interpreting Results (P-Value)

In modern analysis (using software like JMP, SPSS, or Python), we rely more on the **P-Value** than manually comparing t-statistics to critical values.

#### The [P-Value](https://www.tensortonic.com/ml-math/statistics/p-value) Logic

The P-value answers: _"If the true mean was actually 20g, what is the probability of randomly picking a sample with a mean of 21.4g (or more extreme)?"_

P-Value <α< \\alpha(e.g., 0.0046<0.050.0046 < 0.05). Highly unlikely to happen by chance. **Reject H0H\_0.**

P-Value \>α\> \\alpha(e.g., 0.23\>0.050.23 > 0.05). Could easily happen by chance. **Fail to Reject H0H\_0.**

For our energy bar example, software calculates a p-value of **0.0046**. This means there is only a 0.46% chance of seeing this data if the labels were correct. That is strong evidence that the labels are wrong.

## Checking Normality

Before trusting your t-test, you should visualize your data to check if the normality assumption holds.

#### 1\. Histogram

Look for a bell shape. It does not need to be perfect, but it should not be heavily skewed or have massive outliers.

"Is the distribution roughly symmetric and mound-shaped?"

#### 2\. Q-Q Plot (Quantile-Quantile)

Points should roughly follow a straight diagonal line. If they curve significantly, the data might not be normal.

"Do the sample quantiles match theoretical normal quantiles?"

#### What if data is not normal?

If sample size is small (n<30n < 30) and data is skewed, the t-test is invalid. You should use a **Non-Parametric test** like the **Wilcoxon Signed-Rank Test** instead.

## Interactive Demo: Normality Diagnostics

Explore how different data distributions appear in histogram and Q-Q plots. Learn to visually diagnose normality violations.

### Normality Check: Visual Diagnostics

Select different data distributions to see how histogram and Q-Q plots help diagnose normality violations.

#### Histogram

Bell-shaped distribution ✓

#### Q-Q Plot

Points follow diagonal line ✓

✓

#### Assessment

Data appears approximately normal. T-test is valid.

**How to check:** Look for a bell shape in the histogram and points following the diagonal blue line in the Q-Q plot. If n > 30, the t-test is robust to mild violations due to the Central Limit Theorem.