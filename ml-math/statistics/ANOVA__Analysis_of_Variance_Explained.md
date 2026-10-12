**Title:** ANOVA: Analysis of Variance Explained

**Author:** TensorTonic

**Source:** [https://www.tensortonic.com/ml-math/statistics/anova](https://www.tensortonic.com/ml-math/statistics/anova)

---

# Page Structure Map
```text
ANOVA: Analysis of Variance Explained
├── Beyond Two Groups
├── The Multiple Testing Problem
│   ├── The Math of Failure
│   └── Light Bulb Factory Example
├── Intuition: Signal vs Noise
│   ├── Between-Group Variance (SSB)
│   └── Within-Group Variance (SSW)
├── The Math (F-Statistic)
├── Case Study: Bulb Filament Comparison
│   ├── The Scenario
│   ├── The Data
│   └── The ANOVA Table
├── Interactive Simulator
│   ├── Signal (Between Groups)
│   ├── Noise (Within Groups)
│   └── F-Statistic
├── Post-Hoc Tests
│   ├── Tukey's HSD
│   ├── Bonferroni Correction
│   ├── Scheffé's Method
│   └── Benjamini-Hochberg (FDR)
├── Assumptions Checklist
│   ├── 1\. Independence
│   ├── 2\. Normality
│   └── 3\. Homogeneity of Variance (Homoscedasticity)
├── ANOVA Variants
│   ├── One-Way ANOVA
│   ├── Two-Way ANOVA
│   ├── Repeated Measures ANOVA
│   └── MANOVA
└── ML Applications
    ├── Model Selection
    ├── Feature Importance (Categorical)
    ├── Hyperparameter Tuning Validation
    └── Explainability (SHAP/LIME)
```

---

## Beyond Two Groups

A/B testing (T-test) is great for comparing two things (Control vs Variant). But what if you have 3 landing pages? Or 4 different ML models? Or 5 different light bulb filament types?

You might be tempted to just run T-tests for every pair: A vs B, B vs C, A vs C. **Do not do this.** This is a classic statistical trap that will lead you to false discoveries.

## The Multiple Testing Problem

### The Math of Failure

If you set α\=0.05\\alpha = 0.05, you have a 5% chance of a False Positive (finding a difference when none exists) for _one_ test.

If you run 3 pairwise tests (A-B, B-C, A-C), the probability of making _at least one_ False Positive explodes:

P(Error)\=1−(0.95)3≈14.3%P(\\text{Error}) = 1 - (0.95)^3 \\approx 14.3\\%

With 10 groups (45 pairs), your error rate is over 90%! You are guaranteed to find "significance" purely by chance. This is called the **Family-wise Error Rate (FWER)**.

### Light Bulb Factory Example

Your factory uses 4 different filament suppliers. You want to know if bulb lifespan differs by supplier. Running 6 pairwise T-tests (A-B, A-C, A-D, B-C, B-D, C-D) gives you a 26% chance of declaring one supplier "better" even if they're all identical. Bad for quality control!

## Intuition: Signal vs Noise

ANOVA solves this by asking a single global question: **"Is there ANY difference among these groups?"**

It does this by comparing two types of variance. The core insight is that if the groups are truly different, the means should be more spread out than you'd expect from random noise.

### Between-Group Variance (SSB)

How different are the group means from the grand mean? This is the **Signal**.

Example: Average lifespan of bulbs from Supplier A is 1200 hours, B is 1100, C is 1300. The grand mean is 1200. SSB measures this spread.

### Within-Group Variance (SSW)

How spread out is the data inside each group? This is the **Noise** (random error).

Example: Bulbs from Supplier A range from 1000-1400 hours. This internal variation is the baseline noise level.

## The Math (F-Statistic)

We calculate the F-ratio. If Signal > Noise, F will be large. The [F-distribution](https://www.tensortonic.com/ml-math/probability/distributions) tells us how unlikely a given F-value is under the null hypothesis.

F\=MSBetweenMSWithin\=SSB/(k−1)SSW/(N−k)F = \\frac{\\text{MS}\_{Between}}{\\text{MS}\_{Within}} = \\frac{\\text{SSB} / (k-1)}{\\text{SSW} / (N-k)}

**1\. Total Sum of Squares (SST):** ∑(xij−xˉgrand)2\\sum (x\_{ij} - \\bar{x}\_{grand})^2 - Total variance in the dataset.

**2\. Sum of Squares Between (SSB):** ∑ni(xˉi−xˉgrand)2\\sum n\_i(\\bar{x}\_i - \\bar{x}\_{grand})^2 - Variance explained by group differences.

**3\. Sum of Squares Within (SSW):** ∑∑(xij−xˉi)2\\sum \\sum (x\_{ij} - \\bar{x}\_i)^2 - Variance within each group (noise).

**Key Identity:** SST\=SSB+SSW\\text{SST} = \\text{SSB} + \\text{SSW}

**4\. Degrees of Freedom:** dfbetween\=k−1df\_{between} = k-1 (k = number of groups), dfwithin\=N−kdf\_{within} = N-k (N = total samples)

**5\. Mean Squares:** MSB\=SSB/(k−1)\\text{MSB} = \\text{SSB}/(k-1), MSW\=SSW/(N−k)\\text{MSW} = \\text{SSW}/(N-k)

## Case Study: Bulb Filament Comparison

#### The Scenario

Your light bulb company sources tungsten filaments from 3 different suppliers (A, B, C). Quality control tests 20 bulbs from each supplier by running them until failure. You want to know: Do the suppliers differ in average bulb lifespan?

#### The Data

Supplier A

1,180 hrs

Mean (n=20)

Supplier B

1,050 hrs

Mean (n=20)

Supplier C

1,210 hrs

Mean (n=20)

Grand Mean: 1,147 hrs. Pooled Std Dev: 150 hrs.

#### The ANOVA Table

| Source | SS | df | MS | F |
| --- | --- | --- | --- | --- |
| Between | 288,000 | 2 | 144,000 | 6.4 |
| Within | 1,282,500 | 57 | 22,500 | \- |

**F(2, 57) = 6.4**. Critical F at α\=0.05\\alpha = 0.05 is 3.16. Since 6.4 > 3.16, we **reject the null**. At least one supplier is different. Time for Post-Hoc tests!

## Interactive Simulator

Drag the group means apart (increase Signal) or increase the spread (increase Noise). Watch how the F-statistic reacts. When F crosses the critical threshold, the result becomes significant.

Group A Mean

Group B Mean

Group C Mean

Within-Group Variance (Noise)

#### Signal (Between Groups)

8000.0

Variance of group means

#### Noise (Within Groups)

100.0

Average variance inside groups

#### F-Statistic

80.00

Significant Difference!

## Post-Hoc Tests

ANOVA only tells you **"At least one group is different."** It doesn't tell you _which_ one. This is where Post-Hoc (Latin for "after this") tests come in.

### Tukey's HSD

(Honestly Significant Difference). Controls FWER. Tests all pairwise differences against a single critical value. Most common choice.

### Bonferroni Correction

Simple but conservative. Divide α\\alpha by the number of comparisons. For 3 groups: use α/3\=0.0167\\alpha/3 = 0.0167 for each T-test.

### Scheffé's Method

Most conservative. Allows testing any complex contrasts (e.g., is A different from the average of B and C?).

### Benjamini-Hochberg (FDR)

Controls False Discovery Rate instead of FWER. Less conservative. Popular in genomics where you have thousands of tests.

**Bulb Example Continued:** Tukey's HSD reveals Supplier B is significantly worse than both A and C (p < 0.05), but A and C are not significantly different (p = 0.32). Decision: Consider dropping Supplier B.

## Assumptions Checklist

ANOVA is robust to minor violations, but it can fail badly if these assumptions are grossly violated:

#### 1\. Independence

Samples must be independent. One person can't be in Group A and Group B. No repeated measures on the same subject.

**If violated:** Use Repeated Measures ANOVA.

#### 2\. Normality

Residuals should be normally distributed. Check with Q-Q plot or Shapiro-Wilk test.

**If violated:** ANOVA is robust with large samples (n > 30 per group). For small samples, use Kruskal-Wallis (non-parametric).

#### 3\. Homogeneity of Variance (Homoscedasticity)

All groups should have roughly the same variance. Check with Levene's Test.

**If violated:** Use Welch's ANOVA, which doesn't assume equal variances.

## ANOVA Variants

### One-Way ANOVA

One factor (e.g., Supplier). This is what we covered above.

### Two-Way ANOVA

Two factors (e.g., Supplier AND Bulb Wattage). Can test main effects and **interaction effects** (does the supplier effect differ by wattage?).

### Repeated Measures ANOVA

When the same subject is measured multiple times (e.g., bulb brightness at 0, 500, 1000 hours). Accounts for within-subject correlation.

### MANOVA

Multivariate ANOVA. Multiple dependent variables (e.g., test bulb lifespan AND brightness AND color temperature simultaneously).

## ML Applications

### Model Selection

You train 5 architectures (ResNet, VGG, etc.) on 10 random seeds. ANOVA tells you if there's a statistically significant difference between the architectures, not just noise from random seeds.

### Feature Importance (Categorical)

For categorical features with many levels (e.g., "City" has 50 values), ANOVA can tell you if the target variable (e.g., "House Price") varies significantly across these cities. High F-stat = important feature.

### Hyperparameter Tuning Validation

Is learning rate 0.001 really better than 0.01? Run ANOVA on cross-validation scores across multiple learning rates to check if the difference is significant.

### Explainability (SHAP/LIME)

When explaining model predictions, ANOVA can help identify which features contribute most to variance in predictions across different data segments.