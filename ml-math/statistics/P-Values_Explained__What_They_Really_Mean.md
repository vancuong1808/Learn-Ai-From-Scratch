**Title:** P-Values Explained: What They Really Mean

**Author:** TensorTonic

**Source:** [https://www.tensortonic.com/ml-math/statistics/p-value](https://www.tensortonic.com/ml-math/statistics/p-value)

---

# Page Structure Map
```text
P-Values Explained: What They Really Mean
├── The Intuition: How "Surprised" Are You?
│   └── The Coin Toss Analogy
│       └── Key Insight
├── The Formal Definition
│   ├── What it IS
│   └── What it is NOT
├── Interactive Demo: The Tail Area
│   └── P-Value: The Tail Area
├── The P-Value Fallacy (ASA Statement)
│   ├── "P\=0.05P = 0.05 means there is a 95% chance the hypothesis is true."
│   ├── "P\>0.05P > 0.05 means no effect."
│   └── "P\=0.04P = 0.04 is significantly better than P\=0.06P = 0.06."
├── How to Interpret P-Values
├── ML Application: Feature Selection
│   └── Backward Elimination Algorithm
├── Pitfalls: P-Hacking & Corrections
│   ├── The Multiple Comparisons Problem
│   └── The Fix: Bonferroni Correction
└── Interactive Demo: P-Hacking Simulation
    └── P-Hacking Simulation
```

---

**Context:** This chapter dives deep into P-values. If you need a refresher on Null/Alternative hypotheses and the overall testing framework, see theHypothesis Testing chapter first.

## The Intuition: How "Surprised" Are You?

Forget the math for a second. Think of the p-value as a **Surprise Meter**. It measures how weird your observed data would be if the Null Hypothesis were actually true.

### The Coin Toss Analogy

Your friend gives you a coin. The **Null Hypothesis (H0H\_0)** is that the coin is fair. You flip it 10 times:

Scenario A: 5 Heads, 5 Tails

Are you surprised? **No.** This is exactly what you expect.

P≈1.0P \\approx 1.0

Not surprised

Scenario B: 9 Heads, 1 Tail

Are you surprised? **Yes, a little.** Rare but possible.

P≈0.02P \\approx 0.02

Getting suspicious

Scenario C: 10 Heads, 0 Tails

Are you surprised? **Extremely.** This coin is probably rigged.

P≈0.001P \\approx 0.001

Reject H0H\_0!

#### Key Insight

The **lower** the p-value, the more **surprised** you should be if H0H\_0 were actually true. It measures the incompatibility between your data and the Null Hypothesis.

## The Formal Definition

The P-value is the probability of observing test results **at least as extreme** as the results actually observed, under the assumption that the **Null Hypothesis is correct**.

#### What it IS

P(Data∣H0)P(\\text{Data} | H\_0)

The probability of the **evidence** (data) given that the hypothesis is true.

#### What it is NOT

P(H0∣Data)P(H\_0 | \\text{Data})

NOT the probability that the **hypothesis** is true given the evidence. This is a common misconception! See Bayes' theorem.

## Interactive Demo: The Tail Area

Visually, the P-value is the **area under the curve** in the tail(s) beyond your observed test statistic. Drag the slider to see how more extreme data leads to smaller P-values.

### P-Value: The Tail Area

The P-value is the shaded area under the curve beyond your test statistic. Drag the slider to see how extreme data leads to smaller P-values.

Test Type

Your Observed Test Statistic: 1.80

\-3.5 (extreme left)0 (center)3.5 (extreme right)

Test Statistic

1.80

P-Value

0.0719

Surprise Level

Somewhat Surprised

**Interpretation:** The red shaded area IS your P-value. It represents the probability of seeing data this extreme (or more) if H0 were true. Smaller area = smaller P-value = more "surprised" = stronger evidence against H0.

## The P-Value Fallacy (ASA Statement)

In 2016, the American Statistical Association (ASA) released a landmark statement to correct rampant misuse of p-values in science.

1

#### "P\=0.05P = 0.05 means there is a 95% chance the hypothesis is true."

**Correction:** False. It only means that if the null were true, you would see data this extreme 5% of the time. It says nothing about the truth of the hypothesis itself.

2

#### "P\>0.05P > 0.05 means no effect."

**Correction:** Absence of evidence is not evidence of absence. A high p-value might just mean your sample size was too small to detect the effect (Low Power). See Type I vs Type II Errors.

3

#### "P\=0.04P = 0.04 is significantly better than P\=0.06P = 0.06."

**Correction:** The threshold of 0.05 is arbitrary. In reality, 0.04 and 0.06 represent very similar levels of evidence. Treat the p-value as a **continuous measure** of compatibility, not a hard switch.

## How to Interpret P-Values

| P-Value Range | Evidence Against H0 | Typical Interpretation |
| --- | --- | --- |
| P<0.001P < 0.001 | Very Strong | Extremely unlikely under H0H\_0. Strong rejection. |
| 0.001−0.010.001 - 0.01 | Strong | Very unlikely under H0H\_0. Usually reject. |
| 0.01−0.050.01 - 0.05 | Moderate | Conventionally "significant." Reject at α\=0.05\\alpha=0.05. |
| 0.05−0.100.05 - 0.10 | Weak | "Marginally significant." Worth investigating further. |
| P\>0.10P > 0.10 | Little to None | Data consistent with H0H\_0. Fail to reject. |

**Important:** These are guidelines, not rules. Context matters! In medical trials, you might want P<0.01P < 0.01. In exploratory analysis, P<0.10P < 0.10 might warrant further investigation.

## ML Application: Feature Selection

In Machine Learning, p-values are heavily used in **Filter Methods** for feature selection.

### Backward Elimination Algorithm

1

Train a linear regression model with ALL features.

2

Calculate the p-value for the coefficient (weight) of every feature.

H0:Coefficient\=0H\_0: \\text{Coefficient} = 0 (Feature adds no value)

3

Identify the feature with the highest p-value (e.g., 0.85).

4

If P\>0.05P > 0.05, drop that feature.

5

Retrain and repeat until all remaining features have P<0.05P < 0.05.

**Result:** A simpler, more interpretable model with less noise. Features with high p-values likely contain no predictive signal.

## Pitfalls: P-Hacking & Corrections

**P-Hacking** (or Data Dredging) is the practice of performing many statistical tests on data and only reporting those that come back with significant results.

#### The Multiple Comparisons Problem

If you run **100 independent tests** where H0H\_0 is actually true (no effect), with α\=0.05\\alpha=0.05, you will statistically expect:

5tests to return significant results purely by chance

These are Type I errors (false positives). If you only publish these 5 "discoveries," you are publishing garbage.

#### The Fix: Bonferroni Correction

If you run nn tests, divide your significance level by n:

αcorrected\=αn\\alpha\_{\\text{corrected}} = \\frac{\\alpha}{n}

**Example:** Running 20 tests? Use α\=0.05/20\=0.0025\\alpha = 0.05/20 = 0.0025 instead of 0.05. This is conservative but protects against false discoveries.

## Interactive Demo: P-Hacking Simulation

See p-hacking in action. Each test is run where H0H\_0 is TRUE (no real effect). Watch how many "significant" results appear by pure chance.

### P-Hacking Simulation

Each test is run where **H0 is actually TRUE** (no real effect). Watch how many "significant" results appear by pure chance when running multiple tests.

Number of Tests: 20

**Key Insight:** With alpha = 0.05, running 20 tests will produce approximately 1.0 false positives by chance alone. Red boxes = P < 0.05 (would be declared "significant").