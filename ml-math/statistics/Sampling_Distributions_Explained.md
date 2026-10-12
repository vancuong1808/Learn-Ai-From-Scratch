**Title:** Sampling Distributions Explained

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/sampling-distribution](https://www.tensortonic.com/ml-math/statistics/sampling-distribution)

---

# Page Structure Map
```text
Sampling Distributions Explained
├── Introduction
│   └── Why This Matters for ML Engineers
├── Building Intuition: The Bulb Factory Example
│   └── The Intuition
├── The Core Concept
├── Interactive Demo: Watch It Happen
├── Sampling Distribution of the Mean
│   ├── Rule 1: Unbiased Estimator
│   └── Rule 2: Standard Error
├── Crucial: Standard Deviation vs Standard Error
│   └── Practical Example
├── Interactive: Standard Error Demo
├── The Central Limit Theorem (CLT)
│   └── The Central Limit Theorem
│       ├── Try it yourself
│       └── Want the Deep Dive?
├── Sampling Distribution of Proportions
│   ├── Mean of p^\\hat{p}
│   ├── Standard Error of p^\\hat{p}
│   └── Validity Check for Normal Approximation
├── The T-Distribution
│   ├── T-Statistic Formula
│   ├── The T-Test: Why It Matters
│   └── Practical Application
├── Interactive: T-Distribution Demo
│   └── What are Degrees of Freedom (df)?
├── Worked Examples
│   ├── Example 1: Bulb Lifespan Warranty
│   └── Example 2: Defect Rate
└── Machine Learning Applications
    ├── 1\. Cross-Validation Scores
    ├── 2\. Ensemble Learning (Bagging)
    ├── 3\. A/B Testing
    └── 4\. Mini-Batch Gradient Descent
```

---

## Introduction

In the previous chapter, we learned that a **Sample** is a subset of a **Population**. But here is an important realization: if you take one sample and calculate its mean, you get one number. If you take _another_ random sample of the same size, you get a _different_ number.

This raises a critical question: **How much can my sample mean vary from sample to sample?**

If you kept doing this, taking thousands of samples and plotting their means, you would create a new distribution. This is the **Sampling Distribution**. It is not a distribution of raw data; it is a probability distribution of a _statistic_ (like the mean).

### Why This Matters for ML Engineers

In Machine Learning, we treat our training set as a single sample from all possible data. Understanding sampling distributions helps answer:

-   _"If I retrained this model on different data, how much would my accuracy fluctuate?"_
-   _"Is the 2% improvement from Model B actually significant, or just noise?"_
-   _"How confident can I be in my cross-validation score?"_

## Building Intuition: The Bulb Factory Example

Imagine you run a light bulb factory and want to know the **average lifespan** of all bulbs produced (the population mean μ\\mu).

The Problem

You cannot test every single bulb (destructive testing). You only have time to sample 50 bulbs this week.

The Solution

Use the sample mean xˉ\\bar{x} as an **estimate** of μ\\mu.

But here is the catch: if another engineer tested 50 different bulbs, they would get a slightly different average. The **Sampling Distribution** tells us the range of values we should expect for xˉ\\bar{x} and how confident we can be.

### The Intuition

**Think of it this way:** The sampling distribution is like a "meta-distribution" - it describes the behavior of statistics (like means) calculated from many hypothetical samples, not the behavior of individual data points.

## The Core Concept

Imagine a "God View" where we know the entire population. We take repeated samples of a fixed size nn.

Step 1

Take Sample 1

Calc xˉ1\\bar{x}\_1

Step 2

Take Sample 2

Calc xˉ2\\bar{x}\_2

Step 3

Repeat 1000x

xˉ3…xˉ1000\\bar{x}\_3 \\ldots \\bar{x}\_{1000}

Step 4

Plot histogram

Sampling Dist

The resulting histogram of these means is the **Sampling Distribution of the Sample Mean**. Notice we are plotting _statistics_ (means), not raw data points.

## Interactive Demo: Watch It Happen

Click "Start Sampling" to watch how repeatedly sampling from a **skewed** population creates a **bell-shaped** distribution of sample means. This is the Central Limit Theorem in action!

## Sampling Distribution of the Mean

If we draw samples from a population with mean μ\\mu and standard deviation σ\\sigma, the distribution of sample means xˉ\\bar{x} follows two fundamental rules:

### Rule 1: Unbiased Estimator

μxˉ\=μ\\mu\_{\\bar{x}} = \\mu

The average of all your sample means equals the true population mean. There is no systematic tendency to over or underestimate.

Example: If true mean is 50, some samples give 48, some give 52, but the average of all sample means is exactly 50.

### Rule 2: Standard Error

σxˉ\=σn\\sigma\_{\\bar{x}} = \\frac{\\sigma}{\\sqrt{n}}

The spread of sample means is smaller than the spread of individual data points. As sample size nn increases, the means cluster tighter around μ\\mu.

Example: If σ\=20\\sigma=20 and n\=100n=100, then SE\=20/10\=2SE = 20/10 = 2.

## Crucial: Standard Deviation vs Standard Error

This is a **common interview question**. Many candidates confuse these two. Here is the difference:

| Metric | Symbol | Formula | What it measures |
| --- | --- | --- | --- |
| Standard Deviation | σ\\sigma or ss | σ\\sigma | Variability of **individual data points** |
| Standard Error | SESE or σxˉ\\sigma\_{\\bar{x}} | σn\\frac{\\sigma}{\\sqrt{n}} | Variability of the **sample mean** |

#### Practical Example

Suppose bulb lifespans have σ\=200\\sigma = 200 hours (high variation between individual bulbs).

-   If you sample n\=25n=25 bulbs: SE\=200/5\=40SE = 200/5 = 40 hours
-   If you sample n\=100n=100 bulbs: SE\=200/10\=20SE = 200/10 = 20 hours
-   If you sample n\=400n=400 bulbs: SE\=200/20\=10SE = 200/20 = 10 hours

**Key Insight:** To halve the SE, you need to **quadruple** the sample size. This is the square root law!

## Interactive: Standard Error Demo

Drag the slider to see how increasing sample size reduces the Standard Error:

## The Central Limit Theorem (CLT)

The CLT is arguably the most important theorem in statistics. It is the reason we can do most statistical inference. You have already seen it in action in the interactive demo above!

### The Central Limit Theorem

If the sample size nn is **large enough** (typically n≥30n \\ge 30), the sampling distribution of the mean will be approximately Normal, regardless of the shape of the original population distribution.

#### Try it yourself

Go back to the Interactive Demo above and set a small sample size (n=5) vs large (n=50). Notice how the histogram becomes more "Normal" shaped as n increases!

#### Want the Deep Dive?

The CLT has its own dedicated chapter where we cover:

-   Mathematical proof using Moment Generating Functions
-   Why the "n = 30" rule exists and when it fails
-   Applications in Finance (Portfolio Theory, VaR)

Read the full CLT chapter

## Sampling Distribution of Proportions

When dealing with categorical data (Success/Failure, Click/No-Click, 0/1), we look at the sample proportion:

p^\=xn\\hat{p} = \\frac{x}{n}

where x = number of successes, n = sample size

#### Mean of p^\\hat{p}

μp^\=p\\mu\_{\\hat{p}} = p

Unbiased estimator of population proportion.

#### Standard Error of p^\\hat{p}

σp^\=p(1−p)n\\sigma\_{\\hat{p}} = \\sqrt{\\frac{p(1-p)}{n}}

Notice: SE is maximized when p = 0.5.

#### Validity Check for Normal Approximation

The sampling distribution of p^\\hat{p} is approximately Normal only if:

np≥10np \\ge 10

At least 10 expected successes

n(1−p)≥10n(1-p) \\ge 10

At least 10 expected failures

## The T-Distribution

So far, we assumed we know the population standard deviation σ\\sigma. But in practice, we almost never know it! We have to estimate it using the sample standard deviation ss.

When we substitute ss for σ\\sigma in our formulas, the uncertainty increases. The resulting distribution is called the **Student t-distribution**.

#### T-Statistic Formula

t\=xˉ−μs/nt = \\frac{\\bar{x} - \\mu}{s / \\sqrt{n}}

Notice: ss (sample SD) instead of σ\\sigma (population SD)

Fat Tails

The t-distribution has "fatter tails" than Normal. This accounts for extra uncertainty when n is small.

Degrees of Freedom

The shape is determined by df\=n−1df = n - 1. Lower df = fatter tails.

Convergence

As nn grows large (df > 30), the t-distribution becomes nearly identical to Normal.

#### The T-Test: Why It Matters

The **t-test** is a statistical test that uses the t-distribution to determine if there is a significant difference between group means. It answers questions like: "Is this difference real, or just random noise?"

One-Sample T-Test

Compare a sample mean to a known value. Example: "Is our bulb lifespan different from 1000 hours?"

Two-Sample T-Test

Compare means of two groups. Example: "Do users who see version A convert better than version B?"

**Key insight:** The t-test accounts for sample size. With small samples, you need larger differences to claim statistical significance, because small samples have more uncertainty.

#### Practical Application

The t-distribution is used extensively in hypothesis testing. See the One-Sample T-Test chapter for step-by-step examples of using t-tests in practice.

## Interactive: T-Distribution Demo

Drag the slider to see how degrees of freedom affect the shape of the t-distribution:

#### What are Degrees of Freedom (df)?

**Degrees of freedom = n - 1**, where n is your sample size. It represents the number of independent values that can vary when estimating a parameter. When you calculate a sample mean, one value becomes "fixed" (constrained by the mean), so you lose one degree of freedom. With small df, there is more uncertainty, hence fatter tails.

## Worked Examples

### Example 1: Bulb Lifespan Warranty

**Scenario:** Your bulbs have mean lifespan μ\=1000\\mu = 1000 hours and σ\=50\\sigma = 50 hours. You test n\=100n=100 bulbs. What is the probability the sample mean is less than 990 hours?

1\. Standard Error: SE\=50100\=5SE = \\frac{50}{\\sqrt{100}} = 5 hours

2\. Z-Score: Z\=990−10005\=−2.0Z = \\frac{990 - 1000}{5} = -2.0

3\. Probability: P(Z<−2.0)≈0.0228P(Z < -2.0) \\approx 0.0228 (2.28%)

**Conclusion:** Only 2.3% chance of seeing a mean this low by random chance. If you observe this, your production line might be failing!

### Example 2: Defect Rate

**Scenario:** Historical defect rate is p\=0.05p = 0.05 (5%). You run a quality check on n\=1000n = 1000 bulbs. What is the SE of the sample proportion?

1\. Check validity: np\=50≥10np = 50 \\ge 10, n(1−p)\=950≥10n(1-p) = 950 \\ge 10 - OK!

2\. Standard Error: SE\=0.05×0.951000≈0.0069SE = \\sqrt{\\frac{0.05 \\times 0.95}{1000}} \\approx 0.0069

**Meaning:** We expect the sample defect rate to be within about 0.7% of the true 5% rate (so roughly 4.3% to 5.7%).

## Machine Learning Applications

Sampling distributions are everywhere in ML, from model evaluation to optimization.

### 1\. Cross-Validation Scores

When you run 5-fold CV, you get 5 accuracy scores. The mean of these is a sample statistic! The standard error of this mean tells you how stable your estimate is. Report it alongside your mean accuracy.

### 2\. Ensemble Learning (Bagging)

Random Forest creates many bootstrap samples and trains trees on each. The final prediction is an average. By the CLT formula SE\=σ/nSE = \\sigma/\\sqrt{n}, averaging n trees reduces prediction variance by n\\sqrt{n}.

### 3\. A/B Testing

When comparing Model A vs Model B, you compare their average metrics. The sampling distribution helps calculate confidence intervals. If intervals do not overlap, you have a statistically significant difference.

### 4\. Mini-Batch Gradient Descent

In SGD, a mini-batch is a sample. The gradient from that batch is an estimate of the true gradient. Larger batch sizes reduce "noise" (SE of gradient) but compute more per step. This is SE in action!