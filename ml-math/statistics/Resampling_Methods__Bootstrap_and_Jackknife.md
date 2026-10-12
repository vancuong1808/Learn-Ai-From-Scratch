**Title:** Resampling Methods: Bootstrap and Jackknife

**Author:** TensorTonic

**Source:** [https://www.tensortonic.com/ml-math/statistics/resampling](https://www.tensortonic.com/ml-math/statistics/resampling)

---

# Page Structure Map
```text
Resampling Methods: Bootstrap and Jackknife
├── Pulling Yourself Up By Your Bootstraps
├── The Bootstrap
│   ├── The Algorithm
│   └── Why "With Replacement"?
├── Interactive Simulator
├── The Jackknife
│   ├── Bootstrap
│   └── Jackknife
├── Bootstrap Confidence Intervals
│   ├── Percentile Method
│   └── BCa (Bias-Corrected and Accelerated)
├── Case Study: Bulb Lifespan Median
│   ├── The Problem
│   ├── Bootstrap Solution
│   └── The Result
├── Permutation Tests
│   └── The Idea
└── ML Applications
    ├── Bagging (Bootstrap Aggregating)
    ├── Out-of-Bag (OOB) Error
    └── Model Uncertainty Estimation
```

---

## Pulling Yourself Up By Your Bootstraps

Traditional statistics relies on formulas that assume we know the underlying distribution (e.g., "Assume data is Normal"). But what if we don't? What if the math for the standard error of the median is too hard? What if we have a weird statistic with no closed-form formula?

**Resampling** allows us to estimate the precision of sample statistics (medians, variances, percentiles) by using subsets of available data. It trades computing power for mathematical simplicity. In the age of fast computers, this is a no-brainer.

## The Bootstrap

Invented by Bradley Efron at Stanford in 1979. The idea is deceptively simple: Treat your sample as if it were the population.

### The Algorithm

1.  Take your original sample SS of size nn.
2.  Draw a new sample S∗S^\* of size nn from SS **with replacement**. (Some points appear twice, some zero times).
3.  Calculate your statistic (e.g., mean, median, 90th percentile) on S∗S^\*.
4.  Repeat 10,000 times (or more).
5.  The distribution of these 10,000 statistics approximates the true Sampling Distribution.

### Why "With Replacement"?

If you sampled without replacement, every bootstrap sample would be identical to the original (just shuffled). With replacement, each bootstrap sample is a _different_ random subset, mimicking the randomness of sampling from the true population. About 63.2% of original points appear in each bootstrap sample (the rest are duplicates or missing).

## Interactive Simulator

Watch how resampling creates a "Sampling Distribution" from a single dataset. Each bar in the histogram is one bootstrap mean.

## The Jackknife

The older, pre-computer-era cousin of the Bootstrap. Instead of random sampling with replacement, it systematically leaves out one observation at a time.

### Bootstrap

-   Random sampling with replacement.
-   Can generate infinite samples.
-   Works for any statistic (medians, quantiles).
-   Preferred in modern practice.

### Jackknife

-   Deterministic (Leave-one-out).
-   Only generates nn samples.
-   Mainly used for bias estimation.
-   Fails for non-smooth statistics (e.g., median).

θ^Jack\=nθ^−n−1n∑i\=1nθ^(i)\\hat{\\theta}\_{Jack} = n\\hat{\\theta} - \\frac{n-1}{n}\\sum\_{i=1}^{n} \\hat{\\theta}\_{(i)}

θ^(i)\\hat{\\theta}\_{(i)} is the statistic calculated with the i-th observation removed. This formula corrects for bias.

## Bootstrap Confidence Intervals

How do you get a 95% Confidence Interval for the **Median**? There is no simple formula like xˉ±1.96sn\\bar{x} \\pm 1.96 \\frac{s}{\\sqrt{n}} for the mean.

### Percentile Method

The simplest approach. Just take the 2.5th percentile and the 97.5th percentile of your 10,000 bootstrap statistics. That's your 95% CI. It works for _any_ statistic.

### BCa (Bias-Corrected and Accelerated)

Adjusts for bias and skewness in the bootstrap distribution. More accurate than Percentile, especially for small samples. Standard in statistical software.

## Case Study: Bulb Lifespan Median

#### The Problem

You have 30 light bulbs and measured their lifespan until failure. The median lifespan is 1,200 hours. Marketing wants to advertise the median with a confidence interval. What's the 95% CI?

#### Bootstrap Solution

1.  Resample 30 values from your 30 bulbs (with replacement).
2.  Calculate the median of this bootstrap sample.
3.  Repeat 10,000 times.
4.  Sort the 10,000 medians.
5.  Take the 250th value (2.5th percentile) and the 9750th value (97.5th percentile).

#### The Result

**95% CI for Median Lifespan:** \[1,100 hours, 1,320 hours\]. Marketing can now say: "Our bulbs last a median of 1,200 hours, with 95% confidence between 1,100 and 1,320 hours."

## Permutation Tests

A cousin of Bootstrap used for hypothesis testing. Instead of estimating a CI, we ask: "Is this difference significant?"

### The Idea

1.  If the null hypothesis is true (no difference between groups), then the group labels are arbitrary.
2.  Randomly shuffle the group labels 10,000 times.
3.  For each shuffle, calculate the difference in means.
4.  How often does the shuffled difference exceed the observed difference?
5.  That proportion is your p-value.

**Bulb Example:** Compare new packaging (n=50) vs old packaging (n=50). Observed difference in sales rate = 2%. After 10,000 permutations, only 3% of shuffles show a difference ≥ 2%. P-value = 0.03. Significant!

## ML Applications

### Bagging (Bootstrap Aggregating)

This is the secret sauce behind **Random Forests**.

-   Create 100 bootstrap samples of your training data.
-   Train a Decision Tree on each sample.
-   Average their predictions (regression) or vote (classification).

**Why?** It reduces Variance without increasing Bias. Individual trees overfit to specific noise in their bootstrap sample. Averaging cancels out these random errors.

### Out-of-Bag (OOB) Error

In Bagging, about 37% of data is never seen by a specific tree (because sampling is with replacement). We can use this "leftover" data as a free validation set to estimate test error without needing a separate hold-out set.

### Model Uncertainty Estimation

Train the same model on 100 bootstrap samples. The variance in predictions across models gives you an estimate of model uncertainty. Used in [Bayesian](https://www.tensortonic.com/ml-math/probability/bayes-theorm)\-style uncertainty quantification.