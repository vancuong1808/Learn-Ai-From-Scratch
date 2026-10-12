**Title:** Descriptive Statistics: Mean, Median, Mode, Variance & Std Dev

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/statistics/basic-statistics](https://www.tensortonic.com/ml-math/statistics/basic-statistics)

---

# Page Structure Map
```text
Descriptive Statistics: Mean, Median, Mode, Variance & Std Dev
├── Introduction
│   ├── What is Descriptive Statistics?
│   └── Why This Matters for ML Engineers
├── The Mean (Average)
│   └── Definition
│       └── Example: Bulb Lifespans
├── The Mode (Most Frequent)
│   └── Definition
│       └── Example: Categorical Data
├── Skewness
│   └── Skewness Explorer
├── Kurtosis
│   ├── Definition
│   └── Kurtosis Explorer
├── Measures of Spread (Dispersion)
├── Variance and Standard Deviation
│   ├── Variance
│   └── Standard Deviation
│       └── Example: Bulb Lifespans
├── Range and Interquartile Range (IQR)
│   ├── Range
│   ├── Interquartile Range (IQR)
│   └── Measure Spread: Focus + Context
└── Formulas Reference
```

---

## Introduction

Before diving into machine learning algorithms, we need a solid foundation in **descriptive statistics**. These are the tools that allow us to summarize and understand data at a glance.

Think about your exam scores from last semester. Instead of memorizing every single grade, wouldn't it be easier to just know your _average_? That's statistics in action. Taking a bunch of numbers and distilling them into something useful.

### What is Descriptive Statistics?

**Descriptive statistics** are the tools we use to _summarize_ and _describe_ data. Instead of looking at thousands of numbers, we extract a few key metrics that tell the story.

For example: "This app has a 4.2 star rating based on 10,000 reviews." That one number summarizes thousands of individual opinions into a single, useful metric.

In this guide, we'll cover the essential building blocks: **mean, median, mode** (where is the "center" of your data?), and **variance, standard deviation** (how spread out is your data?). These concepts show up everywhere: from understanding survey results to building machine learning models and understanding probability distributions.

### Why This Matters for ML Engineers

-   **Make sense of data:** Quickly summarize thousands of numbers into a few key insights.
-   **Spot problems:** Identify unusual values that might be errors or important exceptions.
-   **Compare fairly:** Understand if two datasets are truly different or just appear so.
-   **Build intuition:** These concepts are the foundation for everything in data science, from Bayesian inference to batch normalization.

## The Mean (Average)

The most common way to summarize data is to find its **center**: a single value that represents the "typical" data point. The mean is the most widely used measure of center.

### Definition

The **arithmetic mean** is the sum of all values divided by the count of values. It is what most people call "the average."

xˉ\=1n∑i\=1nxi\=x1+x2+…+xnn\\bar{x} = \\frac{1}{n} \\sum\_{i=1}^{n} x\_i = \\frac{x\_1 + x\_2 + \\ldots + x\_n}{n}

#### Example: Bulb Lifespans

You test 5 bulbs with lifespans: 980, 1000, 1010, 1020, 1040 hours.

Mean = (980 + 1000 + 1010 + 1020 + 1040) / 5 = **1010 hours**

## The Mode (Most Frequent)

Sometimes you do not care about the average or middle value. You want to know what shows up the most. This is especially useful for categorical data where mean and median do not apply.

### Definition

The **mode** is the value that appears most frequently in the dataset. A dataset can have no mode, one mode (unimodal), or multiple modes (bimodal, multimodal).

#### Example: Categorical Data

Bulb quality ratings: Good, Good, Good, Excellent, Fair, Good, Excellent

Mode = **Good** (appears 4 times)

## Skewness

So far we have covered three ways to find the center of your data: the mean, median, and mode. But here is a question: what if the mean and median give different answers? That happens when your data is not symmetric.

**Skewness** describes the shape of your data distribution. Is it balanced on both sides, or does it have a long tail stretching in one direction?

Symmetric

Skewness ≈0\\approx 0

Mean ≈\\approx Median ≈\\approx Mode

Right-Skewed (+)

Skewness \>0\> 0

Tail stretches right. Mode << Median << Mean.

Left-Skewed (-)

Skewness <0< 0

Tail stretches left. Mean << Median << Mode.

### Skewness Explorer

Drag the slider to see how skewness affects the shape of the distribution and the relationship between Mean, Median, and Mode.

SkewnessSymmetric

LeftRight

Mean

Median

Mode

Mean ≈ Median ≈ Mode

In symmetric data, the measures of center coincide.

**ML Tip:** Many machine learning models perform poorly on skewed data. If your data is heavily skewed (like house prices, where a few mansions distort the average), we often apply mathematical tricks to make it more symmetric before training.

## Kurtosis

While skewness tells us about the _asymmetry_ of our data, **kurtosis** tells us about the _tails_. Specifically: how likely are extreme values (outliers) compared to a normal distribution?

### Definition

**Kurtosis** measures the "tailedness" of a distribution. It tells you whether your data has heavy tails (more outliers) or light tails (fewer outliers) compared to a normal distribution.

Kurtosis\=1n∑i\=1n(xi−xˉσ)4\\text{Kurtosis} = \\frac{1}{n} \\sum\_{i=1}^{n} \\left( \\frac{x\_i - \\bar{x}}{\\sigma} \\right)^4

The fourth power amplifies extreme deviations, making kurtosis highly sensitive to outliers.

Platykurtic

Kurtosis <3< 3

Flatter peak, thinner tails. Fewer extreme values than normal.

Mesokurtic

Kurtosis \=3\= 3

Normal distribution. The baseline for comparison.

Leptokurtic

Kurtosis \>3\> 3

Sharper peak, fatter tails. More extreme values than normal.

### Kurtosis Explorer

Drag the slider to see how kurtosis affects the "tailedness" of the distribution. The dashed line shows a normal distribution for reference.

KurtosisMesokurtic (Normal)

FlatPeaked

Platykurtic

Kurtosis < 3

Flat peak, thin tails

Mesokurtic

Kurtosis = 3

Normal distribution

Leptokurtic

Kurtosis > 3

Sharp peak, fat tails

**Mesokurtic:** Similar tailedness to a normal distribution. This is the baseline (kurtosis = 3, or excess kurtosis = 0).

## Measures of Spread (Dispersion)

We have looked at the **center** (mean, median) and the **shape** (skewness). But there is one more critical piece of the puzzle: **spread**.

Imagine two companies with the exact same average salary of $100k. In Company A, everyone earns $100k. In Company B, the CEO earns $1M and everyone else earns $10k. The average is the same, but the _spread_ tells the real story.

So how do we measure this? We need a way to calculate how far, on average, the data points are from the center.

## Variance and Standard Deviation

These are the most commonly used measures of spread in statistics and machine learning. They quantify how much each data point deviates from the mean.

### Variance

σ2\=1n∑i\=1n(xi−xˉ)2\\sigma^2 = \\frac{1}{n} \\sum\_{i=1}^{n} (x\_i - \\bar{x})^2

The average of the squared deviations from the mean. Larger variance means more spread.

### Standard Deviation

σ\=σ2\=1n∑i\=1n(xi−xˉ)2\\sigma = \\sqrt{\\sigma^2} = \\sqrt{\\frac{1}{n} \\sum\_{i=1}^{n} (x\_i - \\bar{x})^2}

The square root of the variance. Same units as the original data.

#### Example: Bulb Lifespans

Lifespans: 980, 1000, 1010, 1020, 1040 hours

Step 1: Calculate the mean

Mean = (980 + 1000 + 1010 + 1020 + 1040) / 5 = 5050 / 5 = **1010 hours**

Step 2: Find each deviation from the mean

980 - 1010 = -30

1000 - 1010 = -10

1010 - 1010 = 0

1020 - 1010 = +10

1040 - 1010 = +30

Step 3: Square each deviation

(-30)² = 900

(-10)² = 100

(0)² = 0

(+10)² = 100

(+30)² = 900

Step 4: Calculate variance (average of squared deviations)

Variance = (900 + 100 + 0 + 100 + 900) / 5

Variance = 2000 / 5 = **400 hours²**

Step 5: Calculate standard deviation (square root of variance)

Std Dev = √400 = **20 hours**

**Why Square the Deviations?** Two reasons: (1) Positive and negative deviations would cancel out otherwise. (2) Squaring penalizes larger deviations more heavily, making the metric more sensitive to outliers.

## Range and Interquartile Range (IQR)

### Range

Range\=Max−Min\\text{Range} = \\text{Max} - \\text{Min}

The simplest measure of spread. Just the difference between the largest and smallest values.

**Weakness:** Extremely sensitive to outliers. One extreme value can inflate the range dramatically.

### Interquartile Range (IQR)

IQR\=Q3−Q1\\text{IQR} = Q\_3 - Q\_1

The range of the middle 50% of the data. Q1 is the 25th percentile, Q3 is the 75th percentile.

**Strength:** Robust to outliers because it ignores the extreme 25% on each end.

**Outlier Detection:** A common rule is to flag any data point as an outlier if it falls below Q1−1.5×IQRQ\_1 - 1.5 \\times \\text{IQR} or above Q3+1.5×IQRQ\_3 + 1.5 \\times \\text{IQR}. This is how box plots draw their "whiskers."

### Measure Spread: Focus + Context

Use the slider to change variance. Compare the full range vs the central IQR.

Range

40

IQR

15

Variance

139

Std Dev

11.8

## Formulas Reference

| Metric | Formula | Use When |
| --- | --- | --- |
| Mean | xˉ\=1n∑xi\\bar{x} = \\frac{1}{n}\\sum x\_i | Data is symmetric, no outliers |
| Median | Middle value (sorted) | Skewed data or outliers present |
| Mode | Most frequent value | Categorical data or identifying peaks |
| Range | Max−Min\\text{Max} - \\text{Min} | Quick overview, no outliers |
| IQR | Q3−Q1Q\_3 - Q\_1 | Robust spread, outlier detection |
| Variance | σ2\=1n∑(xi−xˉ)2\\sigma^2 = \\frac{1}{n}\\sum(x\_i - \\bar{x})^2 | Mathematical analysis, ML algorithms |
| Std Dev | σ\=σ2\\sigma = \\sqrt{\\sigma^2} | Interpretable spread in original units |