**Title:** Random Variables & Expectation: Complete Guide

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/probability/random-variables](https://www.tensortonic.com/ml-math/probability/random-variables)

---

# Page Structure Map
```text
Random Variables & Expectation: Complete Guide
├── Introduction
├── What is a Random Variable?
│   └── Example: Two Coin Flips
├── Discrete vs Continuous
│   ├── Discrete
│   └── Continuous
├── Expected Value (Mean)
│   ├── Discrete
│   ├── Continuous
│   └── Example: Rigged Die
├── Interactive: Expected Value
├── Variance & Standard Deviation
│   ├── Variance
│   └── Standard Deviation
├── Interactive: Variance Comparison
├── Key Properties
│   ├── Linearity of Expectation
│   ├── Variance of Sum (Independent Variables)
│   └── Scaling Properties
└── ML Applications
    ├── Expected Risk (Loss Minimization)
    ├── SGD Justification
    ├── Reinforcement Learning
    └── Variance in Model Performance
```

---

## Introduction

In algebra, a variable x stands for a fixed number. In probability, a **Random Variable** represents a value determined by chance. It could be any value within a range, each with some probability.

**Random Variable:** A function that maps outcomes of a random process to numbers.

**Expectation:** The theoretical average if we repeated the process infinitely.

In ML, everything is a random variable: input data XX, target labels YY, and even model parameters  heta\\ heta (in [Bayesian learning](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist)). Understanding expectation is how we define what "good" means for a model.

## What is a Random Variable?

Formal Definition

X:Ω→RX: \\Omega \\to \\mathbb{R}

A function that assigns a real number to every possible outcome in the sample space Ω\\Omega.

#### Example: Two Coin Flips

**Sample Space:**

{HH, HT, TH, TT}

**X = number of Heads:**

X(HH) = 2

X(HT) = X(TH) = 1

X(TT) = 0

The random variable X converts outcomes (like "HT") into numbers (like 1) that we can do math with.

## Discrete vs Continuous

#### Discrete

Countable values: dice rolls, coin flips, number of emails.

Described by PMF:

P(X\=x)P(X = x)

Each value has a specific probability. Sum of all probabilities = 1.

#### Continuous

Any value in a range: height, time, temperature.

Described by PDF:

P(a<X<b)\=∫abf(x)dxP(a < X < b) = \\int\_a^b f(x)dx

P(X = exact value) = 0. Only ranges have nonzero probability.

For details on specific distributions, see [Probability Distributions](https://www.tensortonic.com/ml-math/probability/distributions).

## Expected Value (Mean)

The **Expectation** E\[X\] is the theoretical weighted average. Think of it as the "center of mass" of the distribution.

It answers: _"If we repeated this experiment infinitely, what would the average value be?"_

#### Discrete

E\[X\]\=∑xx⋅P(X\=x)E\[X\] = \\sum\_x x \\cdot P(X=x)

Sum of (value x probability)

#### Continuous

E\[X\]\=∫−∞∞x⋅f(x)dxE\[X\] = \\int\_{-\\infty}^{\\infty} x \\cdot f(x)dx

Integral of (value x density)

#### Example: Rigged Die

A die where rolling 6 is 50% likely, and 1-5 are 10% each.

E\[X\] = (1)(0.1) + (2)(0.1) + (3)(0.1) + (4)(0.1) + (5)(0.1) + (6)(0.5)

E\[X\] = 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 3.0 = **4.5**

The expected value is 4.5 (you can never roll 4.5!). The high probability on 6 pulls the average upward.

## Interactive: Expected Value

See how E\[X\] acts as the "balance point" of the distribution. Try different probability distributions and watch where the fulcrum lands.

Click bars to increase weight (+), Right-click to decrease (-)

Fulcrum finds the center of mass

## Variance & Standard Deviation

Expectation tells us the **center**. Variance tells us the **spread** (how uncertain we are).

Variance: Expected squared deviation from mean

Var(X)\=E\[(X−μ)2\]Var(X) = E\[(X - \\mu)^2\]

Computational formula:

Var(X)\=E\[X2\]−(E\[X\])2Var(X) = E\[X^2\] - (E\[X\])^2

#### Variance

Units are squared (e.g., dollars squared). Hard to interpret directly.

#### Standard Deviation

σ\=Var(X)\\sigma = \\sqrt{Var(X)}. Same units as XX. More interpretable.

## Interactive: Variance Comparison

Compare distributions with the same mean but different variances. Higher variance = wider spread = more uncertainty.

Low Variance

Tall & Narrow

High Certainty

Medium Variance

Balanced Shape

Moderate Certainty

High Variance

Short & Wide

Low Certainty

## Key Properties

#### Linearity of Expectation

For **any** random variables X and Y (even if dependent!):

E\[aX+bY\]\=aE\[X\]+bE\[Y\]E\[aX + bY\] = aE\[X\] + bE\[Y\]

This is incredibly powerful. It lets us break complex problems into simple pieces.

#### Variance of Sum (Independent Variables)

If X and Y are **independent**:

Var(X+Y)\=Var(X)+Var(Y)Var(X + Y) = Var(X) + Var(Y)

Variances add for independent variables. This is why errors accumulate.

#### Scaling Properties

E\[aX\]\=aE\[X\]E\[aX\] = aE\[X\]

Var(aX)\=a2Var(X)Var(aX) = a^2 Var(X)

Multiplying by a constant scales variance by a squared!

## ML Applications

#### Expected Risk (Loss Minimization)

We train models to minimize the expected loss on unseen data:

θ∗\=arg⁡min⁡θE(x,y)\[L(f(x;θ),y)\]\\theta^\* = \\arg\\min\_\\theta E\_{(x,y)}\[L(f(x;\\theta), y)\]

Since we cannot compute the true expectation, we approximate with the empirical average over training data.

#### SGD Justification

By linearity of expectation, the expected gradient of a mini-batch equals the expected gradient of the full dataset. This is why [Stochastic Gradient Descent](https://www.tensortonic.com/ml-math/optimization/sgd-variants) works!

#### Reinforcement Learning

The Value Function is the expected future return:

V(s)\=E\[∑t\=0∞γtrt∣s0\=s\]V(s) = E\\left\[\\sum\_{t=0}^\\infty \\gamma^t r\_t \\mid s\_0 = s\\right\]

The agent maximizes expected cumulative reward.

#### Variance in Model Performance

High variance in predictions means the model is sensitive to training data (overfitting). The bias-variance tradeoff is fundamentally about E\[ exterror\]E\[\\ ext{error}\] vs Var( exterror)Var(\\ ext{error}).