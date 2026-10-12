**Title:** Bayes' Theorem & Bayesian Inference

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/probability/bayes-theorm](https://www.tensortonic.com/ml-math/probability/bayes-theorm)

---

# Page Structure Map
```text
Bayes' Theorem & Bayesian Inference
├── Introduction
├── The Core Intuition
│   └── Example: Is My Friend Home?
├── The Formula
│   ├── Posterior
│   ├── Likelihood
│   ├── Prior
│   ├── Evidence
│   ├── Proportional Form
│   └── Odds Form
├── Interactive: Bayes Calculator
├── Derivation
├── The False Positive Paradox
│   └── The Setup
│       ├── Step 1: Numerator (True Positives)
│       ├── Step 2: Denominator (All Positives)
│       ├── Step 3: Result
│       └── Why only 50%?
├── Interactive: Medical Test Simulator
│   └── Total Positive Tests
├── Bayesian Inference in ML
│   ├── MAP Estimation
│   ├── [MLE](https://www.tensortonic.com/ml-math/statistics/maximum-likelihood-estimation) (Special Case)
│   └── The Denominator Problem
└── ML Applications
    ├── Naive Bayes Classifier
    ├── [Regularization](https://www.tensortonic.com/ml-math/optimization/regularization) = Prior
    ├── Bayesian Neural Networks
    └── Thompson Sampling
```

---

## Introduction

In everyday logic, we think cause to effect: "If it rains, the grass gets wet." But in reality, we often observe the **effect** and must infer the **cause**: "The grass is wet. Did it rain, or did the sprinklers turn on?"

**Bayes' Theorem** tells us how to _reverse_ conditional probabilities.

Given P(effect∣cause)P(\\text{effect} | \\text{cause}), compute P(cause∣effect)P(\\text{cause} | \\text{effect}).

This is the foundation of **Bayesian Inference**: treating learning as the continuous process of updating probability distributions as new data arrives. It is arguably the most important theorem in machine learning.

## The Core Intuition

Prior + Evidence = Posterior

Prior

Initial belief before seeing data

Likelihood

How compatible is data with hypothesis?

Posterior

Updated belief after seeing data

#### Example: Is My Friend Home?

-   **Prior:** 50% chance she is home (no idea initially).
-   **Evidence:** Lights are on.
-   **Likelihood:** If home, 90% chance lights are on. If not home, 10% chance lights are on.
-   **Posterior:** After seeing lights on, P(home | lights on) = ?

Bayes tells us: **90%**. The evidence dramatically shifted our belief.

## The Formula

P(A∣B)\=P(B∣A)⋅P(A)P(B)P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}

##### Posterior

P(A|B)

What we want

##### Likelihood

P(B|A)

Data fit

##### Prior

P(A)

Initial belief

##### Evidence

P(B)

Normalizer

#### Proportional Form

P(A∣B)∝P(B∣A)⋅P(A)P(A|B) \\propto P(B|A) \\cdot P(A)

Often we skip P(B)P(B) and just normalize at the end.

#### Odds Form

P(A∣B)P(¬A∣B)\=P(B∣A)P(B∣¬A)⋅P(A)P(¬A)\\frac{P(A|B)}{P(\\neg A|B)} = \\frac{P(B|A)}{P(B|\\neg A)} \\cdot \\frac{P(A)}{P(\\neg A)}

Posterior odds = Likelihood ratio x Prior odds

## Interactive: Bayes Calculator

Adjust prior and likelihood to see how the posterior changes. Notice how strong priors resist change, and how strong evidence can overcome weak priors.

Prior P(A)1.0%

Prevalence: How rare is the event?

Sensitivity P(B|A)99.0%

True Positive Rate: Ability to detect true cases.

false Positive Rate P(B|¬A)5.0%

False Alarm Rate: Healthy people testing positive.

**Geometric Intuition:** The Posterior P(A|B) is the fraction of total colored area (Positives) that is Green (True).

Even with **99% Sensitivity**, if the Prior is very low (e.g. 1%) and False Positive Rate is even moderate (5%), the Posterior drops to ~16%!

Most "positive" tests for rare diseases are actually false alarms. This is the **False Positive Paradox**.

## Derivation

Bayes' Theorem is not a new axiom. It follows directly from the definition of conditional probability and the product rule.

1\. Product rule (two ways):

P(A∩B)\=P(A∣B)⋅P(B)\=P(B∣A)⋅P(A)P(A \\cap B) = P(A|B) \\cdot P(B) = P(B|A) \\cdot P(A)

2\. Set them equal:

P(A∣B)⋅P(B)\=P(B∣A)⋅P(A)P(A|B) \\cdot P(B) = P(B|A) \\cdot P(A)

3\. Divide by P(B):

P(A∣B)\=P(B∣A)⋅P(A)P(B)P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}

See [Conditional Probability](https://www.tensortonic.com/ml-math/probability/conditional-probability) for background on P(A∣B)P(A|B).

## The False Positive Paradox

This classic example reveals why humans are notoriously bad at intuitive probability.

### The Setup

-   **1%** of population has the disease (Prior = 0.01)
-   Test is **99%** accurate for sick people (Sensitivity = 0.99)
-   Test has **1%** false positive rate for healthy people

You test positive. What is P(Disease | Positive)?

Most people guess 99%. They are wrong.

#### Step 1: Numerator (True Positives)

P(+∣D)×P(D)\=0.99×0.01\=P(+|D) \\times P(D) = 0.99 \\times 0.01 = 0.0099

#### Step 2: Denominator (All Positives)

P(+)\=P(+∣D)P(D)+P(+∣H)P(H)P(+) = P(+|D)P(D) + P(+|H)P(H)

\=0.99(0.01)+0.01(0.99)\=\= 0.99(0.01) + 0.01(0.99) = 0.0198

#### Step 3: Result

P(D∣+)\=0.0099/0.0198\=P(D|+) = 0.0099 / 0.0198 = **50%**

#### Why only 50%?

The disease is rare. In 10,000 people, only 100 are sick (99 test positive). But 9,900 are healthy, and 1% of those (99 people) also test positive! Half of all positives are false alarms.

## Interactive: Medical Test Simulator

Adjust disease rate, sensitivity, and false positive rate to see how they affect the posterior probability. Watch the ratio of true vs false positives change.

Prevalence1.0%

Rare (0.1%)Common (10%)

Sensitivity (TPR)99.0%

Good (90%)Perfect (100%)

False Pos Rate (FPR)1.0%

Strict (0.1%)Loose (10%)

#### Total Positive Tests

9.9 Real9.9 False

The Reality

If you test positive, what is the chance you actually have the disease?

50.0%

(Not 99%)

## Bayesian Inference in ML

In ML, we replace events (A, B) with parameters (theta) and data (D):

P(θ∣D)\=P(D∣θ)⋅P(θ)P(D)P(\\theta | D) = \\frac{P(D | \\theta) \\cdot P(\\theta)}{P(D)}

Posterior

Updated weights

Likelihood

Data fit

Prior

Regularization

Evidence

Intractable!

#### MAP Estimation

Find  heta\\ heta that maximizes posterior:

θMAP\=arg⁡max⁡θP(D∣θ)P(θ)\\theta\_{MAP} = \\arg\\max\_\\theta P(D|\\theta)P(\\theta)

#### [MLE](https://www.tensortonic.com/ml-math/statistics/maximum-likelihood-estimation) (Special Case)

If prior is uniform (flat):

θMLE\=arg⁡max⁡θP(D∣θ)\\theta\_{MLE} = \\arg\\max\_\\theta P(D|\\theta)

#### The Denominator Problem

Computing P(D) requires integrating over all possible theta:

P(D)\=∫P(D∣θ)P(θ)dθP(D) = \\int P(D|\\theta)P(\\theta) d\\theta

For neural networks with millions of parameters, this is impossible. Solutions: MCMC, Variational Inference, Laplace Approximation.

## ML Applications

#### Naive Bayes Classifier

Fast text classification (spam detection). Assumes feature independence:

P(y∣x1,...,xn)∝P(y)∏iP(xi∣y)P(y|x\_1,...,x\_n) \\propto P(y)\\prod\_i P(x\_i|y)

#### [Regularization](https://www.tensortonic.com/ml-math/optimization/regularization) = Prior

L2 regularization is equivalent to a Gaussian prior on weights:

P(θ)∼N(0,σ2)P(\\theta) \\sim \\mathcal{N}(0, \\sigma^2)

Belief: "Weights should be small."

#### Bayesian Neural Networks

Instead of point estimates, maintain distributions over weights. Provides uncertainty quantification.

#### Thompson Sampling

Exploration-exploitation in bandits. Sample from posterior, act on sample. Naturally balances uncertainty.

For a comparison of Bayesian vs Frequentist approaches, see [Bayesian vs Frequentist](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist).