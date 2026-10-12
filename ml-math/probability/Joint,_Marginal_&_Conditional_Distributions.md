**Title:** Joint, Marginal & Conditional Distributions

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/probability/joint-marginal](https://www.tensortonic.com/ml-math/probability/joint-marginal)

---

# Page Structure Map
```text
Joint, Marginal & Conditional Distributions
├── Introduction
├── The Trio Defined
│   ├── Joint
│   ├── Marginal
│   └── Conditional
├── Interactive: Contingency Table
│   └── Joint Distribution Landscape
├── Joint Distribution P(X, Y)
│   └── What can we compute from P(X,Y)P(X,Y)?
├── Marginal Distribution P(X)
│   ├── The Sum Rule
│   └── Why is it called "marginal"?
├── Interactive: Marginalization
│   └── The Joint Distribution P(X, Y)
├── Conditional Distribution P(Y|X)
│   ├── Example
│   └── Why normalize?
├── Product Rule & Chain Rule
│   └── GPT Connection
└── ML Applications
    ├── Discriminative Models
    ├── Generative Models
    ├── [Bayesian Inference](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist)
    └── Latent Variable Models
```

---

## Introduction

In ML, we rarely deal with single variables in isolation. Features interact, labels depend on inputs, and model parameters relate to each other. To model these relationships, we need **multivariate probability**.

Think of a dataset with multiple columns. There are three fundamental ways to view it:

Joint

The whole picture

Marginal

One column, ignoring others

Conditional

One column, given another

Prerequisites: [Conditional Probability](https://www.tensortonic.com/ml-math/probability/conditional-probability), [Random Variables](https://www.tensortonic.com/ml-math/probability/random-variables).

## The Trio Defined

#### Joint

P(X,Y)P(X, Y)

Probability X takes value x **AND** Y takes value y simultaneously.

#### Marginal

P(X)P(X)

Probability of X, **regardless** of Y. Sum out the other variable.

#### Conditional

P(Y∣X)P(Y | X)

Probability of Y, **given** we observed X. Slice and normalize.

## Interactive: Contingency Table

This table shows 100 people categorized by Age (Young/Old) and Coffee preference (Latte/Espresso). Click the buttons to highlight different distributions.

### Joint Distribution Landscape

Visualize the joint probability P(X,Y)P(X,Y) as a heatmap. Mutual Information measures how "structured" this landscape is compared to the product of marginals.

Noise / Spread

Entropy H(X)

4.30

Entropy H(Y)

4.30

Joint H(X,Y)

7.84

Mutual Info I(X;Y)

0.76

Observation: When there is structure (lines, circles, clusters), the joint distribution is 'sharper' than the product of marginals. MI is high.

## Joint Distribution P(X, Y)

The joint distribution is the "master" distribution. If you have P(X, Y), you can derive everything else: marginals, conditionals, expectations, covariances.

Normalization: Sum over all combinations = 1

∑x∑yP(X\=x,Y\=y)\=1\\sum\_x \\sum\_y P(X=x, Y=y) = 1

For continuous: double integral of f(x,y) = 1

#### What can we compute from P(X,Y)P(X,Y)?

-   Marginals: P(X),P(Y)P(X), P(Y)
-   Conditionals: P(Y∣X),P(X∣Y)P(Y|X), P(X|Y)
-   Expected values: E\[X\],E\[Y\],E\[XY\]E\[X\], E\[Y\], E\[XY\]
-   Covariance: Cov(X,Y)\=E\[XY\]−E\[X\]E\[Y\]Cov(X,Y) = E\[XY\] - E\[X\]E\[Y\]
-   Independence check: Does P(X,Y)\=P(X)P(Y)P(X,Y) = P(X)P(Y)?

## Marginal Distribution P(X)

**Marginalization** means "summing out" the variables you do not care about. The result is written in the "margins" of a table.

#### The Sum Rule

P(X\=x)\=∑yP(X\=x,Y\=y)P(X=x) = \\sum\_y P(X=x, Y=y)

To find P(Young), add P(Young, Latte) + P(Young, Espresso).

#### Why is it called "marginal"?

In old-school contingency tables, these sums were written in the margins of the paper. The term stuck.

## Interactive: Marginalization

Step through to see how summing rows or columns gives us marginal distributions.

#### The Joint Distribution P(X, Y)

This 3x3 grid contains all information. Each cell is the probability of a specific (X, Y) pair occurring.

Y0

Y1

Y2

X0

X1

X2

0.15

0.10

0.05

0.10

0.20

0.10

0.05

0.10

0.15

## Conditional Distribution P(Y|X)

Conditioning is "slicing" the joint distribution and renormalizing so probabilities sum to 1 in that slice.

P(Y∣X)\=P(X,Y)P(X)P(Y|X) = \\frac{P(X, Y)}{P(X)}

Joint probability divided by marginal probability.

#### Example

P(Espresso∣Young)\=P(Young,Espresso)P(Young)\=0.100.50\=P(\\text{Espresso} | \\text{Young}) = \\frac{P(\\text{Young}, \\text{Espresso})}{P(\\text{Young})} = \\frac{0.10}{0.50} = **0.20**

#### Why normalize?

So that P(Latte∣Young)+P(Espresso∣Young)\=1P(\\text{Latte}|\\text{Young}) + P(\\text{Espresso}|\\text{Young}) = 1. Probabilities in a slice must sum to 1.

For details on conditional probability, see [Conditional Probability](https://www.tensortonic.com/ml-math/probability/conditional-probability).

## Product Rule & Chain Rule

Rearranging the conditional definition gives us the **Product Rule**:

P(X,Y)\=P(Y∣X)⋅P(X)P(X, Y) = P(Y | X) \\cdot P(X)

Joint = Conditional x Marginal

Generalized to n variables, this becomes the **Chain Rule**:

P(X1,X2,...,Xn)\=P(X1)⋅P(X2∣X1)⋅P(X3∣X1,X2)⋯P(Xn∣X1,...,Xn−1)P(X\_1, X\_2, ..., X\_n) = P(X\_1) \\cdot P(X\_2|X\_1) \\cdot P(X\_3|X\_1,X\_2) \\cdots P(X\_n|X\_1,...,X\_{n-1})

#### GPT Connection

Autoregressive language models like GPT compute P(next word | all previous words) using the chain rule. Each token is conditioned on everything before it.

## ML Applications

#### Discriminative Models

Model the **conditional** P(Y∣X)P(Y|X) directly.

P(label∣features)P(\\text{label} | \\text{features})

Examples: Logistic Regression, Neural Networks, SVMs

#### Generative Models

Model the **joint** P(X,Y)\=P(X∣Y)P(Y)P(X, Y) = P(X|Y)P(Y).

Can generate new samples!

Examples: Naive Bayes, GANs, VAEs, Diffusion Models

#### [Bayesian Inference](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist)

Posterior = (Likelihood x Prior) / Evidence

P(θ∣D)∝P(D∣θ)P(θ)P(\\theta|D) \\propto P(D|\\theta)P(\\theta)

The denominator P(D)P(D) is a marginal (intractable integral).

#### Latent Variable Models

Marginalize over hidden variables ZZ:

P(X)\=∑ZP(X,Z)P(X) = \\sum\_Z P(X, Z)

Examples: VAEs, Mixture Models, HMMs