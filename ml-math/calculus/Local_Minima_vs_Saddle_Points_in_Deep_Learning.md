**Title:** Local Minima vs Saddle Points in Deep Learning

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/local-vs-saddle](https://www.tensortonic.com/ml-math/calculus/local-vs-saddle)

---

# Page Structure Map
```text
Local Minima vs Saddle Points in Deep Learning
├── Introduction
│   └── The Key Insight
├── Definitions
│   ├── Local Minimum
│   ├── Saddle Point
│   └── Critical Points
├── The Hessian Test
├── Interactive: Saddle Point
├── The High-Dimensional Curse
│   ├── The Probability Argument
│   └── The Reality
├── Why Saddle Points Cause Problems
│   └── Plateaus
├── Escaping Saddle Points
│   ├── Stochastic Gradient Descent (SGD)
│   ├── Momentum
│   ├── Adam Optimizer
│   └── Learning Rate Warmup
└── Implications for Deep Learning
    ├── Good News: Local Minima Are Often Good
    ├── Flat Minima Generalize Better
    ├── Over-Parameterization Helps
    └── Skip Connections
```

---

## Introduction

When training neural networks, the classic fear is getting stuck in a **local minimum**, a small valley that isn't the lowest point. However, groundbreaking research ([Dauphin et al., 2014](https://arxiv.org/abs/1406.2572)) revealed a surprising truth:

#### The Key Insight

In deep learning, local minima are rare. The vast majority of "stuck" points are **saddle points**.

Understanding this distinction explains why algorithms like [momentum and Adam](https://www.tensortonic.com/ml-math/optimization/momentum-nesterov) are necessary.

## Definitions

#### Local Minimum

A point where the function value is **lower** than all neighboring points.

Imagine standing at the bottom of a **bowl**. Every direction you step goes UP.

#### Saddle Point

A point where gradient = 0, but it's a minimum in some directions and maximum in others.

Imagine a **horse saddle**. Front-to-back goes UP. Side-to-side goes DOWN.

#### Critical Points

Both local minima and saddle points are "critical points" where the gradient equals zero. The difference lies in the **curvature** (second derivatives).

## The Hessian Test

At a critical point (where ∇f\=0\\nabla f = 0), we examine the [Hessian matrix](https://www.tensortonic.com/ml-math/calculus/jacobian-hessian) eigenvalues to classify the point:

| Eigenvalues | Curvature | Classification |
| --- | --- | --- |
| All λi\>0\\lambda\_i > 0 | Curves UP everywhere | Local Minimum |
| All λi<0\\lambda\_i < 0 | Curves DOWN everywhere | Local Maximum |
| Mixed signs | UP in some, DOWN in others | Saddle Point |

## Interactive: Saddle Point

Visualize a saddle surface. The function curves up in one direction (positive eigenvalue) and down in another (negative eigenvalue).

## The High-Dimensional Curse

Why are saddle points more common than local minima in deep learning? Pure probability.

#### The Probability Argument

To be a **local minimum** in n-dimensional space, the curvature must be positive in **all n directions**.

If each eigenvalue has roughly 50% chance of being positive or negative:

P(local minimum)\=0.5nP(\\text{local minimum}) = 0.5^n

For n\=100n = 100: P\=0.5100\=8×10−31P = 0.5^{100} = 8 \\times 10^{-31}

#### The Reality

Modern neural networks have millions or billions of parameters. The probability of ALL eigenvalues being positive (true minimum) is essentially zero. Almost every critical point is a saddle point with some positive and some negative eigenvalues.

## Why Saddle Points Cause Problems

At a saddle point, the gradient is zero (or very small). Standard gradient descent relies on the gradient to update weights:

θnew\=θold−α⋅0\=θold\\theta\_{new} = \\theta\_{old} - \\alpha \\cdot 0 = \\theta\_{old}

If gradient is zero, parameters don't move. The model thinks it has converged.

#### Plateaus

Near saddle points, gradients are very small (not exactly zero), creating "plateaus" where learning stagnates. The loss appears flat even though there are directions that would decrease it.

## Escaping Saddle Points

Since vanilla gradient descent gets stuck, we use techniques that add "momentum" or "noise":

#### Stochastic Gradient Descent (SGD)

The noise from random mini-batches "kicks" the parameters. Even if the _average_ gradient is zero, individual batch gradients are likely non-zero in the escape direction.

#### Momentum

Like a ball rolling downhill. If it enters a flat saddle region, its existing velocity carries it across the plateau.

vt\=βvt−1+∇Jv\_t = \\beta v\_{t-1} + \\nabla J

#### Adam Optimizer

Combines momentum with adaptive learning rates. Uses both first moment (velocity) and second moment (variance) of gradients. The default choice for most deep learning.

#### Learning Rate Warmup

Start with a small learning rate, gradually increase. Helps navigate the initial chaotic landscape before settling into meaningful optimization.

## Implications for Deep Learning

#### Good News: Local Minima Are Often Good

Research shows that in over-parameterized networks, most local minima have loss values very close to the global minimum. Getting "stuck" in a local minimum isn't as catastrophic as feared.

#### Flat Minima Generalize Better

Minima with small Hessian eigenvalues ("flat" regions) tend to generalize better than "sharp" minima. This is related to PAC-Bayesian theory and explains why SGD noise helps.

#### Over-Parameterization Helps

Having more parameters than needed creates many equivalent good solutions, making optimization easier. This is one reason why bigger models are often easier to train.

#### Skip Connections

ResNets with skip connections create a smoother loss landscape with fewer saddle points. This is why very deep networks became trainable with residual connections.