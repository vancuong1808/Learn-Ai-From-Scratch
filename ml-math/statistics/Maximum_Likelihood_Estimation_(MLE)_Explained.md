**Title:** Maximum Likelihood Estimation (MLE) Explained

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/maximum-likelihood-estimation](https://www.tensortonic.com/ml-math/statistics/maximum-likelihood-estimation)

---

# Page Structure Map
```text
Maximum Likelihood Estimation (MLE) Explained
├── Introduction
│   └── The Core Question
├── The Big Idea: A Simple Example
│   └── The Bag of Balls
├── Likelihood vs. Probability
│   ├── Probability P(x∣θ)P(x | \\theta)
│   ├── Likelihood L(θ∣x)L(\\theta | x)
│   └── Key Insight
├── Interactive: Visualizing the Likelihood Function
│   └── Interactive Likelihood Function
├── The Log-Likelihood Trick
│   ├── Problem 1: Underflow
│   └── Problem 2: Derivatives
├── Worked Example: Biased Coin
│   ├── Step 1: Write the Likelihood
│   ├── Step 2: Take the Log
│   ├── Step 3: Differentiate
│   └── Step 4: Set to Zero and Solve
├── Worked Example: Normal Distribution
│   ├── Step 1: Likelihood (PDF Product)
│   ├── Step 2: Log-Likelihood
│   ├── Step 3: Differentiate w.r.t. μ\\mu
│   └── Step 4: Solve
├── Why MLE Works: Key Properties
│   ├── 1\. Consistency
│   ├── 2\. Efficiency
│   ├── 3\. Invariance
│   └── 4\. Asymptotic Normality
├── Interactive: MLE Convergence
├── MLE in Machine Learning
│   ├── MSE Loss = MLE for Gaussian
│   ├── Cross-Entropy Loss = MLE for Bernoulli
│   ├── Softmax + Cross-Entropy = MLE for Categorical
│   └── Bottom Line
└── Limitations and the Bayesian Fix
    ├── The "Zero Count" Problem
    └── The Fix: MAP (Maximum A Posteriori)
```

---

## Introduction

Throughout statistics, we often assume we know the population parameters (like mean μ\\mu or variance σ2\\sigma^2). But in the real world, **we never know these values**. We only have data.

#### The Core Question

Given some observed data, what are the most reasonable values for the unknown parameters?

**Maximum Likelihood Estimation (MLE)** answers this by asking: _"Which parameters would have made our observed data most probable?"_ It is the foundation of most ML loss functions.

## The Big Idea: A Simple Example

### The Bag of Balls

Imagine a bag with 3 balls. Each ball is either Red or Blue, but you do not know the combination. Let θ\\theta = number of Blue balls. Possible values: 0, 1, 2, or 3.

**Experiment:** You draw 4 balls _with replacement_ and observe: Blue, Red, Blue, Blue

Which hypothesis about θ\\theta makes this outcome most probable?

θ\=0\\theta = 0P(Blue) = 0P(data) = 0 (impossible)

θ\=1\\theta = 1P(Blue) = 1/3P(data) = 2/81 = 0.025

θ\=2\\theta = 2P(Blue) = 2/3P(data) = 8/81 = 0.099 (highest!)

θ\=3\\theta = 3P(Red) = 0P(data) = 0 (impossible)

MLE chooses θ\=2\\theta = 2 because it maximizes the probability of observing the data.

## Likelihood vs. Probability

These terms sound interchangeable but have **opposite meanings** in statistics.

#### Probability P(x∣θ)P(x | \\theta)

**Fix the parameter**, ask about data.

"If the coin is fair (θ\=0.5\\theta = 0.5), what is the probability of getting 7 heads in 10 flips?"

#### Likelihood L(θ∣x)L(\\theta | x)

**Fix the data**, ask about parameters.

"I observed 7 heads in 10 flips. How likely is it that θ\=0.5\\theta = 0.5? How about θ\=0.7\\theta = 0.7?"

#### Key Insight

Likelihood is NOT a probability distribution over θ\\theta. It does not sum to 1 across all theta values. It is simply a function that tells us how "compatible" each theta is with our observed data.

## Interactive: Visualizing the Likelihood Function

For the coin flip example, the likelihood function is L(p)\=pk(1−p)n−kL(p) = p^k (1-p)^{n-k} where k = heads observed and n = total flips. Adjust the sliders to see how the likelihood curve changes.

### Interactive Likelihood Function

Adjust the observed coin flips and watch the likelihood function change. The peak shows the MLE estimate.

Heads observed: 7

Total flips: 10

Observed Data

7H, 3T

MLE Estimate

p = 0.700

Max Likelihood

2.22e-3

The curve shows L(p) = p^7 \* (1-p)^3. The peak at p = 0.70 is the Maximum Likelihood Estimate.

## The Log-Likelihood Trick

With many observations, the likelihood becomes a **product of many small numbers**. This causes two problems:

#### Problem 1: Underflow

0.01×0.01×0.01×...\=10−2000.01 \\times 0.01 \\times 0.01 \\times ... = 10^{-200} which computers round to 0.

#### Problem 2: Derivatives

Taking derivatives of products requires the complex Product Rule repeatedly.

**Solution:** Take the natural log. Since log is monotonically increasing, the theta that maximizes L(theta) also maximizes log L(theta).

Log-Likelihood

ℓ(θ)\=ln⁡L(θ)\=∑i\=1nln⁡f(xi;θ)\\ell(\\theta) = \\ln L(\\theta) = \\sum\_{i=1}^n \\ln f(x\_i; \\theta)

Products become sums. Sums are easy to differentiate.

## Worked Example: Biased Coin

Observe n coin flips with k heads. What is the MLE for p (probability of heads)?

##### Step 1: Write the Likelihood

L(p)\=pk(1−p)n−kL(p) = p^k (1-p)^{n-k}

##### Step 2: Take the Log

ℓ(p)\=kln⁡(p)+(n−k)ln⁡(1−p)\\ell(p) = k \\ln(p) + (n-k) \\ln(1-p)

##### Step 3: Differentiate

dℓdp\=kp−n−k1−p\\frac{d\\ell}{dp} = \\frac{k}{p} - \\frac{n-k}{1-p}

##### Step 4: Set to Zero and Solve

kp\=n−k1−p⇒k(1−p)\=p(n−k)⇒k\=np\\frac{k}{p} = \\frac{n-k}{1-p} \\quad \\Rightarrow \\quad k(1-p) = p(n-k) \\quad \\Rightarrow \\quad k = np

p^MLE\=kn\\hat{p}\_{MLE} = \\frac{k}{n}

The MLE for p is simply the observed proportion of heads. Intuitive!

## Worked Example: Normal Distribution

Data x1,…,xnx\_1, \\dots, x\_n from N(μ,σ2)N(\\mu, \\sigma^2). Estimate mu.

##### Step 1: Likelihood (PDF Product)

L(μ)\=∏i\=1n12πσ2exp⁡(−(xi−μ)22σ2)L(\\mu) = \\prod\_{i=1}^n \\frac{1}{\\sqrt{2\\pi\\sigma^2}} \\exp\\left(-\\frac{(x\_i - \\mu)^2}{2\\sigma^2}\\right)

##### Step 2: Log-Likelihood

ℓ(μ)\=−n2ln⁡(2πσ2)−∑i\=1n(xi−μ)22σ2\\ell(\\mu) = -\\frac{n}{2}\\ln(2\\pi\\sigma^2) - \\sum\_{i=1}^n \\frac{(x\_i - \\mu)^2}{2\\sigma^2}

##### Step 3: Differentiate w.r.t. μ\\mu

First term is constant w.r.t. μ\\mu, disappears.

dℓdμ\=1σ2∑(xi−μ)\\frac{d\\ell}{d\\mu} = \\frac{1}{\\sigma^2} \\sum (x\_i - \\mu)

##### Step 4: Solve

∑(xi−μ)\=0⇒∑xi\=nμ\\sum (x\_i - \\mu) = 0 \\quad \\Rightarrow \\quad \\sum x\_i = n\\mu

μ^MLE\=1n∑xi\=xˉ\\hat{\\mu}\_{MLE} = \\frac{1}{n} \\sum x\_i = \\bar{x}

The MLE for the mean is the sample mean!

## Why MLE Works: Key Properties

For large datasets, MLE is theoretically optimal. Here is why:

#### 1\. Consistency

As n approaches infinity, the estimate converges to the true parameter. More data = more accurate.

#### 2\. Efficiency

No other unbiased estimator has lower variance. MLE extracts maximum information from the data.

#### 3\. Invariance

If θ^\\hat{\\theta} is MLE for θ\\theta, then g(θ^)g(\\hat{\\theta}) is MLE for g(θ)g(\\theta). Transformations are easy.

#### 4\. Asymptotic Normality

For large n, MLE follows a Normal distribution. This allows easy confidence interval construction.

## Interactive: MLE Convergence

Watch how the MLE estimate converges to the true parameter as you increase sample size. This demonstrates the **Consistency** property in action.

## MLE in Machine Learning

MLE is not just a statistical concept. It is the foundation of most ML loss functions!

#### MSE Loss = MLE for Gaussian

If we assume errors are normally distributed, minimizing negative log-likelihood gives:

−ℓ(μ)∝∑(xi−μ)2\-\\ell(\\mu) \\propto \\sum (x\_i - \\mu)^2

This is exactly Mean Squared Error! Linear regression minimizes MSE because it assumes Gaussian noise.

#### Cross-Entropy Loss = MLE for Bernoulli

For binary classification with predicted probability p:

−ℓ(p)\=−\[yln⁡(p)+(1−y)ln⁡(1−p)\]\-\\ell(p) = -\[y \\ln(p) + (1-y)\\ln(1-p)\]

This is Binary Cross-Entropy! Logistic regression uses BCE because it models Bernoulli outcomes.

#### Softmax + Cross-Entropy = MLE for Categorical

For multi-class classification, minimizing categorical cross-entropy is equivalent to MLE assuming a Categorical distribution over classes.

#### Bottom Line

When you train a neural network by minimizing cross-entropy or MSE, you are performing MLE. The loss function encodes your assumption about the data distribution.

## Limitations and the Bayesian Fix

MLE is powerful but has a critical weakness with small data.

#### The "Zero Count" Problem

Flip a coin 3 times, get 3 heads. MLE says: P(heads) = 1.0

This means MLE concludes tails is **impossible**. Obviously wrong!

MLE overfits to small samples because it only considers the data, with no prior beliefs.

#### The Fix: MAP (Maximum A Posteriori)

MAP adds a **Prior** distribution encoding our beliefs before seeing data:

θ^MAP\=arg⁡max⁡θL(θ)⏟Likelihood⋅P(θ)⏟Prior\\hat{\\theta}\_{MAP} = \\arg\\max\_{\\theta} \\underbrace{L(\\theta)}\_{\\text{Likelihood}} \\cdot \\underbrace{P(\\theta)}\_{\\text{Prior}}

In ML, the Prior corresponds to **Regularization**:

-   L2 regularization = Gaussian prior on weights
-   L1 regularization = Laplace prior on weights

For a deeper dive, see [Bayesian vs. Frequentist](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist).