**Title:** Bayesian vs Frequentist Statistics

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist](https://www.tensortonic.com/ml-math/statistics/bayesian-frequentist)

---

# Page Structure Map
```text
Bayesian vs Frequentist Statistics
├── The Great Divide
├── Core Philosophies
│   ├── Frequentist
│   └── Bayesian
├── Intuition: The Lost Phone
│   └── Scenario: You hear your phone beeping somewhere in the house.
│       ├── Frequentist Approach
│       └── Bayesian Approach
├── The Mathematical Difference
│   ├── Frequentist: MLE
│   └── Bayesian: MAP
├── Interactive: Watch Beliefs Update
│   └── Bayesian Belief Update
│       └── What to notice:
├── Confidence Intervals vs. Credible Intervals
│   ├── 95% Confidence Interval
│   └── 95% Credible Interval
├── Interactive: Compare Interval Coverage
├── A/B Testing: The Practical Showdown
│   ├── Frequentist A/B Testing
│   └── Bayesian A/B Testing
├── ML Applications
│   ├── Regularization = Bayesian Prior
│   ├── Dropout = Approximate Bayesian Inference
│   ├── Gaussian Processes
│   └── Thompson Sampling (Bandits)
└── When to Use Which
```

---

## The Great Divide

Ask two statisticians "What is probability?" and you might start a war. This is not just a philosophical debate. It affects how we run experiments, how we train models, and how we interpret results.

**The Question:** Is a coin's probability of heads a fixed fact about the world, or a measure of our uncertainty?

## Core Philosophies

### Frequentist

"Data is random. Parameters are fixed."

There exists a TRUE value for any parameter (like the population mean). We never know it exactly, but it is a fixed constant. Probability is defined as the long-run frequency of events if we repeated the experiment infinitely.

**Tools:** P-values, Confidence Intervals, MLE

### Bayesian

"Data is fixed. Parameters are random."

We do not know the true parameter, so we describe it with a probability distribution representing our uncertainty. We start with a Prior belief and update it with data to get a Posterior belief. Probability measures our degree of belief.

**Tools:** Bayes Theorem, Priors, Posteriors, Credible Intervals

## Intuition: The Lost Phone

#### Scenario: You hear your phone beeping somewhere in the house.

##### Frequentist Approach

"I hear a beep from the kitchen direction. Based purely on the acoustic data, there is a 90% probability the phone is in the kitchen."

Only uses the current data.

##### Bayesian Approach

"I hear a beep from the kitchen. BUT I also know I leave my phone in the bedroom 99% of the time (my Prior). Even with this data, I should still check the bedroom first."

Combines data with prior knowledge.

Neither is "wrong." The Frequentist is strictly objective but ignores context. The Bayesian incorporates context but requires specifying beliefs upfront.

## The Mathematical Difference

The core difference is whether you use [Bayes' Theorem](https://www.tensortonic.com/ml-math/probability/bayes-theorm) to incorporate prior beliefs.

#### Frequentist: MLE

θ^\=arg⁡max⁡θP(D∣θ)\\hat{\\theta} = \\arg\\max\_{\\theta} P(D | \\theta)

Find the parameter that maximizes the probability of the observed data.

#### Bayesian: MAP

θ^\=arg⁡max⁡θP(θ∣D)\\hat{\\theta} = \\arg\\max\_{\\theta} P(\\theta | D)

Find the parameter with highest posterior probability.

P(θ∣D)∝P(D∣θ)⋅P(θ)P(\\theta|D) \\propto P(D|\\theta) \\cdot P(\\theta)

**Key insight:** If the prior P(θ)P(\\theta) is uniform (all values equally likely), then MAP = MLE. The prior is what differentiates Bayesian from Frequentist inference.

## Interactive: Watch Beliefs Update

This is the heart of Bayesian inference. Start with a prior belief (the purple dashed line), then flip the coin and watch the posterior distribution (green solid line) evolve. Compare it to the MLE estimate (orange).

### Bayesian Belief Update

Start with a prior belief about the coin's fairness. Then flip the coin and watch the posterior distribution update in real-time.

Prior Alpha: 2

Prior Beta: 2

Flips

0

Heads

0

Prior Mean

0.50

Posterior Mean

0.50

Notice how the posterior (green) starts at the prior (purple dashed) and shifts toward the MLE (orange) as you add data.

#### What to notice:

-   With few flips, the posterior stays close to the prior (prior dominates)
-   With many flips, the posterior converges to the MLE (data dominates)
-   A strong prior (high alpha + beta) resists change more than a weak prior

## Confidence Intervals vs. Credible Intervals

Both give you a range of plausible values, but they answer fundamentally different questions.

#### 95% Confidence Interval

"If we repeated this experiment many times, 95% of the resulting intervals would contain the true parameter."

**NOT:** "There is a 95% chance the true value is in this interval."

The true value is fixed. It is either in the interval or not. The 95% refers to the long-run coverage of the procedure.

#### 95% Credible Interval

"Given the data and my prior, there is a 95% probability the true parameter lies in this interval."

**This IS** what most people intuitively want!

The parameter is treated as random (uncertain), so we can assign direct probability statements about it.

## Interactive: Compare Interval Coverage

Run many simulations and see how often each type of interval captures the true parameter. In theory, both should achieve about 95% coverage.

## A/B Testing: The Practical Showdown

This is where the philosophical difference has real business impact.

#### Frequentist A/B Testing

-   **Pre-register** sample size before starting
-   Run test until sample size is reached
-   Calculate p-value at the end
-   **Binary outcome:** Significant or Not Significant

**Warning:** You cannot "peek" at results early. Early stopping inflates false positive rates (the Peeking Problem).

#### Bayesian A/B Testing

-   Start with a prior on conversion rates
-   Update posterior after each observation
-   **Continuous output:** "92% probability B is better than A"
-   Can stop early when posterior is decisive

**Advantage:** Natural interpretation. No peeking problem. Can stop when confident.

## ML Applications

#### Regularization = Bayesian Prior

L2 regularization (Ridge) is equivalent to placing a Gaussian prior on weights. L1 regularization (Lasso) is equivalent to a Laplace prior. When you add regularization, you are being Bayesian!

MAP\=arg⁡max⁡θ\[log⁡P(D∣θ)+log⁡P(θ)\]\\text{MAP} = \\arg\\max\_{\\theta} \[\\log P(D|\\theta) + \\log P(\\theta)\]

#### Dropout = Approximate Bayesian Inference

Dropout during training can be viewed as approximating a Bayesian neural network. Running inference with dropout enabled ("MC Dropout") gives you uncertainty estimates.

#### Gaussian Processes

GPs are fully Bayesian models that provide uncertainty quantification out of the box. Great for small data scenarios or when you need reliable confidence bounds.

#### Thompson Sampling (Bandits)

A Bayesian approach to the explore-exploit tradeoff. Sample from the posterior distribution of rewards and pick the arm with highest sampled value.

## When to Use Which

| Scenario | Recommendation | Why |
| --- | --- | --- |
| Large dataset, need objectivity | Frequentist | Prior becomes irrelevant; simpler to communicate |
| Small dataset, have domain knowledge | Bayesian | Prior stabilizes estimates; prevents overfitting |
| Academic publication | Frequentist | P-values are the standard; reviewers expect them |
| Real-time decision making | Bayesian | Can update beliefs continuously; no fixed sample size |
| Need uncertainty quantification | Bayesian | Provides natural probability distributions over predictions |
| Computational constraints | Frequentist | Closed-form solutions; no [MCMC](https://www.tensortonic.com/ml-math/probability/markov-chains) needed |

**Modern Reality:** Most practitioners use both. Use Frequentist methods for standard hypothesis tests and Bayesian methods when you need uncertainty estimates or have strong priors. The "war" is largely academic. Use what works.