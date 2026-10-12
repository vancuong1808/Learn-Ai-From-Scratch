**Title:** A/B Testing: Statistical Guide for ML Engineers

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/ab-testing](https://www.tensortonic.com/ml-math/statistics/ab-testing)

---

# Page Structure Map
```text
A/B Testing: Statistical Guide for ML Engineers
├── What is A/B Testing?
│   └── The "Fair Test" Rule
├── Setting the Rules
│   ├── The Skeptic (H0H\_0)
│   └── The Discovery (H1H\_1)
├── [Type I & Type II Errors](https://www.tensortonic.com/ml-math/statistics/type-one-vs-two)
│   └── Business Impact Example: Light Bulb Packaging
├── The Math (Two-Sample ZZ\-Test)
│   └── Critical Z-Values Table
├── Power Analysis
│   └── [Statistical Power (1−β1 - \\beta)](https://www.tensortonic.com/ml-math/statistics/type-one-vs-two)
├── Sample Size Calculation
│   └── Numerical Example: Bulb Packaging
├── Case Study: Bulb Packaging Redesign
│   ├── The Scenario
│   ├── The Setup
│   └── The Results
├── Interactive Simulator
│   ├── Distribution Overlap
│   └── Convergence History
├── Common Pitfalls
│   ├── 1\. Peeking (P-Hacking)
│   ├── 2\. Novelty Effect
│   ├── 3\. Simpson's Paradox
│   └── 4\. Network Effects / Spillover
└── ML Applications
    ├── Model Deployment
    ├── Multi-Armed Bandits
    ├── Hyperparameter Tuning
    └── Causal ML
```

---

## What is A/B Testing?

Imagine you have an idea: "Changing this button from blue to green will get more clicks." How do you know if you're right?

You could just change it and see what happens. But maybe sales went up because it was a holiday, not because of the button. You'd never know for sure.

**A/B Testing** is the scientific way to find out. You split your users into two groups. Group A sees the old blue button. Group B sees the new green button. Since everything else is the same, if Group B clicks more, you _know_ the green button caused it.

### The "Fair Test" Rule

The key is [randomness](https://www.tensortonic.com/ml-math/probability/random-variables). You can't just show the new button to VIP users and the old one to new users. That's not a fair fight. By flipping a coin for every user, you ensure the two groups are practically identical before the test starts.

## Setting the Rules

Before we start, we need to agree on what we're testing. In statistics, we act like a judge in a courtroom. We assume the change does _nothing_ until the evidence is overwhelming.

### The Skeptic (H0H\_0)

Also called the **Null Hypothesis**. This is the assumption that your idea didn't work. "The green button is just as good (or bad) as the blue one."

Group A\=Group B\\text{Group A} = \\text{Group B}

### The Discovery (H1H\_1)

Also called the **Alternative Hypothesis**. This is what you hope to prove. "The green button is actually different."

Group A≠Group B\\text{Group A} \\neq \\text{Group B}

## [Type I & Type II Errors](https://www.tensortonic.com/ml-math/statistics/type-one-vs-two)

We can never be 100% sure. We deal in probabilities. There are two ways to be wrong, and they have real business consequences.

|  | Truth: H0H\_0 is True  
(No real effect) | Truth: H1H\_1 is True  
(Real effect exists) |
| --- | --- | --- |
| We Decide:  
Reject H0H\_0 | 
Type I Error

False Positive

We thought it worked, but it was luck.

Prob = α\\alpha

 | 

Correct Decision

True Positive

We found a real win!

Prob = 1−β1 - \\beta (Power)

 |
| We Decide:  
Keep H0H\_0 | 

Correct Decision

True Negative

We rightly ignored the noise.

 | 

Type II Error

False Negative

We missed a real opportunity.

Prob = β\\beta

 |

### Business Impact Example: Light Bulb Packaging

Type I Error (False Positive):

You roll out the new packaging to all stores based on a fluke result. The packaging actually doesn't help. You spent $50,000 on new printing plates for nothing.

Type II Error (False Negative):

The new packaging would have increased sales by 5%, but your test was too short/small to detect it. You abandon a great idea. You lose $100,000/year in potential revenue.

## The Math (Two-Sample ZZ\-Test)

How do we know if the difference between Group A and Group B is real? We calculate a **ZZ\-score**, which tells us how many standard deviations away our observed difference is from zero (no difference).

Z\=p^B−p^Ap^(1−p^)(1nA+1nB)Z = \\frac{\\hat{p}\_B - \\hat{p}\_A}{\\sqrt{\\hat{p}(1-\\hat{p})(\\frac{1}{n\_A} + \\frac{1}{n\_B})}}

p^A,p^B\\hat{p}\_A, \\hat{p}\_B: Conversion rates of A and B.

p^\\hat{p}: Pooled conversion rate (total conversions / total visitors).

nA,nBn\_A, n\_B: Sample sizes.

If ∣Z∣\>1.96|Z| > 1.96, we are 95% confident the difference is not due to chance ([p<0.05p < 0.05](https://www.tensortonic.com/ml-math/statistics/p-value)).

### Critical Z-Values Table

90% Confidence

Z = 1.645

95% Confidence

Z = 1.96

99% Confidence

Z = 2.576

## Power Analysis

Before starting, you must ask: "How many users do I need?" If you stop too early, you might miss a real effect (Type II Error). This is called being "underpowered."

### [Statistical Power (1−β1 - \\beta)](https://www.tensortonic.com/ml-math/statistics/type-one-vs-two)

The probability of detecting an effect _if it actually exists_. Industry standard is 80%, meaning you have an 80% chance of finding a true effect. Some critical tests (e.g., medical) require 90%.

Power is affected by four levers:

-   **Baseline Rate:** Current conversion rate (e.g., 5%). Higher baselines are easier to shift.
-   **MDE (Minimum Detectable Effect):** The smallest lift you care about (e.g., +1%). Detecting smaller lifts requires _much_ more data.
-   **Significance Level (α\\alpha):** Usually 5%. Lower α\\alpha (stricter) requires more data.
-   **Sample Size (nn):** Bigger nn = more power. This is the main lever you control.

## Sample Size Calculation

The formula for required sample size per group for a two-proportion Z-test is:

n\=2pˉ(1−pˉ)(Zα/2+Zβ)2(pB−pA)2n = \\frac{2\\bar{p}(1-\\bar{p})(Z\_{\\alpha/2} + Z\_\\beta)^2}{(p\_B - p\_A)^2}

### Numerical Example: Bulb Packaging

You want to test if a new packaging increases sales rate from 5% to 6% (1% lift).

-   Baseline (pAp\_A) = 5%
-   Expected (pBp\_B) = 6%
-   α\\alpha = 0.05, β\\beta = 0.20
-   Z0.025Z\_{0.025} = 1.96, Z0.20Z\_{0.20} = 0.84

Result: You need ~3,000 customers per group (6,000 total) to detect a 1% lift with 80% power.

## Case Study: Bulb Packaging Redesign

#### The Scenario

Your light bulb company has been selling bulbs in plain cardboard boxes. The marketing team proposes a premium-looking box with a window showing the bulb. They believe it will increase the purchase rate by 10%.

#### The Setup

-   **Control (A):** Existing cardboard box. Shown to 50% of store visitors randomly.
-   **Treatment (B):** New premium box. Shown to the other 50%.
-   **Metric:** Purchase rate (buyers / visitors).
-   **Duration:** 2 weeks (to account for weekend/weekday effects).

#### The Results

Control (A)

5.2%

520 / 10,000 visitors

Treatment (B)

5.9%

590 / 10,000 visitors

**Lift:** +0.7% absolute (+13.5% relative). **[P-value](https://www.tensortonic.com/ml-math/statistics/p-value):** 0.018. **Decision:** Statistically significant at α\=0.05\\alpha = 0.05. Roll out the new packaging!

## Interactive Simulator

Run a simulated A/B test. In this simulation, **Control (A)** represents your current baseline performance, while **Treatment (B)** has the added "True Lift". Watch how the p-value fluctuates early on (why you shouldn't peek!) and how it converges as sample size grows. This demonstrates why pre-determining your sample size is critical.

Baseline (A)

12%

Lift (B)

2.0%

P-Value

1.0000

Not Significant

#### Distribution Overlap

As N grows, curves narrow

Start simulation to generate data...

#### Convergence History

Avoid "Peeking" early!

## Common Pitfalls

### 1\. Peeking (P-Hacking)

Checking the results every day and stopping as soon as p<0.05p < 0.05. This inflates your False Positive rate massively. If you peek 10 times, your true α\\alpha is closer to 26%, not 5%.

**Solution:** Use Sequential Testing or decide sample size _before_ starting and stick to it.

### 2\. Novelty Effect

Users click the new bulb packaging just because it's new. The lift disappears after a week as they get used to it.

**Solution:** Run tests for at least 1-2 full business cycles (weeks). Analyze day-over-day trends.

### 3\. Simpson's Paradox

The new packaging looks better overall, but is worse for every single store. This happens if the traffic mix shifted (e.g., more visitors to high-performing stores in Treatment).

**Solution:** Always check segments (by store, user type, device).

### 4\. Network Effects / Spillover

If Treatment users tell Control users about the new packaging, the Control group is contaminated. This is a violation of SUTVA (Stable Unit Treatment Value Assumption).

**Solution:** Use cluster randomization (randomize by geography, not individual).

## ML Applications

### Model Deployment

You trained a new Recommender System. Offline metrics (RMSE) look great. But does it drive revenue? You A/B test Model A (Control) vs Model B (Treatment) on live traffic. The metric that matters is business revenue, not RMSE.

### Multi-Armed Bandits

A/B testing wastes traffic on the loser. Bandit algorithms (Thompson Sampling, ϵ\\epsilon\-Greedy) dynamically shift traffic to the winner _during_ the test. It's "Exploration vs Exploitation." Used by Google for ad optimization.

### Hyperparameter Tuning

Is learning rate 0.001 significantly better than 0.0001? You can run A/B-like comparisons in cross-validation folds to ensure your "best" hyperparameter is statistically significantly better, not just lucky on one fold.

### Causal ML

When A/B testing is impossible (ethics, cost), we use Causal Inference techniques (Propensity Score Matching, Instrumental Variables) to estimate treatment effects from observational data.