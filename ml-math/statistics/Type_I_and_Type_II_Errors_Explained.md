**Title:** Type I and Type II Errors Explained

**Author:** TensorTonic Team

**Source:** [https://www.tensortonic.com/ml-math/statistics/type-one-vs-two](https://www.tensortonic.com/ml-math/statistics/type-one-vs-two)

---

# Page Structure Map
```text
Type I and Type II Errors Explained
├── Introduction
│   └── The Fundamental Question
├── Core Definitions
│   ├── Null Hypothesis
│   └── Alternative Hypothesis
├── The Decision Matrix
├── Deep Dive: Type I Error (False Positive)
│   └── Definition
│       ├── The Probability: Alpha (α\\alpha)
│       ├── The Consequence
│       └── Analogy: The False Alarm
├── Deep Dive: Type II Error (False Negative)
│   └── Definition
│       ├── The Probability: Beta (β\\beta)
│       ├── The Consequence
│       └── Analogy: The Silent Fire
├── Interactive Demo: The Alpha-Beta Trade-off
│   └── Type I & Type II Errors Visualized
├── Statistical Power (1−β1 - \\beta)
│   └── Factors that Increase Power:
├── Interactive Demo: Power Analysis Calculator
│   └── Power Analysis: Sample Size Calculator
├── Real-World Scenarios
│   ├── Criminal Trial
│   ├── Medical Testing
│   └── Spam Filter (ML)
└── Interactive Demo: Which Error is Worse?
    └── Which Error is Worse? Context Matters!
```

---

## Introduction

**Prerequisite:** This chapter builds on concepts fromHypothesis Testing. Make sure you understand Null/Alternative hypotheses and P-values before continuing.

In Hypothesis Testing, we never know the absolute truth about a population; we only make inferences based on a sample. Because we are operating with incomplete information, our decisions are subject to uncertainty.

### The Fundamental Question

Whenever you make a binary decision (Reject H0H\_0 or Fail to Reject H0H\_0) based on probability, there are **two ways to be right**, and **two ways to be wrong**.

"Which mistake is worse: raising a false alarm, or missing a real discovery?"

Understanding **Type I** and **Type II** errors is arguably the most important practical skill in A/B testing, medical diagnosis, and machine learning model evaluation.

## Core Definitions

To understand errors, we must first recall the two competing hypotheses in any test:

H0H\_0

#### Null Hypothesis

The default assumption. "There is no effect," "The drug does nothing," or "The defendant is innocent."

H1H\_1

#### Alternative Hypothesis

The claim we want to detect. "There is an effect," "The drug works," or "The defendant is guilty."

## The Decision Matrix

This 2x2 matrix is the mental model you must memorize. It maps "Reality" (which we do not know) against "Our Decision" (based on data).

|  | 
H0H\_0 is TRUE

(Nothing happened)

 | 

H0H\_0 is FALSE

(Real effect exists)

 |
| --- | --- | --- |
| 

Fail to Reject H0H\_0

(Do nothing)

 | Correct  
(True Negative) | Type II Error  
(False Negative)  
"Missed Opportunity" |
| 

Reject H0H\_0

(Take action)

 | Type I Error  
(False Positive)  
"False Alarm" | Correct  
(True Positive)  
Power = 1 - beta |

**Memory Trick:** Type **I** = **I**ncorrectly reject true null (False **P**ositive). Type I**I** = **II**ncorrectly fail to reject false null (False **N**egative).

## Deep Dive: Type I Error (False Positive)

### Definition

Rejecting a **True** Null Hypothesis. You conclude there is an effect when there is none.

#### The Probability: Alpha (α\\alpha)

The probability of committing a Type I error is exactly equal to the **Significance Level** you choose before the test.

α\=0.05\\alpha = 0.05

5% risk of false positive

#### The Consequence

This error usually leads to taking an action that should not have been taken:

-   \- Prescribing a drug that does not work
-   \- Launching a feature that does not increase revenue
-   \- Publishing a "discovery" that is not real

#### Analogy: The False Alarm

The smoke detector goes off (Reject H0H\_0) but there is no fire (Null is True). You panic and evacuate for no reason.

## Deep Dive: Type II Error (False Negative)

### Definition

Failing to Reject a **False** Null Hypothesis. You fail to detect an effect that actually exists.

#### The Probability: Beta (β\\beta)

Unlike Alpha, we do not set Beta directly. It depends on:

-   \- Sample size (bigger = lower beta)
-   \- Effect size (bigger = lower beta)
-   \- Variance /noise (lower = lower beta)
-   \- Alpha level (higher = lower beta)

#### The Consequence

This is a missed opportunity:

-   \- Failing to treat a sick patient
-   \- Killing a project that would have been profitable
-   \- Missing a scientific breakthrough

#### Analogy: The Silent Fire

There is a fire (H0H\_0 is False), but the alarm stays silent (Fail to Reject). You burn down.

## Interactive Demo: The Alpha-Beta Trade-off

Here is the cruel reality of statistics: **You generally cannot minimize both errors simultaneously** without changing the sample size. Watch how adjusting alpha affects beta in this visualization.

### Type I & Type II Errors Visualized

Two distributions: H0 (null is true) and H1 (alternative is true). See how alpha, beta, and power interact.

Alpha (Type I Error): 0.05

Smaller alpha = harder to reject H0

Effect Size (True Difference): 2.0

Larger effect = easier to detect (more power)

Type I Error (alpha)

5.0%

False Positive

Type II Error (beta)

36.1%

False Negative

Power (1-beta)

63.9%

True Positive

Effect Size

2.0

Cohen's d

**Key Trade-off:** Decreasing alpha (stricter threshold) increases beta (miss more true effects). The only way to reduce BOTH errors is to increase sample size or study larger effects.

## Statistical Power (1−β1 - \\beta)

Power is the probability that a test correctly rejects a false null hypothesis. In simple terms, it is the ability of the test to detect an effect if one actually exists.

80%

Standard Target

Most A/B tests aim for 80% power. This means if there is a real difference, we have an 80% chance of finding it.

This implies we accept a 20% risk (β\=0.20\\beta = 0.20) of missing the effect (Type II Error).

### Factors that Increase Power:

Increase Sample Size

Narrows the sampling distributions, reducing overlap. This is the main lever for reducing BOTH errors.

Increase Alpha

Makes it easier to reject H0H\_0, increasing power. But also increases Type I error risk.

Larger Effect Size

Easier to detect a massive difference than a tiny one. Not always controllable.

## Interactive Demo: Power Analysis Calculator

How many samples do you need to achieve 80% power? Adjust the parameters to find out. This is a critical tool for experiment design.

### Power Analysis: Sample Size Calculator

See how sample size affects your ability to detect real effects. The curve shows power for different sample sizes.

Sample Size (n): 30

Effect Size (d): 0.50

SmallMediumLarge

Alpha: 0.05

Power

86.3%

Beta (Type II)

13.7%

Alpha (Type I)

5%

Need for 80% Power

n = 25

Your study has adequate power (86%). You have a good chance of detecting a real effect of size d=0.5.

## Real-World Scenarios

### Criminal Trial

H0H\_0: Innocent | H1H\_1: Guilty

**Type I:** Convict innocent

Society considers this very bad

**Type II:** Acquit guilty

Less worse in this context

### Medical Testing

H0H\_0: Healthy | H1H\_1: Has Disease

**Type I:** Diagnose healthy as sick

Unnecessary treatment. Manageable.

**Type II:** Miss actual disease

Patient gets worse. Critical!

### Spam Filter (ML)

H0H\_0: Legitimate Email | H1H\_1: Spam

**Type I:** Block important email

Miss job offer. Critical failure!

**Type II:** Spam reaches inbox

Annoyance. User deletes.

## Interactive Demo: Which Error is Worse?

The optimal threshold depends entirely on the relative costs of Type I vs Type II errors in your specific domain. Explore different scenarios:

### Which Error is Worse? Context Matters!

Select a scenario to see which type of error is more costly and how to optimize your test accordingly.

H0 (Null)

Defendant is innocent

H1 (Alternative)

Defendant is guilty

Type I Error (alpha)WORSE

Convict innocent person

Destroys life of innocent person. Irreversible harm.

HIGH SEVERITY

Type II Error (beta)

Acquit guilty person

Criminal goes free. Can potentially be retried.

MEDIUM SEVERITY

Recommended Strategy

Set very strict alpha (0.01). "Beyond reasonable doubt."

**Key Insight:** There is no universal "correct" alpha. The optimal threshold depends entirely on the relative costs of Type I vs Type II errors in your specific domain.