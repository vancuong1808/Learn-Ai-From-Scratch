**Title:** Conditional Probability Explained

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/probability/conditional-probability](https://www.tensortonic.com/ml-math/probability/conditional-probability)

---

# Page Structure Map
```text
Conditional Probability Explained
├── Introduction
│   └── Simple Example: Rolling a Die
├── The Formal Definition
│   ├── P(E∩F)P(E \\cap F)
│   ├── P(F)P(F)
│   └── Why divide?
├── Interactive: Visualizing Conditional Probability
│   └── Interactive Conditional Probability
│       └── Analysis: Independent Events
├── Worked Examples
│   ├── Movie Recommendations
│   └── Family Composition
├── Independence
│   ├── Testing Independence
│   └── Independence ≠ Mutual Exclusivity
├── The Chain Rule (Multiplication Rule)
│   ├── Example: Drawing Marbles
│   └── Chain Rule in Language Models
├── Interactive: Chain Rule with Probability Trees
│   └── Interactive Chain Rule
├── Connection to Bayes' Theorem
│   └── Medical Diagnosis Example
├── ML Applications
│   ├── Naive Bayes
│   ├── Hidden Markov Models
│   ├── Bayesian Networks
│   └── Reinforcement Learning
└── Common Pitfalls
    ├── Confusing P(A|B) with P(B|A)
    ├── Assuming Independence
    ├── Ignoring Base Rates
    └── Wrong Chain Rule
```

---

## Introduction

We constantly update our beliefs based on new information. If it is cloudy, you revise your estimate of the probability of rain. If a medical test is positive, a doctor updates their assessment of whether a patient has a disease.

**Conditional Probability** answers: "What is the chance of E happening _given that_ F has already happened?"

Written as P(E∣F)P(E|F), read "the probability of E given F"

#### Simple Example: Rolling a Die

Roll a fair six-sided die. What is P(rolled a 5)?

Answer: P(5)\=1/6P(5) = 1/6 since there are 6 equally likely outcomes.

Now someone tells you _the roll was odd_. What is P(5 | odd)?

The possible outcomes are now {1, 3, 5}. Only one of these three is a 5.

P(5∣odd)\=1/3P(5 | \\text{odd}) = 1/3

The probability jumped from 1/6 to 1/3 because conditioning eliminated half the sample space.

## The Formal Definition

P(E∣F)\=P(E∩F)P(F)P(E|F) = \\frac{P(E \\cap F)}{P(F)}

provided that P(F)\>0P(F) > 0

#### P(E∩F)P(E \\cap F)

Probability that both E and F occur (the intersection)

#### P(F)P(F)

Probability that F occurs (the new "universe")

#### Why divide?

We rescale probabilities to the new sample space F

## Interactive: Visualizing Conditional Probability

Adjust the probabilities and see how P(E|F) and P(F|E) change. Notice that they are generally NOT equal!

### Interactive Conditional Probability

Adjust the probabilities and see how P(E|F) changes. The shaded intersection represents P(E and F).

P(E): 0.40

P(F): 0.50

P(E∩F): 0.20

P(E|F)

0.400

\= 0.20 / 0.50

P(F|E)

0.500

\= 0.20 / 0.40

Independence

YES

P(E)P(F) = 0.20

P(E|F) asks: "Of the green area (F), what fraction is orange (intersection)?"

#### Analysis: Independent Events

Knowing F occurred gives us **no new information** about E. The probability stays the same (P(E|F) ≈ P(E)).

## Worked Examples

### Movie Recommendations

Netflix wants to predict if a user will watch "Life is Beautiful" (E), given they watched "Amelie" (F).

Overall

P(E) = 0.02 (2% of all users)

Given Amelie

P(E|F) = 0.09 (9%)

Watching Amelie makes a user **4.5x more likely** to watch Life is Beautiful. This is why recommendation systems track conditional probabilities.

### Family Composition

A family has 4 children. Given that they have at least one boy, what is P(exactly 2 boys)?

Let E = "exactly 2 boys", F = "at least 1 boy"

All 16 sequences (BBBB, BBBG, ..., GGGG) are equally likely.

F eliminates only GGGG, leaving 15 outcomes.

Among these 15, exactly 6 have two boys.

P(E∣F)\=6/1615/16\=615\=0.4P(E|F) = \\frac{6/16}{15/16} = \\frac{6}{15} = 0.4

## Independence

Two events are **independent** if knowing one occurred does not change the probability of the other.

Definition 1

P(E∣F)\=P(E)P(E|F) = P(E)

Definition 2 (Equivalent)

P(E∩F)\=P(E)⋅P(F)P(E \\cap F) = P(E) \\cdot P(F)

#### Testing Independence

Roll a die. Let A = {3} and B = {1, 3, 5} (odd numbers).

P(A) = 1/6

P(B) = 3/6 = 1/2

P(A ∩ B) = P({3}) = 1/6

Check: P(A) x P(B) = (1/6)(1/2) = 1/12

Since 1/12 ≠ 1/6, the events are NOT independent.

#### Independence ≠ Mutual Exclusivity

**Mutually exclusive:** P(E ∩ F) = 0 (cannot both occur)

**Independent:** P(E ∩ F) = P(E)P(F) (one does not affect the other)

If events are mutually exclusive with nonzero probabilities, they are **never** independent. Knowing A occurred tells you B definitely did NOT occur.

## The Chain Rule (Multiplication Rule)

Rearranging the conditional probability formula gives us the **Chain Rule**:

P(E∩F)\=P(F)⋅P(E∣F)P(E \\cap F) = P(F) \\cdot P(E|F)

or equivalently

P(E∩F)\=P(E)⋅P(F∣E)P(E \\cap F) = P(E) \\cdot P(F|E)

This extends to any number of events. For three events:

P(A∩B∩C)\=P(A)⋅P(B∣A)⋅P(C∣A,B)P(A \\cap B \\cap C) = P(A) \\cdot P(B|A) \\cdot P(C|A,B)

#### Example: Drawing Marbles

A jar has 7 black and 3 white marbles. Draw two without replacement. What is P(both black)?

P(B1) = 7/10

P(B2|B1) = 6/9 (after removing one black, 6 black remain out of 9)

P(B1 ∩ B2) = (7/10)(6/9) = 42/90 = **0.467**

#### Chain Rule in Language Models

The chain rule is the foundation of autoregressive models like GPT:

P(w1,w2,…,wn)\=P(w1)⋅P(w2∣w1)⋅P(w3∣w1,w2)⋯P(w\_1, w\_2, \\ldots, w\_n) = P(w\_1) \\cdot P(w\_2|w\_1) \\cdot P(w\_3|w\_1,w\_2) \\cdots

GPT learns to estimate each P(next word | previous words), then multiplies along the chain.

## Interactive: Chain Rule with Probability Trees

Probability trees visualize the chain rule. Each path from root to leaf represents a sequence of events. Multiply along the path to get the joint probability.

### Interactive Chain Rule

Adjust the sliders to explore how probabilities propagate through the chain.

P(A)0.30

P(B | A)0.80

P(B | ~A)0.20

## Connection to Bayes' Theorem

Since P(A ∩ B) = P(B ∩ A), we can equate the two chain rule forms:

P(A∣B)⋅P(B)\=P(B∣A)⋅P(A)P(A|B) \\cdot P(B) = P(B|A) \\cdot P(A)

Rearranging gives **Bayes' Theorem**:

P(A∣B)\=P(B∣A)⋅P(A)P(B)P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}

Prior

P(A)

Likelihood

P(B|A)

Posterior

P(A|B)

Evidence

P(B)

#### Medical Diagnosis Example

A rare disease affects 0.1% of people. A test has 92% sensitivity and 89% specificity.

If someone tests positive, what is P(disease | positive)?

P(+) = P(+|D)P(D) + P(+|D')P(D') = (0.92)(0.001) + (0.11)(0.999) = 0.111

P(D|+) = (0.92 x 0.001) / 0.111 = **0.0083 (0.83%)**

Despite a positive test, only 0.83% chance of having the disease. The low base rate (0.1%) dominates.

For more on Bayesian reasoning, see [Bayes' Theorem](https://www.tensortonic.com/ml-math/probability/bayes-theorm).

## ML Applications

### Naive Bayes

Assumes features are conditionally independent given class:

P(C∣X)∝P(C)∏iP(Xi∣C)P(C|X) \\propto P(C) \\prod\_i P(X\_i|C)

Used in spam filtering, sentiment analysis

### Hidden Markov Models

Model sequences with hidden states:

Transition: P(s\_t | s\_{t-1})

Emission: P(o\_t | s\_t)

Speech recognition, POS tagging

### Bayesian Networks

Graphical models encoding conditional independence:

P(X1,…,Xn)\=∏iP(Xi∣ extparents(Xi))P(X\_1, \\ldots, X\_n) = \\prod\_i P(X\_i | \\ ext{parents}(X\_i))

Medical diagnosis, causal reasoning

### Reinforcement Learning

Policies are conditional distributions:

π(a∣s)\=P( extaction\=a∣ extstate\=s)\\pi(a|s) = P(\\ ext{action}=a | \\ ext{state}=s)

Game playing, robotics

## Common Pitfalls

#### Confusing P(A|B) with P(B|A)

The "Prosecutor's Fallacy." P(test+ | disease) = 99% does NOT mean P(disease | test+) = 99%.

The posterior depends critically on the base rate (prior).

#### Assuming Independence

Just because events "seem" unrelated does not mean they are statistically independent. Always verify: does P(A ∩ B) = P(A)P(B)?

#### Ignoring Base Rates

Even with strong evidence (high likelihood), a very low prior keeps the posterior low. This is why screening tests for rare diseases produce many false positives.

#### Wrong Chain Rule

Correct

P(A)P(B|A)P(C|A,B)

Wrong

P(A)P(B|A)P(C|B)

Each term must condition on ALL previous events.