**Title:** Limits and Continuity: Foundations of Calculus for ML

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/limits-continuity](https://www.tensortonic.com/ml-math/calculus/limits-continuity)

---

# Page Structure Map
```text
Limits and Continuity: Foundations of Calculus for ML
├── Introduction
│   └── Why Limits Matter for ML
├── What is a Limit?
│   ├── Classic Example
│   ├── Left-hand Limit
│   ├── Right-hand Limit
│   └── Limit Exists If and Only If
├── Interactive: Approaching a Limit
├── The Epsilon-Delta Definition
│   ├── The Manufacturing Analogy
│   └── Formal Definition
├── Limit Laws
├── L'Hopital's Rule
│   ├── L'Hopital's Rule
│   ├── Example
│   └── Caution
├── Continuity
│   ├── Three Conditions for Continuity at c
│   └── Important Properties
├── Types of Discontinuity
│   ├── Removable
│   ├── Jump
│   └── Infinite
├── Continuity vs Differentiability
│   ├── The Hierarchy
│   ├── Differentiable implies Continuous
│   ├── Continuous does NOT imply Differentiable
│   └── The Classic Example: |x|
└── ML Applications
    ├── The Death of the Perceptron
    ├── ReLU: The Practical Compromise
    ├── Softmax and Numerical Stability
    └── Loss Function Continuity
```

---

## Introduction

Before we can discuss **derivatives** (the engine of optimization) or **integrals** (the foundation of probability), we must understand the bedrock they stand on: **limits**.

Calculus is the mathematics of change. But change happens at an instant, and measuring something "at an instant" involves dividing by zero (Δy0\\frac{\\Delta y}{0}). To bypass this mathematical impossibility, we use limits: asking not "what is the value **at** this point," but "what value do we **approach** as we get infinitely close?"

#### Why Limits Matter for ML

-   **Derivatives:** The definition of derivative is a limit: f′(x)\=lim⁡h→0f(x+h)−f(x)hf'(x) = \\lim\_{h \\to 0} \\frac{f(x+h) - f(x)}{h}
-   **Gradient Descent:** Requires continuous, differentiable functions. Limits ensure this.
-   **Activation Functions:** Understanding where functions break (discontinuities) explains ReLU vs Sigmoid choices.
-   **Convergence:** Training loops converge when loss approaches a limit.

## What is a Limit?

A limit describes the value a function **approaches** as the input approaches some value. The function doesn't need to actually reach that value, it just needs to get arbitrarily close.

#### Classic Example

Consider f(x)\=x2−1x−1f(x) = \\frac{x^2 - 1}{x - 1}. If we plug in x = 1, we get 00\\frac{0}{0}, which is undefined.

But we can factor: x2−1x−1\=(x+1)(x−1)x−1\=x+1\\frac{x^2-1}{x-1} = \\frac{(x+1)(x-1)}{x-1} = x + 1 for x != 1.

So as x approaches 1, f(x) approaches 2. The **limit exists** even though f(1) is undefined!

#### Left-hand Limit

lim⁡x→c−f(x)\\lim\_{x \\to c^-} f(x)

Approaching c from values less than c (from the left on number line).

#### Right-hand Limit

lim⁡x→c+f(x)\\lim\_{x \\to c^+} f(x)

Approaching c from values greater than c (from the right).

#### Limit Exists If and Only If

The limit lim⁡x→cf(x)\=L\\lim\_{x \\to c} f(x) = L exists if and only if both one-sided limits exist AND are equal:

lim⁡x→c−f(x)\=lim⁡x→c+f(x)\=L\\lim\_{x \\to c^-} f(x) = \\lim\_{x \\to c^+} f(x) = L

## Interactive: Approaching a Limit

Watch how both sides approach the same value as we get closer to the point. The function has a "hole" at x = 1, but the limit still exists!

## The Epsilon-Delta Definition

The formal definition of a limit is often considered one of the hardest concepts in Calculus 1. But it's actually just a way of describing a **guarantee of precision**.

#### The Manufacturing Analogy

Imagine you are manufacturing a high-precision piston (input xx) that must fit into a cylinder (output f(x)f(x)).

-   1.**The Goal (L):** The cylinder has a perfect target width, say 10cm.
-   2.**The Tolerance (ϵ\\epsilon):** The customer says, "The width must be within 0.01cm of the target." This is your error margin.
-   3.**The Input Precision (δ\\delta):** You ask, "How precise must my piston mold be?" Maybe if the mold is within 0.005cm of the target size, the final piston will be within the customer's tolerance.

**The Limit Exists If:** No matter how strict the customer's tolerance (ϵ\\epsilon) is, you can always find a high-enough precision for your machine (δ\\delta) to satisfy it.

#### Formal Definition

We say lim⁡x→cf(x)\=L\\lim\_{x \\to c} f(x) = L if:

For every ϵ\>0\\epsilon > 0 (the challenge), there exists a δ\>0\\delta > 0 (the response) such that:

0<∣x−c∣<δ  ⟹  ∣f(x)−L∣<ϵ0 < |x - c| < \\delta \\implies |f(x) - L| < \\epsilon

**Translation:** If the input xx is within δ\\delta of cc, then the output f(x)f(x) is guaranteed to be within ϵ\\epsilon of LL.

## Limit Laws

These rules let us compute limits of complex expressions from simpler ones. If lim⁡x→cf(x)\=L\\lim\_{x \\to c} f(x) = L and lim⁡x→cg(x)\=M\\lim\_{x \\to c} g(x) = M:

Sum/Differencelim⁡\[f(x)±g(x)\]\=L±M\\lim \[f(x) \\pm g(x)\] = L \\pm M

Productlim⁡\[f(x)⋅g(x)\]\=L⋅M\\lim \[f(x) \\cdot g(x)\] = L \\cdot M

Quotientlim⁡f(x)g(x)\=LM\\lim \\frac{f(x)}{g(x)} = \\frac{L}{M}(if M != 0)

Scalar Multiplelim⁡\[k⋅f(x)\]\=k⋅L\\lim \[k \\cdot f(x)\] = k \\cdot L

Powerlim⁡\[f(x)\]n\=Ln\\lim \[f(x)\]^n = L^n

Compositionlim⁡f(g(x))\=f(M)\\lim f(g(x)) = f(M)(if f continuous at M)

## L'Hopital's Rule

When direct substitution gives an **indeterminate form** like 00\\frac{0}{0} or ∞∞\\frac{\\infty}{\\infty}, L'Hopital's Rule provides a way forward.

#### L'Hopital's Rule

If lim⁡x→cf(x)g(x)\\lim\_{x \\to c} \\frac{f(x)}{g(x)} gives 00\\frac{0}{0} or ±∞±∞\\frac{\\pm\\infty}{\\pm\\infty}, then:

lim⁡x→cf(x)g(x)\=lim⁡x→cf′(x)g′(x)\\lim\_{x \\to c} \\frac{f(x)}{g(x)} = \\lim\_{x \\to c} \\frac{f'(x)}{g'(x)}

provided the limit on the right exists (or is infinity).

#### Example

Find lim⁡x→0sin⁡xx\\lim\_{x \\to 0} \\frac{\\sin x}{x}:

Direct substitution: sin⁡00\=00\\frac{\\sin 0}{0} = \\frac{0}{0} (indeterminate)

Apply L'Hopital: lim⁡x→0cos⁡x1\=cos⁡01\=1\\lim\_{x \\to 0} \\frac{\\cos x}{1} = \\frac{\\cos 0}{1} = 1

This limit (sin x / x = 1 as x goes to 0) is fundamental in deriving the derivative of sine.

#### Caution

Only apply L'Hopital when you have an indeterminate form! Applying it to 10\\frac{1}{0} (which is just undefined, not indeterminate) gives wrong answers.

## Continuity

Intuitively, a function is **continuous** if you can draw it without lifting your pen. Formally, continuity requires three conditions at a point c:

#### Three Conditions for Continuity at c

1

f(c)f(c) is defined

No holes at the point.

2

lim⁡x→cf(x)\\lim\_{x \\to c} f(x) exists

Left and right limits agree.

3

lim⁡x→cf(x)\=f(c)\\lim\_{x \\to c} f(x) = f(c)

The limit equals the function value.

#### Important Properties

-   Polynomials are continuous everywhere.
-   Rational functions are continuous except where denominator = 0.
-   Exponential, logarithmic, and trig functions are continuous on their domains.
-   Sums, products, and compositions of continuous functions are continuous.

## Types of Discontinuity

Understanding where and how functions "break" is crucial for choosing activation functions and understanding gradient flow.

#### Removable

Limit exists but f(c) is undefined or wrong. Can be "fixed" by redefining f(c) = L.

Example: (x^2-1)/(x-1) at x=1

#### Jump

Left and right limits both exist but differ. Function "jumps."

Example: Heaviside step function

#### Infinite

Function approaches infinity (vertical asymptote). Limit DNE.

Example: 1/x at x=0

## Continuity vs Differentiability

This distinction is critical for understanding activation functions in deep learning.

#### The Hierarchy

All FunctionsincludesContinuousincludesDifferentiable

#### Differentiable implies Continuous

If f'(c) exists, then f must be continuous at c. You can't have a derivative at a gap or jump.

#### Continuous does NOT imply Differentiable

A function can be continuous but have a "corner" where the derivative is undefined.

#### The Classic Example: |x|

The absolute value function f(x) = |x| is continuous everywhere. But at x = 0:

-   Left derivative: lim⁡h→0−∣0+h∣−∣0∣h\=lim⁡h→0−−hh\=−1\\lim\_{h \\to 0^-} \\frac{|0+h|-|0|}{h} = \\lim\_{h \\to 0^-} \\frac{-h}{h} = -1
-   Right derivative: lim⁡h→0+∣0+h∣−∣0∣h\=lim⁡h→0+hh\=+1\\lim\_{h \\to 0^+} \\frac{|0+h|-|0|}{h} = \\lim\_{h \\to 0^+} \\frac{h}{h} = +1

Since -1 != +1, the derivative at 0 does not exist. The function has a "corner."

## ML Applications

#### The Death of the Perceptron

Early neural networks used the **step function**: output 0 if input is negative, 1 otherwise.

**Problem:** It has a **jump discontinuity** at 0. The derivative is 0 everywhere else, so gradients cannot flow. Backpropagation fails completely. This is why we moved to smooth activations like Sigmoid.

#### ReLU: The Practical Compromise

ReLU: f(x)\=max⁡(0,x)f(x) = \\max(0, x). It is **continuous everywhere** but has a corner at x = 0 (not differentiable there).

**Solution:** We use **subgradients**. We arbitrarily define f'(0) = 0 or 1. Since the probability of x being exactly 0.0000... is negligible, this works perfectly in practice.

#### Softmax and Numerical Stability

Softmax: σ(zi)\=ezi∑jezj\\sigma(z\_i) = \\frac{e^{z\_i}}{\\sum\_j e^{z\_j}}. When ziz\_i gets large, ezie^{z\_i} can overflow to infinity. We use the trick of subtracting max(z) from all inputs, which doesn't change the output but keeps values bounded. This is a practical application of limit behavior.

#### Loss Function Continuity

MSE Loss (y−y^)2(y - \\hat{y})^2 is continuous and differentiable everywhere, making gradient descent smooth. Cross-entropy loss is continuous but has a singularity as predictions approach 0 or 1, which is why we clip probabilities away from exactly 0/1.