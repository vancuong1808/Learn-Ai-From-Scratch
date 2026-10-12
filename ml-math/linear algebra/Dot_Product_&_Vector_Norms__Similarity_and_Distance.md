**Title:** Dot Product & Vector Norms: Similarity and Distance

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/dot-norm](https://www.tensortonic.com/ml-math/linear-algebra/dot-norm)

---

# Page Structure Map
```text
Dot Product & Vector Norms: Similarity and Distance
├── Introduction
│   └── Why These Matter
├── The Dot Product
│   ├── Algorithm
│   └── Matrix Notation
├── Geometric Interpretation
│   ├── Positive
│   ├── Zero
│   ├── Negative
│   └── The Projection View
├── Properties of the Dot Product
│   ├── Commutative
│   ├── Distributive
│   └── Self Dot Product
├── Interactive: Dot Product
├── Vector Norms
│   ├── L2 Norm (Euclidean)
│   ├── L1 Norm (Manhattan)
│   └── L-infinity Norm (Max)
├── Properties of Norms
├── The p-Norm Family
├── Interactive: Unit Balls
├── Normalization (Unit Vectors)
│   └── Applications
├── ML Applications
│   ├── Regularization
│   ├── Attention Mechanism
│   └── Distance Metrics
└── Computational Considerations
    ├── Time Complexity
    └── Hardware
```

---

## Introduction

At the heart are two fundamental operations: **Dot Product** (alignment) and **Norms** (magnitude).

Every neural network prediction, recommender system suggestion, and regularization penalty uses these operations. Together, they form the foundation of how machines understand similarity, distance, and direction in high-dimensional space.

### Why These Matter

Understanding dot products and norms is crucial for modern ML architectures. [Matrix multiplication](https://www.tensortonic.com/ml-math/linear-algebra/matrix-multiplication) is just batched dot products. **Transformer attention** computes similarity via scaled dot products. Regularization techniques like Ridge and Lasso use different norms to constrain model complexity.

## The Dot Product

The dot product (also called inner product or scalar product) takes two vectors of equal length and returns a single scalar. It is the most fundamental operation in linear algebra.

a⋅b\=∑i\=1naibi\\mathbf{a} \\cdot \\mathbf{b} = \\sum\_{i=1}^{n} a\_i b\_i

Multiply corresponding elements, then sum.

#### Algorithm

a = \[1, 3\]  
b = \[4, -2\]  
a · b = (1)(4) + (3)(-2) = **\-2**

#### Matrix Notation

aTb\\mathbf{a}^T \\mathbf{b}

Row vector × Column vector. This generalizes to specific rows and columns in larger matrices.

## Geometric Interpretation

The dot product measures **alignment** between vectors. It bridges algebra and geometry.

a⋅b\=∣∣a∣∣⋅∣∣b∣∣cos⁡(θ)\\mathbf{a} \\cdot \\mathbf{b} = ||\\mathbf{a}|| \\cdot ||\\mathbf{b}|| \\cos(\\theta)

Product of magnitudes times the cosine of the angle between them.

#### Positive

Vectors point in roughly the same direction (θ<90°\\theta < 90°).

#### Zero

Vectors are orthogonal (perpendicular) (θ\=90°\\theta = 90°).

#### Negative

Vectors point in opposite directions (θ\>90°\\theta > 90°).

#### The Projection View

Think of it as casting a shadow of vector **a** onto **b**. The dot product tells you: "How much of a goes in the direction of b?". This "projection" concept is why dot products measure similarity.

## Properties of the Dot Product

#### Commutative

a⋅b\=b⋅a\\mathbf{a} \\cdot \\mathbf{b} = \\mathbf{b} \\cdot \\mathbf{a}

Order does not matter.

#### Distributive

a⋅(b+c)\=a⋅b+a⋅c\\mathbf{a} \\cdot (\\mathbf{b} + \\mathbf{c}) = \\mathbf{a} \\cdot \\mathbf{b} + \\mathbf{a} \\cdot \\mathbf{c}

Distributes over vector addition.

#### Self Dot Product

a⋅a\=∣∣a∣∣2\\mathbf{a} \\cdot \\mathbf{a} = ||\\mathbf{a}||^2

Dotting a vector with itself gives the squared norm (magnitude).

## Interactive: Dot Product

Drag the sliders to change vectors. See how the dot product relates to angle, cosine, and projection.

## Vector Norms

A norm measures the "size" or "magnitude" of a vector. Different norms measure size in different ways, each with unique geometric interpretations and ML applications.

#### L2 Norm (Euclidean)

∣∣x∣∣2\=∑xi2||\\mathbf{x}||\_2 = \\sqrt{\\sum x\_i^2}

Straight-line distance. Used in Ridge Regression, KNN.

#### L1 Norm (Manhattan)

∣∣x∣∣1\=∑∣xi∣||\\mathbf{x}||\_1 = \\sum |x\_i|

Sum of absolute values (taxi-cab distance). Promotes sparsity (Lasso).

#### L-infinity Norm (Max)

∣∣x∣∣∞\=max⁡∣xi∣||\\mathbf{x}||\_\\infty = \\max |x\_i|

Largest single element. Used in adversarial robustness.

## Properties of Norms

For a function to be a valid norm, it must satisfy three axioms:

-   **Non-negativity:** Norm is always non-negative. Zero only if the vector is zero.
-   **Absolute Homogeneity:** Scaling the vector scales the norm by the absolute scalar value.
-   **Triangle Inequality:** The direct path is always shorter than any detour. ∣∣x+y∣∣≤∣∣x∣∣+∣∣y∣∣||\\mathbf{x} + \\mathbf{y}|| \\leq ||\\mathbf{x}|| + ||\\mathbf{y}||

## The p-Norm Family

All common norms are special cases of the generalized p-norm (where p≥1p \\geq 1).

∣∣x∣∣p\=(∑i\=1n∣xi∣p)1/p||\\mathbf{x}||\_p = \\left(\\sum\_{i=1}^{n} |x\_i|^p\\right)^{1/p}

p\=1p = 1

L1 Norm (Manhattan)

p\=2p = 2

L2 Norm (Euclidean)

p\=3,4,…p = 3, 4, \\ldots

Higher-order norms

p→∞p \\to \\infty

L-infinity (Max)

## Interactive: Unit Balls

The "unit ball" (vectors with norm = 1) has a different shape for each p-norm. This shape explains why [L1 regularization](https://www.tensortonic.com/ml-math/optimization/regularization) promotes sparsity (sharp corners on axes).

## Normalization (Unit Vectors)

Normalization converts a vector to a unit vector (norm = 1) pointing in the same direction, isolating direction from magnitude.

x^\=x∣∣x∣∣\\hat{\\mathbf{x}} = \\frac{\\mathbf{x}}{||\\mathbf{x}||}

Divide by the norm.

#### Applications

-   **Cosine Similarity:** Normalized vectors reduce dot product to cosine similarity.
-   **Batch Norm:** Normalizes layer activations to stabilize training.
-   **Word Embeddings:** Often normalized so only direction (meaning) matters, not frequency.

## ML Applications

### Regularization

Penalizing the norm of weights prevents overfitting.

-   **L1 (Lasso):** λ∣∣w∣∣1\\lambda ||\\mathbf{w}||\_1 - creates sparse models (feature selection).
-   **L2 (Ridge):** λ∣∣w∣∣22\\lambda ||\\mathbf{w}||\_2^2 - distributes weight values, prevents unstable large weights.

### Attention Mechanism

Transformer attention is a scaled dot product. It calculates relevance scores between tokens.

### Distance Metrics

KNN and K-Means rely on L2 distance ∣∣x−y∣∣2||\\mathbf{x} - \\mathbf{y}||\_2. Using different norms here changes the algorithm's behavior drastically.

## Computational Considerations

#### Time Complexity

Dot Product (n-dim)O(n)O(n)

Norm CalculationO(n)O(n)

#### Hardware

GPUs optimize these O(n) ops via massive parallelism (SIMD). Always use vectorized libraries (PyTorch/NumPy), never loops.