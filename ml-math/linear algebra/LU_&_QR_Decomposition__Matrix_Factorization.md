**Title:** LU & QR Decomposition: Matrix Factorization

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/lu-qr-decomposition](https://www.tensortonic.com/ml-math/linear-algebra/lu-qr-decomposition)

---

# Page Structure Map
```text
LU & QR Decomposition: Matrix Factorization
├── Why Decompose Matrices?
│   ├── LU Decomposition
│   └── QR Decomposition
├── LU Decomposition
│   └── Solving with LU
│       └── Partial Pivoting (PA = LU)
├── Interactive Simulator
├── QR Decomposition
│   └── Why QR is Stable
├── Gram-Schmidt Process
│   └── Initial Vectors
├── Case Study: Bulb Characteristic Fitting
│   ├── The Problem
│   ├── Using QR for Least Squares
│   └── Why Not Normal Equations?
├── LU vs QR vs SVD
└── ML Applications
    ├── Cholesky (LLᵀ)
    ├── Eigenvalue Algorithms
    ├── Backpropagation Efficiency
    └── Randomized Linear Algebra
```

---

## Why Decompose Matrices?

Solving Ax\=bAx = b directly by computing A−1A^{-1} is expensive (O(n3)O(n^3)) and numerically unstable. Matrix decompositions factor AA into simpler matrices that make solving systems faster and more stable.

#### LU Decomposition

Factor into Lower × Upper triangular. Good for repeated solves with same A, different b. Analogy: Gaussian Elimination.

#### QR Decomposition

Factor into Orthogonal × Upper triangular. Numerically stable, ideal for least squares. Analogy: Gram-Schmidt.

## LU Decomposition

Factor a square matrix into a product of Lower and Upper triangular matrices. It is essentially recording the steps of Gaussian Elimination.

A\=LUA = LU

LL = lower triangular (1s on diagonal), UU = upper triangular (pivots on diagonal)

### Solving with LU

1.  Factor once: A\=LUA = LU (O(n³))
2.  For each b, solve Ly\=bLy = b (forward substitution, O(n²))
3.  Then solve Ux\=yUx = y (back substitution, O(n²))

#### Partial Pivoting (PA = LU)

In practice, we permute rows to avoid division by small numbers. This gives PA\=LUPA = LU where PP is a permutation matrix.

## Interactive Simulator

Step through the LU and QR decomposition process. Watch how matrices are factored at each step.

## QR Decomposition

Factor any matrix (even rectangular) into an Orthogonal matrix times an Upper triangular matrix.

A\=QRA = QR

QTQ\=IQ^TQ = I (orthonormal columns), RR = upper triangular

### Why QR is Stable

[Orthogonal matrices](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality) preserve lengths: ∣∣Qx∣∣\=∣∣x∣∣||Qx|| = ||x||. This means errors do not amplify during computation (condition number of QQ is 1).

## Gram-Schmidt Process

The algorithm to construct QQ is called Gram-Schmidt. It iteratively subtracts projections to force orthogonality.

#### Initial Vectors

v₁ Angle15°

v₂ Angle60°

1.  u1\=a1u\_1 = a\_1, then q1\=u1/∣∣u1∣∣q\_1 = u\_1 / ||u\_1||
2.  u2\=a2−(a2⋅q1)q1u\_2 = a\_2 - (a\_2 \\cdot q\_1)q\_1, then q2\=u2/∣∣u2∣∣q\_2 = u\_2 / ||u\_2||
3.  uk\=ak−∑j\=1k−1(ak⋅qj)qju\_k = a\_k - \\sum\_{j=1}^{k-1}(a\_k \\cdot q\_j)q\_j, then qk\=uk/∣∣uk∣∣q\_k = u\_k / ||u\_k||

**Modified Gram-Schmidt** is more numerically stable. Modern libraries use Householder reflections (a sequence of mirrors) instead of projections to compute QR.

## Case Study: Bulb Characteristic Fitting

#### The Problem

You have voltage vs brightness data for a bulb. Fit a polynomial: brightness = a₀ + a₁V + a₂V². This is an overdetermined system (more data points than unknowns).

#### Using QR for Least Squares

1.  Build Vandermonde matrix: Aij\=VijA\_{ij} = V\_i^j
2.  Compute A\=QRA = QR
3.  Solve Ra^\=QTbR\\hat{a} = Q^Tb by back substitution (since R is triangular, this is easy)

#### Why Not Normal Equations?

For high degree polynomials, ATAA^TA becomes [ill-conditioned](https://www.tensortonic.com/ml-math/linear-algebra/determinants-inverses). QR avoids forming this product, preserving numerical stability.

## LU vs QR vs SVD

|  | LU | QR | SVD |
| --- | --- | --- | --- |
| Matrix Shape | Square only | Any shape | Any shape |
| Speed | Fastest (∼n3/3)(\\sim n^3/3) | Medium (∼2n3/3)(\\sim 2n^3/3) | Slowest (∼10n3)(\\sim 10n^3) |
| Stability | Needs pivoting | Good | Best (Total Least Squares) |
| Best For | Linear solves (Ax=b) | Least squares | Rank, compression |

## ML Applications

### Cholesky (LLᵀ)

For symmetric positive definite matrices (like covariance matrices), Cholesky is 2x faster than LU. Used in [Gaussian Processes](https://www.tensortonic.com/ml-math/probability/distributions) for sampling.

### Eigenvalue Algorithms

The **QR Algorithm** for [eigenvalues](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) repeatedly computes QR decompositions. This is the standard way `numpy.linalg.eig` works for non-symmetric matrices.

### Backpropagation Efficiency

Computing gradients through matrix inverses involves solving linear systems, which typically uses pre-computed LU factors. Frameworks like JAX exploit this for efficiency.

### Randomized Linear Algebra

Large scale ML uses randomized QR (sampling columns) to get approximate decompositions. This is much faster than full SVD for massive recommendation system matrices.