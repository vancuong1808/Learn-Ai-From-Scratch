**Title:** Positive Definite Matrices: Convexity Guarantees

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/positive-definite](https://www.tensortonic.com/ml-math/linear-algebra/positive-definite)

---

# Page Structure Map
```text
Positive Definite Matrices: Convexity Guarantees
├── What is Positive Definite?
│   └── The Core Property
├── The Quadratic Form
│   ├── Positive Definite
│   ├── Indefinite
│   └── Negative Definite
├── Interactive Visualization
├── Positive Definite
├── Tests for Definiteness
│   ├── For 2×2
│   └── Computational
├── Cholesky Decomposition
│   └── Why Cholesky?
├── The Definiteness Spectrum
└── ML Applications
    ├── Covariance Matrices
    ├── Convex Optimization
    ├── Kernel Matrices
    └── Gaussian Processes
```

---

## What is Positive Definite?

A symmetric matrix AA is **positive definite** if for every nonzero vector xx, the quadratic form xTAx\>0x^T A x > 0. Intuitively, this means the matrix always "curves upward" and has a unique minimum at the origin.

Positive definite matrices are the "well behaved" matrices in optimization. They guarantee that gradient descent will find a unique global minimum. They appear everywhere: covariance matrices, Hessians of convex functions, and kernel matrices.

### The Core Property

xTAx\>0x^T A x > 0 for all x≠0x \\neq 0. This single condition has profound consequences for optimization, statistics, and numerical stability.

## The Quadratic Form

The expression xTAxx^T A x is called the **quadratic form** associated with AA. For a 2×2 symmetric matrix, it expands to:

xT\[abbc\]x\=ax12+2bx1x2+cx22x^T \\begin{bmatrix} a & b \\\\ b & c \\end{bmatrix} x = ax\_1^2 + 2bx\_1x\_2 + cx\_2^2

This is a paraboloid in 3D. Its shape depends on the matrix entries.

#### Positive Definite

Bowl shape. Unique minimum at origin. All level curves are ellipses.

#### Indefinite

Saddle shape. Neither max nor min at origin. Level curves are hyperbolas.

#### Negative Definite

Inverted bowl. Unique maximum at origin. All level curves are ellipses.

## Interactive Visualization

Adjust the symmetric matrix entries and watch the level curve of the quadratic form change. The arrows show gradient directions. Notice how positive definite matrices create elliptical level curves that all point inward.

Shape Control

λ₁ (Curvature 1)1.0

λ₂ (Curvature 2)0.5

Rotation (Principal Axes)0°

---

## Positive Definite

Bowl shape (Convex). Unique global minimum.

## Tests for Definiteness

There are several equivalent ways to check if a symmetric matrix is positive definite:

1.  **Eigenvalue Test:** All [eigenvalues](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) are strictly positive.
2.  **Sylvester's Criterion:** All leading principal minors (upper-left determinants) are positive.
3.  **Cholesky Test:** The Cholesky decomposition A\=LLTA = LL^T exists with positive diagonal entries.
4.  **Pivot Test:** In [LU decomposition](https://www.tensortonic.com/ml-math/linear-algebra/lu-qr-decomposition), all pivots are positive.

#### For 2×2

A\=\[abbc\]A = \\begin{bmatrix} a & b \\\\ b & c \\end{bmatrix} is PD if a\>0a > 0 and ac−b2\>0ac - b^2 > 0.

#### Computational

In practice, attempt Cholesky factorization. If it succeeds, the matrix is PD.

## Cholesky Decomposition

A positive definite matrix AA can be uniquely factored as A\=LLTA = LL^T, where LL is a lower triangular matrix with positive diagonal entries. This is the "square root" of a matrix.

A\=LLTA = LL^T

LL is lower triangular, LTL^T is the transpose (upper triangular)

### Why Cholesky?

-   **Speed:** 2× faster than [LU](https://www.tensortonic.com/ml-math/linear-algebra/lu-qr-decomposition) because we exploit symmetry.
-   **Stability:** No pivoting needed (guaranteed stable for PD).
-   **Sampling:** To sample from N(0,Σ)\\mathcal{N}(0, \\Sigma), compute LL from Σ\=LLT\\Sigma = LL^T, then x\=Lzx = Lz where z∼N(0,I)z \\sim \\mathcal{N}(0, I).

## The Definiteness Spectrum

The classification of a symmetric matrix depends on the signs of its eigenvalues:

| Type | Eigenvalues | Quadratic Form | Optimization |
| --- | --- | --- | --- |
| Positive Definite | All λ > 0 | xTAx\>0x^TAx > 0 | Unique minimum |
| Positive Semidefinite | All λ ≥ 0 | xTAx≥0x^TAx \\geq 0 | Minimum subspace |
| Indefinite | Mixed signs | Both + and - | Saddle point |
| Negative Semidefinite | All λ ≤ 0 | xTAx≤0x^TAx \\leq 0 | Maximum subspace |
| Negative Definite | All λ < 0 | xTAx<0x^TAx < 0 | Unique maximum |

## ML Applications

### Covariance Matrices

Sample covariance matrices are always positive semidefinite. In [Gaussians](https://www.tensortonic.com/ml-math/probability/distributions), we need PD covariance for the density to be well defined.

### Convex Optimization

A function is convex if its [Hessian](https://www.tensortonic.com/ml-math/linear-algebra/matrix-calculus) is PSD everywhere. This guarantees gradient descent converges to the global minimum.

### Kernel Matrices

A valid kernel function must produce PSD Gram matrices. This is Mercer's condition, ensuring the implicit feature space is well defined.

### Gaussian Processes

GP priors require PD covariance matrices for sampling. Cholesky is used to generate correlated samples from a GP posterior.