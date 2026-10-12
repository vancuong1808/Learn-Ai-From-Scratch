**Title:** Singular Value Decomposition (SVD) Explained

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/svd](https://www.tensortonic.com/ml-math/linear-algebra/svd)

---

# Page Structure Map
```text
Singular Value Decomposition (SVD) Explained
├── Introduction
│   └── What SVD Reveals
├── The Movie Rating Intuition
│   ├── U (User-Concept)
│   ├── Σ (Strength)
│   └── VTV^T (Concept-Movie)
├── Mathematical Formulation
│   ├── UU (m×mm \\times m)
│   ├── Σ\\Sigma (m×nm \\times n)
│   ├── VTV^T (n×nn \\times n)
│   └── Relationship to [Eigendecomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector)
├── Interactive: SVD Steps
│   └── Current Operation
├── Geometric Interpretation
│   ├── VTV^T (Rotation in Input Space)
│   ├── Σ\\Sigma (Scaling)
│   ├── UU (Rotation in Output Space)
│   └── Unit Circle to Ellipse
├── Derivation
│   ├── Step 1: Form Symmetric Matrices
│   ├── Step 2: Key Identity
│   ├── Step 3: Define Components
│   └── Verification
├── Truncated SVD (Low-Rank Approximation)
│   ├── Eckart-Young Theorem (Optimality)
│   └── Compressing a 1000×1000 Image
├── Interactive: Low-Rank Approximation
├── Key Properties
│   ├── Rank
│   ├── Frobenius Norm
│   ├── Spectral Norm (2-Norm)
│   ├── Condition Number
│   └── Pseudo-Inverse (Moore-Penrose)
├── Applications
│   ├── Image Compression
│   ├── Recommendation Systems
│   ├── Latent Semantic Analysis (LSA)
│   └── Noise Reduction
└── SVD vs PCA vs Eigendecomposition
    └── PCA via SVD
```

---

## Introduction

[Eigendecomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) is powerful, but it has a fatal flaw: it only works on **square** matrices. Real-world data matrices are rarely square. We might have m users and n movies, or m documents and n words.

**Singular Value Decomposition (SVD)** is the generalization that works on **any** matrix, regardless of shape. It is arguably the most important matrix factorization in applied linear algebra and machine learning.

### What SVD Reveals

-   **Rank:** Number of non-zero singular values = rank of matrix.
-   **Range:** Column space is spanned by left singular vectors (U).
-   **Null Space:** Right singular vectors (V) with zero singular values span null space.
-   **Condition Number:** Ratio of largest to smallest singular value measures numerical stability.

## The Movie Rating Intuition

Consider a matrix A where rows are **Users** and columns are **Movies**. Entry A\[i,j\] is user i's rating of movie j. SVD factorizes this into three parts that reveal hidden structure:

#### U (User-Concept)

Each row describes a user in terms of hidden "concepts" (latent factors).

"Alice: 0.9 Sci-Fi lover, 0.1 Romance fan, 0.3 Action enthusiast"

#### Σ (Strength)

Diagonal values show how "strong" each concept is in the dataset.

"Sci-Fi concept explains 40% of variance, Romance 25%, Action 15%..."

#### VTV^T (Concept-Movie)

Each column describes a movie in terms of the same concepts.

"The Matrix: 0.95 Sci-Fi, 0.05 Romance, 0.8 Action"

To predict Alice's rating for The Matrix: multiply Alice's concept profile by concept strengths by Matrix's concept profile. The "concepts" are learned automatically from the data, not pre-defined!

## Mathematical Formulation

**SVD Theorem:** For any real matrix A of size m × n, there exist orthogonal matrices U and V and a diagonal matrix Σ such that:

A\=UΣVTA = U \\Sigma V^T

##### UU (m×mm \\times m)

Left Singular Vectors

-   [Orthogonal](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality): UTU\=IU^T U = I
-   Columns are eigenvectors of AATAA^T

##### Σ\\Sigma (m×nm \\times n)

Singular Values

-   Diagonal with entries σi≥0\\sigma\_i \\geq 0
-   Sorted: σ1≥σ2≥…≥0\\sigma\_1 \\geq \\sigma\_2 \\geq \\ldots \\geq 0

##### VTV^T (n×nn \\times n)

Right Singular Vectors

-   [Orthogonal](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality): VTV\=IV^T V = I
-   Rows are eigenvectors of ATAA^T A

#### Relationship to [Eigendecomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector)

The singular values and singular vectors come from eigendecompositions of symmetric matrices:

Left Singular Vectors

AAT\=UΛUTAA^T = U \\Lambda U^T

UU columns are eigenvectors of AATAA^T

Right Singular Vectors

ATA\=VΛVTA^T A = V \\Lambda V^T

VV columns are eigenvectors of ATAA^T A

**Singular values = square roots of eigenvalues:** σi\=λi\\sigma\_i = \\sqrt{\\lambda\_i}

## Interactive: SVD Steps

SVD decomposes any linear transformation into three operations: rotate, scale, then rotate. Watch how a unit circle transforms through each step.

Matrix Constructor

Vᵀ (Input Rotation)\-45°

Σ (Scaling)

σ₁2

σ₂1

U (Output Rotation)30°

---

#### Current Operation

Start with a unit circle. The orthonormal basis vectors (i, j) are aligned with the axes.

## Geometric Interpretation

SVD tells us that **every** linear transformation, no matter how complex, is equivalent to three simple operations in sequence:

1

#### VTV^T (Rotation in Input Space)

Rotates the coordinate system to align with the "principal axes" of the transformation. After this rotation, the transformation will only stretch along coordinate axes.

2

#### Σ\\Sigma (Scaling)

Stretches or compresses along the principal axes by the singular values. This is where the transformation does its "work." A unit circle becomes an ellipse whose semi-axes have lengths σ1,σ2\\sigma\_1, \\sigma\_2, etc.

3

#### UU (Rotation in Output Space)

Final rotation that orients the scaled result to its output position. If the matrix changes dimension (m≠nm \\neq n), this rotation happens in the output space which may have different dimensions than input.

#### Unit Circle to Ellipse

When you apply any matrix A to a unit circle (or sphere in higher dimensions), the result is an ellipse (or ellipsoid). The **singular values are the lengths of the semi-axes** of this ellipse, and the **singular vectors are the directions of these axes**.

## Derivation

How do we prove SVD exists and find it? The key insight is using ATAA^T A and AATAA^T.

#### Step 1: Form Symmetric Matrices

For any AA, both ATAA^T A (n×nn \\times n) and AATAA^T (m×mm \\times m) are symmetric positive semi-definite. Therefore they have:

-   Real, non-negative eigenvalues
-   [Orthogonal](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality) eigenvectors

#### Step 2: Key Identity

If ATAv\=λvA^T A \\mathbf{v} = \\lambda \\mathbf{v}, then

AAT(Av)\=A(ATAv)\=A(λv)\=λ(Av)AA^T (A\\mathbf{v}) = A(A^T A \\mathbf{v}) = A(\\lambda \\mathbf{v}) = \\lambda (A\\mathbf{v})

So if v\\mathbf{v} is an eigenvector of ATAA^T A with eigenvalue λ\\lambda, then AvA\\mathbf{v} is an eigenvector of AATAA^T with the same eigenvalue λ\\lambda.

#### Step 3: Define Components

VV (Right Vectors)

Eigenvectors of ATAA^T A, normalized.

Σ\\Sigma (Singular Values)

σi\=λi\\sigma\_i = \\sqrt{\\lambda\_i}

UU (Left Vectors)

ui\=1σiAvi\\mathbf{u}\_i = \\frac{1}{\\sigma\_i} A \\mathbf{v}\_i

#### Verification

With these definitions, Avi\=σiuiA\\mathbf{v}\_i = \\sigma\_i \\mathbf{u}\_i for each ii. In matrix form: AV\=UΣAV = U\\Sigma, which rearranges to A\=UΣVTA = U\\Sigma V^T (since VV is orthogonal, V−1\=VTV^{-1} = V^T).

## Truncated SVD (Low-Rank Approximation)

The real power of SVD is not in exact factorization but in **approximation**. By keeping only the top k singular values, we get the best rank-k approximation of A.

Ak\=∑i\=1kσiuiviTA\_k = \\sum\_{i=1}^{k} \\sigma\_i \\mathbf{u}\_i \\mathbf{v}\_i^T

This is the sum of k rank-1 matrices, each scaled by its singular value.

#### Eckart-Young Theorem (Optimality)

Among all rank-kk matrices, AkA\_k minimizes the Frobenius norm of the error:

∣∣A−Ak∣∣F\=σk+12+σk+22+…+σr2||A - A\_k||\_F = \\sqrt{\\sigma\_{k+1}^2 + \\sigma\_{k+2}^2 + \\ldots + \\sigma\_r^2}

The error equals the sum of squared discarded singular values! This is why keeping the largest singular values is optimal.

#### Compressing a 1000×1000 Image

**Original:** 1,000,000 pixels.  
**SVD (k=50):** 50 left vectors + 50 singular values + 50 right vectors = 50 × (1000 + 1 + 1000) = 100,050 numbers.  
**Result:** 90% space reduction with minimal visual loss!

## Interactive: Low-Rank Approximation

Drag the slider to see how much "energy" (information) is captured by keeping different numbers of singular values.

## Key Properties

#### Rank

Number of non-zero singular values.

#### Frobenius Norm

Sum of squared entries = sum of squared singular values.  
∣∣A∣∣F\=∑σi2||A||\_F = \\sqrt{\\sum \\sigma\_i^2}

#### Spectral Norm (2-Norm)

Maximum stretching factor = largest singular value.  
∣∣A∣∣2\=σ1||A||\_2 = \\sigma\_1

#### Condition Number

Measures numerical stability.  
κ(A)\=σ1/σr\\kappa(A) = \\sigma\_1 / \\sigma\_r

#### Pseudo-Inverse (Moore-Penrose)

How do you invert a non-square matrix? SVD gives the answer:

A+\=VΣ+UTA^+ = V \\Sigma^+ U^T

Where Σ+\\Sigma^+ is formed by taking reciprocals of non-zero singular values (1/σi1/\\sigma\_i) and transposing. This solves least squares problems: x\=A+b\\mathbf{x} = A^+ \\mathbf{b}.

## Applications

### Image Compression

An image is a matrix of pixel values. Keep top k singular values, discard the rest. The image remains recognizable even with k much smaller than rank.

### Recommendation Systems

User-item rating matrix is sparse. SVD finds latent factors. Predicted rating = UkΣkVkTU\_k \\Sigma\_k V\_k^T. This fills in missing entries. (Netflix Prize key component).

### Latent Semantic Analysis (LSA)

Document-term matrix: rows = documents, columns = words. SVD reveals "topics" as latent dimensions. Documents and words are embedded in the same semantic space.

### Noise Reduction

Noise typically appears in small singular values (random fluctuations don't create strong patterns). Truncating small singular values removes noise while keeping signal.

## SVD vs PCA vs Eigendecomposition

| Method | Matrix Type | What It Finds | Main Use |
| --- | --- | --- | --- |
| [Eigendecomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) | Square (n×nn \\times n) | Eigenvalues & eigenvectors | Theoretical analysis, dynamics |
| SVD | Any (m×nm \\times n) | Singular values & vectors | Data compression, inversion |
| [PCA](https://www.tensortonic.com/ml-math/linear-algebra/pca) | Data matrix | Principal components | Dimensionality reduction |

#### PCA via SVD

[PCA](https://www.tensortonic.com/ml-math/linear-algebra/pca) is often **computed using SVD** rather than eigendecomposition of the covariance matrix:

1.  Center data matrix X
2.  Compute SVD: X = U Σ V^T
3.  Principal components = columns of V
4.  Projected data = U Σ

This is more numerically stable than forming X^T X explicitly.