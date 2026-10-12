**Title:** Projections & Least Squares Regression

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/projections-least-squares](https://www.tensortonic.com/ml-math/linear-algebra/projections-least-squares)

---

# Page Structure Map
```text
Projections & Least Squares Regression
├── The Geometry of Fitting
│   └── The Core Insight
├── Vector Projection
│   ├── The Projection
│   └── The Error
├── Interactive: Fitting Lines
│   ├── Optimal Line
│   ├── Orthogonality Check
│   └── Total Squared Error
├── Column Space View
│   └── Key Geometric Fact
├── The Normal Equations
│   ├── When Does This Work?
│   └── The Pseudoinverse
├── Case Study: Bulb Lifespan Prediction
│   ├── The Problem
│   ├── The Setup
│   └── The Geometric View
├── QR Decomposition Solution
└── ML Applications
    ├── Linear Regression
    ├── Ridge Regression
    ├── PCA via SVD
    └── Kernel Methods
```

---

## The Geometry of Fitting

Most real world systems are **overdetermined**: we have more equations than unknowns. The system Ax\=bAx = b has no exact solution. The data is too noisy, the model is too simple, or both.

**Least Squares** asks: "If we cannot hit bb exactly, what is the closest we can get?" The answer is the projection of bb onto the column space of AA.

### The Core Insight

Linear regression, polynomial fitting, and even deep learning (in parts) all reduce to projections. The error is minimized when it is perpendicular to the subspace of possible outputs.

## Vector Projection

Before matrices, let us project a single vector aa onto another vector bb.

projb(a)\=a⋅bb⋅bb\=a⋅b∣∣b∣∣2b\\text{proj}\_b(a) = \\frac{a \\cdot b}{b \\cdot b} b = \\frac{a \\cdot b}{||b||^2} b

The scalar a⋅bb⋅b\\frac{a \\cdot b}{b \\cdot b} tells us how far along bb the projection lands.

#### The Projection

The component of aa that lies along bb. This is what we keep.

#### The Error

e\=a−projb(a)e = a - \\text{proj}\_b(a). The residual, always perpendicular to bb.

## Interactive: Fitting Lines

In coordinate space, "perpendicular error" translates to minimizing the sum of vertical distance squared. Drag the points to see how the optimal line shifts to keep the residuals orthogonal to the feature space.

#### Optimal Line

y^\=0.56x+1.39\\hat{y} = 0.56x + 1.39

#### Orthogonality Check

The error vector ee must be orthogonal to the feature space (Column Space of X).

e · 1 (Bias)

0.0000

e · x (Feature)

0.0000

#### Total Squared Error

4.41

Visualize this as the total area of all the squares in the plot. Least Squares finds the minimum possible area.

## Column Space View

For a matrix AA, the column space C(A)C(A) is the set of all possible outputs AxAx. If bb is not in C(A)C(A), we project it.

1.  The target bb lives in Rm\\mathbb{R}^m (m data points).
2.  The column space C(A)C(A) is a subspace of Rm\\mathbb{R}^m (spanned by n features).
3.  We find the point b^∈C(A)\\hat{b} \\in C(A) closest to bb.
4.  The error e\=b−b^e = b - \\hat{b} is perpendicular to C(A)C(A).

### Key Geometric Fact

The shortest distance from a point to a subspace is measured along the perpendicular. This is why ATe\=0A^T e = 0 (the error is [orthogonal](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality) to every column of AA).

## The Normal Equations

From the perpendicularity condition AT(b−Ax)\=0A^T(b - Ax) = 0, we simplify to find the best weights x^\\hat{x}:

ATAx^\=ATbA^T A \\hat{x} = A^T b

Solution: x^\=(ATA)−1ATb\\hat{x} = (A^T A)^{-1} A^T b

### When Does This Work?

The inverse (ATA)−1(A^T A)^{-1} exists only when AA has **full column rank**, i.e., all columns are linearly independent. If features are collinear (one is a linear combination of others), ATAA^T A is singular and cannot be inverted. This is exactly why **ridge regression** adds λI\\lambda I to the diagonal: the matrix ATA+λIA^T A + \\lambda I is always invertible for λ\>0\\lambda > 0, regardless of rank.

### The Pseudoinverse

The matrix A+\=(ATA)−1ATA^+ = (A^T A)^{-1} A^T is called the **Moore Penrose Pseudoinverse**. It gives the least squares solution even when AA is not square.

## Case Study: Bulb Lifespan Prediction

#### The Problem

You have 100 bulbs with features (voltage, temperature) and lifespan measurements. You want to fit a linear model: Lifespan = w₁×Voltage + w₂×Temperature + w₀.

#### The Setup

Design matrix AA is 100×3 (100 samples, 3 features including bias). Target bb is 100×1. We solve for ww (3×1).

w^\=(ATA)−1ATb\\hat{w} = (A^T A)^{-1} A^T b

#### The Geometric View

The column space of AA is a 3D subspace in 100D space. The vector bb (lifespans) is projected onto this subspace. The residuals e\=b−Aw^e = b - A\\hat{w} are the prediction errors, perpendicular to all features.

## QR Decomposition Solution

Directly computing (ATA)−1(A^T A)^{-1} is numerically unstable. In practice, we use [QR decomposition](https://www.tensortonic.com/ml-math/linear-algebra/orthogonality#qr-decomposition).

1.  Decompose A\=QRA = QR where QQ is orthogonal, RR is upper triangular.
2.  Substitute: (QR)T(QR)x^\=(QR)Tb(QR)^T(QR)\\hat{x} = (QR)^T b
3.  Simplify: RTQTQRx^\=RTQTbR^T Q^T Q R \\hat{x} = R^T Q^T b
4.  Since QTQ\=IQ^T Q = I: RTRx^\=RTQTbR^T R \\hat{x} = R^T Q^T b
5.  Result: Rx^\=QTbR \\hat{x} = Q^T b (solve by back substitution)

**Numerical Advantage:** QR avoids squaring the condition number of AA. This is how `numpy.linalg.lstsq` works internally.

## ML Applications

### Linear Regression

The closed form solution w\=(XTX)−1XTyw = (X^TX)^{-1}X^Ty is exactly the normal equations. Gradient descent converges to the same point.

### Ridge Regression

Add regularization: w\=(XTX+λI)−1XTyw = (X^TX + \\lambda I)^{-1}X^Ty. When XTXX^TX is singular (features are collinear), OLS breaks. Adding λI\\lambda I shifts all eigenvalues away from zero, guaranteeing [invertibility](https://www.tensortonic.com/ml-math/linear-algebra/determinants-inverses) while also shrinking coefficients toward zero.

### PCA via SVD

[PCA](https://www.tensortonic.com/ml-math/linear-algebra/pca) finds the subspace that best approximates the data (least reconstruction error). This is a projection problem solved via SVD.

### Kernel Methods

Kernel Ridge Regression projects data into a high dimensional feature space and applies least squares there (via the kernel trick).