**Title:** Covariance Matrices: The Shape of Data

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/probability/covariance-matrix](https://www.tensortonic.com/ml-math/probability/covariance-matrix)

---

# Page Structure Map
```text
Covariance Matrices: The Shape of Data
├── Multidimensional Variance
├── The Covariance Matrix (Σ\\Sigma)
│   ├── Diagonal Elements
│   └── Off-Diagonal Elements
├── Interactive Simulator
│   └── Covariance Playground
│       └── Covariance Matrix Σ
├── Key Properties
│   ├── 1\. Symmetry
│   ├── 2\. [Positive Semi-Definite](https://www.tensortonic.com/ml-math/linear-algebra/positive-definite) (PSD)
│   ├── 3\. Linear Transformation
│   └── 4\. [Correlation Matrix](https://www.tensortonic.com/ml-math/statistics/correlation)
├── Geometric Intuition
│   ├── Spherical (Identity Covariance)
│   └── Ellipsoidal (General Covariance)
├── [Eigen-Decomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) (Spectral Theorem)
│   └── Why This Matters
├── Case Study: Bulb Quality Control
│   ├── The Scenario
│   ├── The Covariance Matrix
│   └── Mahalanobis Distance (Anomaly Detection)
└── [PCA](https://www.tensortonic.com/ml-math/linear-algebra/pca) (Principal Component Analysis)
    ├── The Algorithm
    ├── Variance Explained
    └── Whitening / Sphering
```

---

## Multidimensional Variance

In 1D, variance tells you how spread out data is along a single axis. But real data lives in many dimensions. Light bulb quality might be characterized by filament thickness, gas pressure, and glass clarity simultaneously.

In 2D (or n-D), data can be spread out in different _directions_. The **Covariance Matrix** captures this shape. It tells us not just how much each variable varies, but how they vary _together_.

## The Covariance Matrix (Σ\\Sigma)

For nn variables, it's an n×nn \\times n symmetric matrix.

Σ\=\[Var(X1)Cov(X1,X2)⋯Cov(X1,Xn)Cov(X2,X1)Var(X2)⋯Cov(X2,Xn)⋮⋮⋱⋮Cov(Xn,X1)Cov(Xn,X2)⋯Var(Xn)\]\\Sigma = \\begin{bmatrix} Var(X\_1) & Cov(X\_1,X\_2) & \\cdots & Cov(X\_1,X\_n) \\\\ Cov(X\_2,X\_1) & Var(X\_2) & \\cdots & Cov(X\_2,X\_n) \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ Cov(X\_n,X\_1) & Cov(X\_n,X\_2) & \\cdots & Var(X\_n) \\end{bmatrix}

#### Diagonal Elements

Variances of each variable. How spread out each variable is on its own.

#### Off-Diagonal Elements

Covariances between pairs of variables. How they move together.

## Interactive Simulator

Manipulate the covariance matrix for 3 variables (X, Y, Z). See how changing the covariances changes the 2D projections of the data cloud.

### Covariance Playground

#### Covariance Matrix Σ

X

Y

Z

X

1.0

0.80

0.50

Y

0.80

1.0

0.20

Z

0.50

0.20

1.0

**Invalid Matrix:** Impossible correlations. Reduce values.

**How to use:**Adjust sliders in the upper triangle. The matrix is symmetric, so changes reflect in the lower triangle automatically. Trying to set strong pairwise correlations (e.g. X-Y=0.9, Y-Z=0.9, X-Z=-0.9) will break geometry and show an error.

X vs Yr=0.80

X vs Zr=0.50

Y vs Zr=0.20

## Key Properties

#### 1\. Symmetry

Cov(X,Y)\=Cov(Y,X)Cov(X,Y) = Cov(Y,X). The matrix is symmetric, so Σ\=ΣT\\Sigma = \\Sigma^T.

#### 2\. [Positive Semi-Definite](https://www.tensortonic.com/ml-math/linear-algebra/positive-definite) (PSD)

xTΣx≥0x^T \\Sigma x \\geq 0 for any vector xx. Why? Because variance cannot be negative. The quadratic form represents a variance.

#### 3\. Linear Transformation

If Y\=AXY = AX, then ΣY\=AΣXAT\\Sigma\_Y = A \\Sigma\_X A^T. This is how covariance propagates through linear operations.

#### 4\. [Correlation Matrix](https://www.tensortonic.com/ml-math/statistics/correlation)

Normalize by dividing each entry by σiσj\\sigma\_i \\sigma\_j. Correlation matrix has 1s on diagonal, values between −1\-1 and 11 off-diagonal.

## Geometric Intuition

Think of the Covariance Matrix as defining an **ellipsoid** (a stretched sphere) in n-dimensional space. The equation (x−μ)TΣ−1(x−μ)\=c(x - \\mu)^T \\Sigma^{-1} (x - \\mu) = c defines an elliptical contour of constant probability.

#### Spherical (Identity Covariance)

Variables are uncorrelated with equal variance. Data cloud is a perfect sphere.

#### Ellipsoidal (General Covariance)

Variables have different variances and correlations. Data cloud is a tilted ellipsoid.

## [Eigen-Decomposition](https://www.tensortonic.com/ml-math/linear-algebra/eigenvalue-eigenvector) (Spectral Theorem)

This is the bridge between Linear Algebra and Statistics. Since Σ\\Sigma is symmetric PSD, it has a special decomposition:

Σ\=VΛVT\\Sigma = V \\Lambda V^T

**Eigenvectors (VV):**

The columns of VV point in the directions of the principal axes of the ellipsoid. These are the "natural" coordinate axes of the data, where variables are uncorrelated.

**Eigenvalues (Λ\\Lambda):**

The diagonal matrix where λi\\lambda\_i is the variance along the i-th principal axis. Larger eigenvalue = more spread in that direction.

### Why This Matters

By rotating data into the eigenvector basis, you **decorrelate** the variables. The covariance matrix in the new basis is diagonal. This is exactly what PCA does.

## Case Study: Bulb Quality Control

#### The Scenario

You measure 3 properties of each bulb: Filament Resistance (Ω), Luminosity (Lumens), and Heat Output (°C). You have data from 1,000 bulbs. How do you detect anomalies?

#### The Covariance Matrix

Σ = \[\[2.5, 1.8, 0.9\], \[1.8, 3.2, 1.4\], \[0.9, 1.4, 1.8\]\]

The off-diagonal values (1.8, 0.9, 1.4) are positive, meaning all three properties are positively correlated. A brighter bulb is also hotter.

#### Mahalanobis Distance (Anomaly Detection)

DM\=(x−μ)TΣ−1(x−μ)D\_M = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}

Unlike Euclidean distance (which ignores correlations), Mahalanobis distance accounts for the covariance structure. A bulb that's hot but not bright is an outlier, even if each value alone seems normal.

## [PCA](https://www.tensortonic.com/ml-math/linear-algebra/pca) (Principal Component Analysis)

PCA is literally just finding the eigenvectors of the Covariance Matrix and projecting data onto the top ones.

### The Algorithm

1.  Center the data (subtract mean from each column).
2.  Compute Covariance Matrix Σ\=1n−1XTX\\Sigma = \\frac{1}{n-1} X^T X.
3.  Find Eigenvectors and Eigenvalues of Σ\\Sigma.
4.  Sort by Eigenvalues (Largest = Most Variance = PC1).
5.  Project data: Z\=XVZ = XV, where VV is the matrix of top kk eigenvectors.

### Variance Explained

The proportion of variance explained by PC ii is λi/∑λj\\lambda\_i / \\sum \\lambda\_j. If the first 2 PCs explain 95% of variance, you can visualize 100D data in 2D with minimal information loss.

### Whitening / Sphering

After PCA, scale each PC by 1/λi1/\\sqrt{\\lambda\_i}. This transforms the ellipsoid back into a sphere (uncorrelated, unit variance). Common preprocessing for neural networks.