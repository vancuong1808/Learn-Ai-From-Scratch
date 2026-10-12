**Title:** Matrix Calculus: Gradients in High Dimensions

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/linear-algebra/matrix-calculus](https://www.tensortonic.com/ml-math/linear-algebra/matrix-calculus)

---

# Page Structure Map
```text
Matrix Calculus: Gradients in High Dimensions
├── Beyond Scalar Calculus
│   └── Why It Matters for ML
├── Layout Conventions
│   ├── Numerator Layout
│   └── Denominator Layout
├── Gradients
│   └── Key Gradient Identities
├── Jacobian Matrix
│   └── Linear Transformation View
├── Interactive: Jacobian as Linearization
├── Hessian Matrix
│   ├── H positive definite
│   ├── H negative definite
│   ├── H indefinite
│   └── H singular
├── Interactive Rules
├── Matrix Chain Rule
│   └── Backpropagation is Chain Rule
├── Case Study: Linear Regression
│   ├── The Problem
│   ├── The Loss Function
│   └── The Gradient
└── ML Applications
    ├── Custom Autograd
    ├── Natural Gradient Descent
    ├── Variational Inference
    └── Gaussian Processes
```

---

## Beyond Scalar Calculus

In standard calculus, you differentiate a scalar function f:R→Rf: \\mathbb{R} \\to \\mathbb{R}. The derivative is a single number.

In Machine Learning, we deal with functions of **many variables** (neural networks have millions of parameters) that output vectors, matrices, or scalars. Matrix Calculus provides the notation and rules to compute these derivatives efficiently.

### Why It Matters for ML

Every time you call `loss.backward()` in PyTorch, you are computing matrix derivatives. Understanding matrix calculus lets you derive gradients by hand, debug backpropagation, and design custom layers.

## Layout Conventions

There are two conventions for arranging derivatives: **Numerator Layout** and **Denominator Layout**. ML typically uses **Numerator Layout** (also called Jacobian layout).

### Numerator Layout

The derivative has the same shape as the numerator. If y∈Rmy \\in \\mathbb{R}^m and x∈Rnx \\in \\mathbb{R}^n, then ∂y∂x\\frac{\\partial y}{\\partial x} is m×nm \\times n.

Shape(∂y∂x)\=Shape(y)×Shape(x)T\\text{Shape}(\\frac{\\partial y}{\\partial x}) = \\text{Shape}(y) \\times \\text{Shape}(x)^T

### Denominator Layout

The derivative has the same shape as the denominator transposed. Used in some optimization textbooks.

Shape(∂y∂x)\=Shape(x)T×Shape(y)\\text{Shape}(\\frac{\\partial y}{\\partial x}) = \\text{Shape}(x)^T \\times \\text{Shape}(y)

**Warning:** Always check which convention a paper or library uses. Transposing the wrong matrix leads to dimension mismatch bugs in backprop.

## Gradients

The gradient of a scalar function f:Rn→Rf: \\mathbb{R}^n \\to \\mathbb{R} is a vector of partial derivatives. In numerator layout, it is technically a row vector, but we often treat it as a column vector for convenience in update rules.

∇f\=\[∂f∂x1∂f∂x2⋮∂f∂xn\]\\nabla f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x\_1} \\\\ \\frac{\\partial f}{\\partial x\_2} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x\_n} \\end{bmatrix}

The gradient points in the direction of steepest ascent of the function.

### Key Gradient Identities

-   ∇x(aTx)\=a\\nabla\_x (a^T x) = a (Linear function)
-   ∇x(xTx)\=2x\\nabla\_x (x^T x) = 2x (Squared norm, [recall norms](https://www.tensortonic.com/ml-math/linear-algebra/dot-norm))
-   ∇x(xTAx)\=(A+AT)x\\nabla\_x (x^T A x) = (A + A^T)x (Quadratic form)
-   ∇x∣∣Ax−b∣∣2\=2AT(Ax−b)\\nabla\_x ||Ax - b||^2 = 2A^T(Ax - b) (Least squares)

## Jacobian Matrix

When the output is a vector f:Rn→Rmf: \\mathbb{R}^n \\to \\mathbb{R}^m, we need an m×nm \\times n matrix to capture all partial derivatives. This is the **Jacobian**.

J\=∂f∂x\=\[∂f1∂x1⋯∂f1∂xn⋮⋱⋮∂fm∂x1⋯∂fm∂xn\]J = \\frac{\\partial f}{\\partial x} = \\begin{bmatrix} \\frac{\\partial f\_1}{\\partial x\_1} & \\cdots & \\frac{\\partial f\_1}{\\partial x\_n} \\\\ \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial f\_m}{\\partial x\_1} & \\cdots & \\frac{\\partial f\_m}{\\partial x\_n} \\end{bmatrix}

### Linear Transformation View

The Jacobian tells you how the output space locally stretches and rotates. If f(x)\=Axf(x) = Ax, then J\=AJ = A. The derivative of a linear map is the map itself!

## Interactive: Jacobian as Linearization

Explore how the Jacobian matrix acts as a local linear approximation for a non-linear transformation. Notice how the grid lines locally look like the Jacobian vectors.

## Hessian Matrix

The Hessian is the matrix of second derivatives. It tells you about the curvature of a scalar function.

H\=∇2f\=\[∂2f∂x12∂2f∂x1∂x2⋯∂2f∂x2∂x1∂2f∂x22⋯⋮⋮⋱\]H = \\nabla^2 f = \\begin{bmatrix} \\frac{\\partial^2 f}{\\partial x\_1^2} & \\frac{\\partial^2 f}{\\partial x\_1 \\partial x\_2} & \\cdots \\\\ \\frac{\\partial^2 f}{\\partial x\_2 \\partial x\_1} & \\frac{\\partial^2 f}{\\partial x\_2^2} & \\cdots \\\\ \\vdots & \\vdots & \\ddots \\end{bmatrix}

#### H positive definite

Local minimum. Bowl curving up in all directions (convex).

#### H negative definite

Local maximum. Bowl curving down (concave).

#### H indefinite

Saddle point. Curves up in some directions, down in others.

#### H singular

Flat direction. Need higher order derivatives.

## Interactive Rules

Explore the three fundamental rules of matrix calculus: scalar-by-vector, vector-by-vector (Jacobian), and the chain rule.

## Matrix Chain Rule

The chain rule in matrix calculus is about multiplying Jacobians in the right order.

∂L∂x\=∂L∂y⋅∂y∂x\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial x}

For L\=f(g(x))L = f(g(x)), chain the Jacobians. Be careful with dimensions!

### Backpropagation is Chain Rule

A neural network is a composition of functions: L\=fn(fn−1(...f1(x)))L = f\_n(f\_{n-1}(...f\_1(x))).

Backprop computes gradients by starting from the loss and multiplying Jacobians backwards:

∂L∂x\=∂L∂fn⋅∂fn∂fn−1⋯∂f1∂x\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial f\_n} \\cdot \\frac{\\partial f\_n}{\\partial f\_{n-1}} \\cdots \\frac{\\partial f\_1}{\\partial x}

## Case Study: Linear Regression

#### The Problem

Predict bulb lifespan (y) from features (voltage, temperature, filament thickness). We have data matrix X∈Rn×dX \\in \\mathbb{R}^{n \\times d} and targets y∈Rny \\in \\mathbb{R}^n. Find optimal weights ww.

#### The Loss Function

L(w)\=∣∣Xw−y∣∣2\=(Xw−y)T(Xw−y)L(w) = ||Xw - y||^2 = (Xw - y)^T(Xw - y)

#### The Gradient

Using matrix calculus rules:

∇wL\=2XT(Xw−y)\\nabla\_w L = 2X^T(Xw - y)

Setting to zero gives the Normal Equations: XTXw\=XTyX^TXw = X^Ty. This is an exact solution!

## ML Applications

### Custom Autograd

When implementing a custom layer (e.g., in PyTorch or JAX), you must define the backward pass. This requires analytically deriving the Jacobian of your operation.

### Natural Gradient Descent

Uses the Fisher Information Matrix (expected Hessian of log-likelihood) to precondition gradients, correcting for the geometry of the parameter space.

### Variational Inference

Optimizing the ELBO in VAEs requires computing derivatives of log-determinants and trace operations, which are pure matrix calculus.

### Gaussian Processes

Optimizing kernel hyperparameters requires gradients of the marginal likelihood, involving derivatives of matrix inverses and determinants: ∂K−1∂θ\=−K−1∂K∂θK−1\\frac{\\partial K^{-1}}{\\partial \\theta} = -K^{-1} \\frac{\\partial K}{\\partial \\theta} K^{-1}.