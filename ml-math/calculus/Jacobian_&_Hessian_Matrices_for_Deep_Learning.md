**Title:** Jacobian & Hessian Matrices for Deep Learning

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/jacobian-hessian](https://www.tensortonic.com/ml-math/calculus/jacobian-hessian)

---

# Page Structure Map
```text
Jacobian & Hessian Matrices for Deep Learning
├── Introduction
│   ├── Jacobian (First Order)
│   └── Hessian (Second Order)
├── The Derivative Hierarchy
├── The Jacobian Matrix
│   └── Geometric Meaning
├── Jacobian: Worked Example
├── Interactive: Jacobian in Action
├── The Hessian Matrix
│   ├── Symmetric
│   └── Curvature
├── Hessian: Worked Example
├── Interactive: Curvature & Critical Points
├── Critical Points: The Eigenvalue Test
│   └── Why This Matters
└── ML Applications
    ├── Backpropagation = Jacobian-Vector Products
    ├── Newton's Method
    ├── Hessian-Free Optimization
    └── Loss Landscape Analysis
```

---

## Introduction

In single-variable calculus, we have f'(x) (first derivative/slope) and f''(x) (second derivative/curvature). But neural networks operate on vectors with millions of dimensions.

When we generalize derivatives to vectors:

#### Jacobian (First Order)

Matrix of all first partial derivatives. Generalizes gradient to vector-valued functions.

#### Hessian (Second Order)

Matrix of all second partial derivatives. Captures curvature information.

## The Derivative Hierarchy

The type of derivative depends on the input and output dimensions:

| Input | Output | Derivative | Shape |
| --- | --- | --- | --- |
| Scalar (x) | Scalar (y) | Derivative | 1 x 1 |
| Vector (x) | Scalar (y) | Gradient | n x 1 |
| Vector (x) | Vector (y) | Jacobian | m x n |
| Vector (x) | Scalar (y) | Hessian (2nd) | n x n |

## The Jacobian Matrix

For a function f:Rn→Rm\\mathbf{f}: \\mathbb{R}^n \\to \\mathbb{R}^m (n inputs, m outputs), the Jacobian is an m x n matrix:

J\=\[∂f1∂x1∂f1∂x2⋯∂f1∂xn∂f2∂x1∂f2∂x2⋯∂f2∂xn⋮⋮⋱⋮∂fm∂x1∂fm∂x2⋯∂fm∂xn\]J = \\begin{bmatrix} \\frac{\\partial f\_1}{\\partial x\_1} & \\frac{\\partial f\_1}{\\partial x\_2} & \\cdots & \\frac{\\partial f\_1}{\\partial x\_n} \\\\ \\frac{\\partial f\_2}{\\partial x\_1} & \\frac{\\partial f\_2}{\\partial x\_2} & \\cdots & \\frac{\\partial f\_2}{\\partial x\_n} \\\\ \\vdots & \\vdots & \\ddots & \\vdots \\\\ \\frac{\\partial f\_m}{\\partial x\_1} & \\frac{\\partial f\_m}{\\partial x\_2} & \\cdots & \\frac{\\partial f\_m}{\\partial x\_n} \\end{bmatrix}

Row i = how output i changes with each input. Column j = how each output changes with input j.

#### Geometric Meaning

The Jacobian represents the best linear approximation to f near a point. It tells us how a small change in input δx\\delta x affects the output:

f(x+δx)≈f(x)+J⋅δx\\mathbf{f}(\\mathbf{x} + \\delta\\mathbf{x}) \\approx \\mathbf{f}(\\mathbf{x}) + J \\cdot \\delta\\mathbf{x}

## Jacobian: Worked Example

Consider a function from R2→R2\\mathbb{R}^2 \\to \\mathbb{R}^2:

f(x,y)\=\[x2+yxy\]\\mathbf{f}(x, y) = \\begin{bmatrix} x^2 + y \\\\ xy \\end{bmatrix}

Step 1: Compute all partial derivatives

∂f1∂x\=2x\\frac{\\partial f\_1}{\\partial x} = 2x

∂f1∂y\=1\\frac{\\partial f\_1}{\\partial y} = 1

∂f2∂x\=y\\frac{\\partial f\_2}{\\partial x} = y

∂f2∂y\=x\\frac{\\partial f\_2}{\\partial y} = x

Step 2: Assemble the Jacobian

J\=\[2x1yx\]J = \\begin{bmatrix} 2x & 1 \\\\ y & x \\end{bmatrix}

Step 3: Evaluate at a point (x=2, y=3)

J∣(2,3)\=\[4132\]J|\_{(2,3)} = \\begin{bmatrix} 4 & 1 \\\\ 3 & 2 \\end{bmatrix}

## Interactive: Jacobian in Action

See how the Jacobian provides a **linear approximation** to how outputs change with inputs. Move the base point and perturbation to explore.

## The Hessian Matrix

For a scalar-valued function f:Rn→Rf: \\mathbb{R}^n \\to \\mathbb{R} (like a loss function), the Hessian is the n x n matrix of second-order partial derivatives:

H\=∇2f\=\[∂2f∂x12∂2f∂x1∂x2⋯∂2f∂x2∂x1∂2f∂x22⋯⋮⋮⋱\]H = \\nabla^2 f = \\begin{bmatrix} \\frac{\\partial^2 f}{\\partial x\_1^2} & \\frac{\\partial^2 f}{\\partial x\_1 \\partial x\_2} & \\cdots \\\\ \\frac{\\partial^2 f}{\\partial x\_2 \\partial x\_1} & \\frac{\\partial^2 f}{\\partial x\_2^2} & \\cdots \\\\ \\vdots & \\vdots & \\ddots \\end{bmatrix}

#### Symmetric

For continuous second partials:

∂2f∂xi∂xj\=∂2f∂xj∂xi\\frac{\\partial^2 f}{\\partial x\_i \\partial x\_j} = \\frac{\\partial^2 f}{\\partial x\_j \\partial x\_i}

So H\=HTH = H^T

#### Curvature

The Hessian captures how the gradient itself changes. It describes the "bowl shape" of the function.

## Hessian: Worked Example

Consider a loss function f(x,y)\=x2+3xy+y2f(x, y) = x^2 + 3xy + y^2:

Step 1: First partial derivatives (gradient)

∇f\=\[2x+3y3x+2y\]\\nabla f = \\begin{bmatrix} 2x + 3y \\\\ 3x + 2y \\end{bmatrix}

Step 2: Second partial derivatives

∂2f∂x2\=2\\frac{\\partial^2 f}{\\partial x^2} = 2

∂2f∂x∂y\=3\\frac{\\partial^2 f}{\\partial x \\partial y} = 3

∂2f∂y∂x\=3\\frac{\\partial^2 f}{\\partial y \\partial x} = 3

∂2f∂y2\=2\\frac{\\partial^2 f}{\\partial y^2} = 2

Step 3: Assemble Hessian

H\=\[2332\]H = \\begin{bmatrix} 2 & 3 \\\\ 3 & 2 \\end{bmatrix}

Note: constant because f is quadratic.

## Interactive: Curvature & Critical Points

Adjust the Hessian eigenvalues to see how they determine the shape of the loss surface and classify critical points.

## Critical Points: The Eigenvalue Test

At a critical point (where gradient = 0), the Hessian's eigenvalues tell us the nature of that point:

All positive

Function curves UP in all directions. **Local minimum.**

All negative

Function curves DOWN in all directions. **Local maximum.**

Mixed signs

Curves up in some directions, down in others. **Saddle point.**

#### Why This Matters

In high-dimensional neural network loss landscapes, saddle points are far more common than local minima. Understanding the Hessian helps explain why optimization can stall and why momentum-based methods help.

## ML Applications

#### Backpropagation = Jacobian-Vector Products

When computing gradients through a neural network, each layer contributes a Jacobian. The chain rule becomes: ∇xL\=J1TJ2T⋯JnT∇yL\\nabla\_x L = J\_1^T J\_2^T \\cdots J\_n^T \\nabla\_y L.

#### Newton's Method

Second-order optimization: θnew\=θ−H−1∇f\\theta\_{new} = \\theta - H^{-1} \\nabla f. Uses curvature to take smarter steps. O(n³) to compute.

#### Hessian-Free Optimization

Clever algorithms that use Hessian information without computing the full matrix. Conjugate gradient methods can compute Hessian-vector products efficiently.

#### Loss Landscape Analysis

Researchers study Hessian eigenvalue distributions. Sharp minima (large eigenvalues) tend to generalize worse than flat minima (small eigenvalues).