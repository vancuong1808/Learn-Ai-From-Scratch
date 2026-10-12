**Title:** Partial Derivatives & Gradients for Machine Learning

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/partial-derivatives](https://www.tensortonic.com/ml-math/calculus/partial-derivatives)

---

# Page Structure Map
```text
Partial Derivatives & Gradients for Machine Learning
├── Introduction
│   └── Why Partial Derivatives Matter
├── Formal Definition
│   └── The Key Insight
├── Notation & Computation
│   └── Worked Example
├── Geometric Intuition
│   ├── Slicing the Surface
│   └── Walking East-West
├── The Gradient Vector
│   ├── Direction of Steepest Ascent
│   ├── Gradient Descent
│   └── Magnitude = Steepness
├── Interactive: The Gradient
├── Directional Derivatives
│   ├── Directional Derivative
│   └── Gradient is Perpendicular to Level Curves
├── Higher-Order Partial Derivatives
│   ├── Second-Order Partials
│   └── Clairaut's Theorem (Symmetry of Mixed Partials)
└── ML Applications
    ├── Backpropagation
    ├── Automatic Differentiation
    ├── Feature Importance (Saliency Maps)
    └── Regularization via Gradients
```

---

## Introduction

In single-variable calculus, the derivative dydx\\frac{dy}{dx} tells us the rate of change of y with respect to x. But neural networks have **millions** of parameters. A loss function might depend on weights w1,w2,…,w1000000w\_1, w\_2, \\ldots, w\_{1000000}.

We can't simply ask "what's the slope?" because the function slopes differently in every direction. Instead, we ask: _"How does the output change if I wiggle just **one** input while holding all others constant?"_

#### Why Partial Derivatives Matter

-   **Gradient Descent:** The gradient is a vector of partial derivatives.
-   **Backpropagation:** Computing partials for every weight in a neural network.
-   **Optimization:** Finding which direction reduces loss the most.
-   **Sensitivity Analysis:** Which inputs affect outputs the most?

## Formal Definition

Let f(x,y)f(x, y) be a function of two variables. The **partial derivative** with respect to x is:

∂f∂x\=lim⁡h→0f(x+h,y)−f(x,y)h\\frac{\\partial f}{\\partial x} = \\lim\_{h \\to 0} \\frac{f(x+h, y) - f(x, y)}{h}

Note the ∂\\partial symbol (partial) instead of d. This explicitly indicates that y is held constant.

Similarly, the partial with respect to y is:

∂f∂y\=lim⁡h→0f(x,y+h)−f(x,y)h\\frac{\\partial f}{\\partial y} = \\lim\_{h \\to 0} \\frac{f(x, y+h) - f(x, y)}{h}

#### The Key Insight

When computing ∂f∂x\\frac{\\partial f}{\\partial x}, treat y as if it were a constant number (like 5 or pi). Then differentiate with respect to x using your normal single-variable rules.

## Notation & Computation

Several notations are used for partial derivatives:

∂f∂x\\frac{\\partial f}{\\partial x}

Leibniz

fxf\_x

Subscript

∂xf\\partial\_x f

Operator

DxfD\_x f

D-notation

#### Worked Example

Let f(x,y)\=x2y+3y3+5f(x, y) = x^2 y + 3y^3 + 5. Find all partial derivatives.

Partial with respect to x:

Treat y as constant:

∂f∂x\=2xy+0+0\=2xy\\frac{\\partial f}{\\partial x} = 2xy + 0 + 0 = 2xy

Partial with respect to y:

Treat x as constant:

∂f∂y\=x2+9y2+0\=x2+9y2\\frac{\\partial f}{\\partial y} = x^2 + 9y^2 + 0 = x^2 + 9y^2

## Geometric Intuition

Consider a 3D surface z\=f(x,y)z = f(x, y) representing a landscape with hills and valleys. Standing at a point (x, y), you can walk in infinitely many directions.

#### Slicing the Surface

When we compute ∂f∂x\\frac{\\partial f}{\\partial x}, we "slice" the 3D surface with a vertical plane where y = constant.

This slice creates a 2D curve. The partial derivative is the slope of this curve at our point.

#### Walking East-West

∂f∂x\\frac{\\partial f}{\\partial x} tells you the slope if you walk strictly in the x-direction (East-West).

∂f∂y\\frac{\\partial f}{\\partial y} tells you the slope if you walk strictly in the y-direction (North-South).

## The Gradient Vector

The **gradient** collects all partial derivatives into a single vector. For a function f:Rn→Rf: \\mathbb{R}^n \\to \\mathbb{R}:

∇f\=\[∂f∂x1∂f∂x2⋮∂f∂xn\]\\nabla f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x\_1} \\\\ \\frac{\\partial f}{\\partial x\_2} \\\\ \\vdots \\\\ \\frac{\\partial f}{\\partial x\_n} \\end{bmatrix}

The symbol ∇\\nabla is called "nabla" or "del".

#### Direction of Steepest Ascent

The gradient ∇f\\nabla f points in the direction where f increases fastest. If you're standing on a hill, the gradient points uphill.

#### Gradient Descent

To minimize f (like a loss function), move in the **opposite** direction: θnew\=θold−α∇f\\theta\_{new} = \\theta\_{old} - \\alpha \\nabla f

#### Magnitude = Steepness

The magnitude ∣∣∇f∣∣||\\nabla f|| tells you how steep the slope is. Large gradient = steep terrain = big updates.

## Interactive: The Gradient

For the function f(x,y)\=x2+y2f(x, y) = x^2 + y^2, the gradient is:

∇f\=\[∂f∂x∂f∂y\]\=\[2x2y\]\\nabla f = \\begin{bmatrix} \\frac{\\partial f}{\\partial x} \\\\ \\frac{\\partial f}{\\partial y} \\end{bmatrix} = \\begin{bmatrix} 2x \\\\ 2y \\end{bmatrix}

Drag the point to see how the gradient always points toward steepest ascent (away from the minimum at the origin).

## Directional Derivatives

Partial derivatives tell us the slope along coordinate axes. But what if we want to know the slope in an arbitrary direction?

#### Directional Derivative

The rate of change of f in direction of unit vector u\\mathbf{u} is:

Duf\=∇f⋅u\=∣∣∇f∣∣cos⁡(θ)D\_{\\mathbf{u}}f = \\nabla f \\cdot \\mathbf{u} = ||\\nabla f|| \\cos(\\theta)

where θ\\theta is the angle between the gradient and direction u.

Maximum

When u = direction of gradient, cos(0) = 1. Maximum increase.

Minimum

When u = opposite of gradient, cos(180) = -1. Maximum decrease.

Zero

When u perpendicular to gradient, cos(90) = 0. Level curve (contour).

#### Gradient is Perpendicular to Level Curves

The gradient is always perpendicular to level curves (contours where f is constant). This is because moving along a level curve means zero change in f, which requires cos(theta) = 0.

## Higher-Order Partial Derivatives

We can take partial derivatives of partial derivatives. These second-order partials are crucial for optimization (You can read more about it in the [Hessian matrix](https://www.tensortonic.com/ml-math/calculus/jacobian-hessian)).

#### Second-Order Partials

Pure second partial

∂2f∂x2\=∂∂x(∂f∂x)\\frac{\\partial^2 f}{\\partial x^2} = \\frac{\\partial}{\\partial x}\\left(\\frac{\\partial f}{\\partial x}\\right)

Mixed partial

∂2f∂x∂y\=∂∂x(∂f∂y)\\frac{\\partial^2 f}{\\partial x \\partial y} = \\frac{\\partial}{\\partial x}\\left(\\frac{\\partial f}{\\partial y}\\right)

#### Clairaut's Theorem (Symmetry of Mixed Partials)

If the mixed partials are continuous, then the order of differentiation doesn't matter:

∂2f∂x∂y\=∂2f∂y∂x\\frac{\\partial^2 f}{\\partial x \\partial y} = \\frac{\\partial^2 f}{\\partial y \\partial x}

This is why the Hessian matrix is symmetric for most functions we encounter.

## ML Applications

#### Backpropagation

When training a neural network, we need ∂L∂wi\\frac{\\partial L}{\\partial w\_i} for every weight w\_i. The chain rule lets us compute these efficiently by propagating gradients backward through the network.

#### Automatic Differentiation

PyTorch and TensorFlow build computation graphs and use the chain rule to compute all partial derivatives automatically. When you call `loss.backward()`, it computes the gradient of loss with respect to all parameters.

#### Feature Importance (Saliency Maps)

For a trained model, ∂output∂inputi\\frac{\\partial \\text{output}}{\\partial \\text{input}\_i} tells us how sensitive the output is to each input feature. This creates "saliency maps" showing which pixels matter most for image classification.

#### Regularization via Gradients

Some regularization techniques penalize large gradients. For example, spectral normalization in GANs constrains the Lipschitz constant (maximum gradient magnitude) of the discriminator.