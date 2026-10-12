**Title:** Vector Fields: Visualizing Gradient Flows

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/vector-fields](https://www.tensortonic.com/ml-math/calculus/vector-fields)

---

# Page Structure Map
```text
Vector Fields: Visualizing Gradient Flows
├── Fields in Space
├── Gradient Fields
│   ├── Conservative Fields
│   └── Gradient Descent as Flow
├── Interactive Simulator
├── Divergence
│   ├── div F > 0
│   ├── div F < 0
│   └── div F = 0
├── Curl
│   └── Gradient Fields Have Zero Curl
├── Case Study: Navigating Loss Landscapes
└── ML Applications
    ├── Gradient Flow
    ├── Normalizing Flows
    ├── Physics-Informed NNs
    └── Score Matching (Diffusion)
```

---

## Fields in Space

A **vector field** assigns a vector to every point in space. Think of wind patterns on a weather map, or water currents in the ocean. At each location, there is a direction and magnitude.

In ML, the most important vector field is the **gradient field**. At every point in parameter space, the gradient tells us which direction increases the loss fastest.

F⃗(x,y)\=P(x,y)i^+Q(x,y)j^\\vec{F}(x, y) = P(x,y)\\hat{i} + Q(x,y)\\hat{j}

A 2D vector field: at each point (x,y), there is a vector with components P and Q.

## Gradient Fields

The gradient of a scalar function forms a vector field. If f(x,y)f(x,y) is a loss surface, then ∇f\\nabla f is the gradient field.

∇f\=∂f∂xi^+∂f∂yj^\\nabla f = \\frac{\\partial f}{\\partial x}\\hat{i} + \\frac{\\partial f}{\\partial y}\\hat{j}

### Conservative Fields

Gradient fields are **conservative**. The line integral between two points is path independent. This means there is a well defined "potential" (the loss function) that we are descending.

### Gradient Descent as Flow

Gradient descent follows the streamlines of the negative gradient field: dθdt\=−∇L(θ)\\frac{d\\theta}{dt} = -\\nabla L(\\theta). We flow "downhill" toward the minimum.

## Interactive Simulator

Explore different vector fields. Toggle streamlines to see how a particle would flow through the field.

## Divergence

Divergence measures how much a vector field "spreads out" at a point. It is a scalar field derived from a vector field.

∇⋅F⃗\=∂P∂x+∂Q∂y\\nabla \\cdot \\vec{F} = \\frac{\\partial P}{\\partial x} + \\frac{\\partial Q}{\\partial y}

#### div F > 0

Source: vectors spread outward

#### div F < 0

Sink: vectors converge inward

#### div F = 0

Incompressible: no net flow

## Curl

Curl measures the rotation or "swirl" of a vector field around a point.

∇×F⃗\=(∂Q∂x−∂P∂y)k^\\nabla \\times \\vec{F} = \\left(\\frac{\\partial Q}{\\partial x} - \\frac{\\partial P}{\\partial y}\\right)\\hat{k}

In 2D, curl gives a scalar (the z-component of the 3D curl).

### Gradient Fields Have Zero Curl

If F⃗\=∇f\\vec{F} = \\nabla f, then ∇×F⃗\=0\\nabla \\times \\vec{F} = 0. This is because mixed partials are equal: ∂2f∂x∂y\=∂2f∂y∂x\\frac{\\partial^2 f}{\\partial x \\partial y} = \\frac{\\partial^2 f}{\\partial y \\partial x}.

## Case Study: Navigating Loss Landscapes

You train a neural network to predict bulb lifespan. The loss function L(θ) defines a surface in high-D parameter space. Let's understand gradient descent as flow.

Step 1: Define the Loss Surface

The loss function L(θ)L(\\theta) maps parameters to a scalar loss value. This creates a "landscape" in parameter space.

Step 2: Compute the Gradient Field

At every point θ, compute ∇L(θ)\\nabla L(\\theta). This vector points "uphill" toward increasing loss.

∇L\=\[∂L∂θ1,∂L∂θ2,…\]\\nabla L = \\left\[\\frac{\\partial L}{\\partial \\theta\_1}, \\frac{\\partial L}{\\partial \\theta\_2}, \\ldots\\right\]

Step 3: Follow the Negative Gradient

Move in direction −∇L(θ)\-\\nabla L(\\theta) to descend the loss. This is gradient descent: flowing downhill through the gradient field.

Step 4: Watch for Saddle Points

Where ∇L\=0\\nabla L = 0, we've found a critical point. The Hessian eigenvalues tell us if it's a minimum (all positive) or saddle (mixed signs).

## ML Applications

### Gradient Flow

Continuous-time gradient descent: dθ/dt = -∇L. The solution traces a path through the gradient field. Used in Neural ODE analysis.

### Normalizing Flows

Transform probability distributions using invertible vector fields. The change in density involves the determinant of the Jacobian.

### Physics-Informed NNs

Encode PDEs (involving div, curl, grad) as loss terms. The network learns solutions to physical equations.

### Score Matching (Diffusion)

Learn the score function ∇log p(x) via a neural network. This is the gradient of the log-density. Diffusion models use this to generate samples.