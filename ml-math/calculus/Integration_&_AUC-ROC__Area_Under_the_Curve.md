**Title:** Integration & AUC-ROC: Area Under the Curve

**Author:** TensorTonic

**Date:** 1/1/2026

**Source:** [https://www.tensortonic.com/ml-math/calculus/integration-auc](https://www.tensortonic.com/ml-math/calculus/integration-auc)

---

# Page Structure Map
```text
Integration & AUC-ROC: Area Under the Curve
├── From Sums to Areas
├── Riemann Sums
│   ├── Left Riemann
│   ├── Right Riemann
│   └── Midpoint Rule
├── Interactive Simulator
├── AUC-ROC: The Area Under the ROC Curve
│   ├── AUC = 1.0
│   ├── AUC = 0.5
│   └── Probabilistic Interpretation
├── Case Study: Bulb Defect Classifier
├── Numerical Integration Methods
│   ├── Trapezoidal Rule
│   ├── Simpson's Rule
│   └── Monte Carlo Integration
└── ML Applications
    ├── AUC-ROC
    ├── AUC-PR
    ├── Expected Calibration Error
    └── ELBO in VAEs
```

---

## From Sums to Areas

Integration is the reverse of differentiation. While derivatives tell us the rate of change, integrals accumulate quantities over an interval. In ML, we use integrals to compute areas under curves like the ROC curve.

∫abf(x) dx\=lim⁡n→∞∑i\=1nf(xi)Δx\\int\_a^b f(x) \\, dx = \\lim\_{n \\to \\infty} \\sum\_{i=1}^{n} f(x\_i) \\Delta x

The definite integral is the limit of Riemann sums as rectangles become infinitely thin.

## Riemann Sums

The area under a curve is approximated by summing rectangles. The height of each rectangle is the function value, and width is Δx\=(b−a)/n\\Delta x = (b-a)/n.

#### Left Riemann

Use left endpoint of each interval for height. Underestimates for increasing functions.

#### Right Riemann

Use right endpoint. Overestimates for increasing functions.

#### Midpoint Rule

Use midpoint. Generally more accurate than left or right.

## Interactive Simulator

Adjust the number of rectangles and watch the Riemann sum converge to the true AUC as n increases.

## AUC-ROC: The Area Under the ROC Curve

The ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various classification thresholds. The Area Under this Curve (AUC) is a single number summarizing classifier performance.

AUC\=∫01TPR(FPR) d(FPR)AUC = \\int\_0^1 TPR(FPR) \\, d(FPR)

#### AUC = 1.0

Perfect classifier. The curve hugs the top left corner.

#### AUC = 0.5

Random classifier. The curve is the diagonal line. No discrimination power.

### Probabilistic Interpretation

AUC = P(classifier ranks a random positive higher than a random negative). If AUC = 0.8, given a random positive and negative sample, the model correctly ranks them 80% of the time.

## Case Study: Bulb Defect Classifier

You train a model to detect defective bulbs. It outputs a probability score from 0 to 1. Let's compute the AUC step by step.

Example model outputs:

Defective: \[0.9, 0.8, 0.7, 0.6\]Normal: \[0.4, 0.3, 0.2, 0.1\]

Step 1: Vary threshold from 0 to 1

At each threshold τ, predict "defective" if score ≥ τ. This gives different TPR/FPR pairs.

Step 2: Compute TPR and FPR at each threshold

TPR\=TPTP+FNTPR = \\frac{TP}{TP + FN}

True positives / All actual positives

FPR\=FPFP+TNFPR = \\frac{FP}{FP + TN}

False positives / All actual negatives

Step 3: Plot the ROC curve

Plot (FPR, TPR) for each threshold. Connect points to form curve from (0,0) to (1,1).

Step 4: Compute AUC using trapezoidal rule

AUC≈∑i(FPRi+1−FPRi)×TPRi+TPRi+12AUC \\approx \\sum\_{i} (FPR\_{i+1} - FPR\_i) \\times \\frac{TPR\_i + TPR\_{i+1}}{2}

For our example: AUC ≈ 0.94 (excellent discrimination)

## Numerical Integration Methods

#### Trapezoidal Rule

Use trapezoids instead of rectangles. Error: O(h2)O(h^2)

∫abf(x)dx≈h2\[f(a)+2∑f(xi)+f(b)\]\\int\_a^b f(x)dx \\approx \\frac{h}{2}\[f(a) + 2\\sum f(x\_i) + f(b)\]

#### Simpson's Rule

Fit parabolas to three points. Error: O(h4)O(h^4)

∫abf(x)dx≈h3\[f(a)+4∑f(xodd)+2∑f(xeven)+f(b)\]\\int\_a^b f(x)dx \\approx \\frac{h}{3}\[f(a) + 4\\sum f(x\_{odd}) + 2\\sum f(x\_{even}) + f(b)\]

#### Monte Carlo Integration

Sample random points and average. Scales well to high dimensions unlike grid methods.

## ML Applications

### AUC-ROC

Threshold independent evaluation for binary classifiers. Used by sklearn.metrics.roc\_auc\_score.

### AUC-PR

Precision Recall AUC. Better for imbalanced datasets where negatives dominate.

### Expected Calibration Error

Integrate the gap between predicted probability and actual frequency. Uses binning (discrete integration).

### ELBO in VAEs

The Evidence Lower Bound involves integrals over latent variables. Approximated via Monte Carlo sampling.