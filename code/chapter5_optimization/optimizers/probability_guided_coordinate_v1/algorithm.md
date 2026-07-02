# Probability-Guided Coordinate v1

## Intuition

This optimizer is designed for surrogates that expose probabilities but no useful
input gradient. It probes each coordinate direction at increasing normalized step
magnitudes, then selects the nearest valid candidate crossing the surrogate
threshold. If no threshold crossing exists, the fallback candidate is chosen by
probability gain minus a distance penalty.

## Configuration

- Directions: 26 signed coordinate axes.
- Step magnitudes: log-spaced normalized distances from `1e-4` to `100`.
- Threshold: runner-provided `tau`.
- Distance: Chapter 5 benchmark-normalized L2.

## Inputs And Outputs

Inputs are one blocked benchmark start, a `predict_proba` surrogate, and the
Chapter 5 validity checks. Outputs use the standard Chapter 5 trajectory schema
and viewer format.

## Mathematical Description

For each unit coordinate direction \(d_i\), evaluate

$$
x = x_0 + \alpha d_i \odot \sigma
$$

where \(\sigma\) are the benchmark normalization standard deviations. If no valid
candidate reaches \(P(y=1|x) \ge \tau\), choose the fallback maximizing

$$
\left(P(y=1|x)-P(y=1|x_0)\right) - 0.08 \lVert x-x_0 \rVert_2.
$$

## Failure Modes

Axis-aligned search can miss coupled repairs. Tree probabilities can also be
piecewise constant, so many probes may show no change until a split boundary is
crossed.

## Thesis Interpretation

This is a clean non-gradient comparison against MLP-gradient repair. If it works,
the tree surrogate's probability landscape is useful even without gradients; if it
fails, that supports the value of differentiability for inverse design.
