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
- Distance: benchmark-normalized L2 on `blocked_200_v1`.

## Inputs And Outputs

Inputs are one blocked benchmark start, a `predict_proba` surrogate, and the clevis validity checks. Outputs use the public viewer trajectory schema.

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


## Run-Specific Model Note

This run pairs the optimizer with an XGBoost tree surrogate checkpoint from `results/chapter5_optimization/checkpoints/` as a gradient-free control (`tab:tree-gradient-free-control`). Optimizer mechanics are unchanged; probabilities come from the tree model, not from MLP input gradients.
