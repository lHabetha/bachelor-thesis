# Adaptive Multi-Step Penalized Gradient

## Intuition

This optimizer combines adaptive gradient ascent with a proximity penalty. It
tries to raise surrogate assemblability while discouraging unnecessary movement
away from the blocked start.

## Configuration

| Parameter | Value |
|-----------|-------|
| Step rule | bold-driver backtracking |
| Distance penalty | 0.10 |
| Maximum iterations | 180 |
| Maximum evaluations | 900 |
| Crossing refinement | binary bracketing after the first valid threshold crossing |
| Selection rule | nearest valid crossing after bracketing |
| Oracle use | reporting only |

## Inputs And Outputs

Standard clevis repair benchmark inputs and public viewer schema v1 outputs.

## Mathematical Description

The local direction is the normalized gradient of a proximity-regularized
surrogate objective:

$$
d_t =
\frac{\nabla_{\tilde{x}} P(y=1\mid x_t)-\lambda(\tilde{x}_t-\tilde{x}_0)}
{\|\nabla_{\tilde{x}} P(y=1\mid x_t)-\lambda(\tilde{x}_t-\tilde{x}_0)\|_2}.
$$

The optimizer proposes \(x_{t+1}=x_t+\alpha_t d_t\sigma\). Invalid or
non-improving candidates trigger backtracking; strong accepted progress expands
the next step up to a local cap. After the first valid threshold crossing, binary
bracketing searches the last accepted segment for a closer crossing.

## Failure Modes

- A large penalty can prevent difficult starts from reaching the threshold.
- A small penalty can behave similarly to dense unpenalized gradient repair.
- Validity boundaries can force repeated backtracking and stop progress.

## Thesis Interpretation

The penalized adaptive variant exposes the trade-off between verified repair
coverage and normalized design-change distance.
