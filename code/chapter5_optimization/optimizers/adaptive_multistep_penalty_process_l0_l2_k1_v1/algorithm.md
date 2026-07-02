# Adaptive Penalty Process L0-L2, Top-1

## Intuition

This optimizer keeps the adaptive multi-step penalty mechanism, but adds sparsity
pressure during the optimization process. At each step it computes the usual
penalized adaptive direction and keeps only the strongest normalized coordinate.
There is no post-crossing coordinate restoration stage.

## Configuration

| Parameter | Value |
|-----------|-------|
| Step rule | `bold_backtrack` |
| L2 proximity penalty | `0.10` |
| L0 process pressure | keep top 1 direction coordinate |
| L0 objective penalty | active-coordinate fraction weight `0.03` |
| Max iterations | 180 |
| Max evaluations | 900 |
| Sparse endpoint restoration | none |
| Success criterion | valid candidate with `P(assemblable) >= tau` |

## Inputs And Outputs

The optimizer uses the active `blocked_200_v1` starts and writes standard
Chapter 5 schema-v1 artifacts.

## Mathematical Description

The dense adaptive penalty direction is

$$
d = \nabla_x P(y=1\mid x)\odot\sigma - \lambda_2 \frac{x-x_0}{\sigma}.
$$

The process-sparse variant keeps only the largest absolute component of \(d\)
before normalizing the update direction. Candidate acceptance uses the penalized
objective

$$
P(y=1\mid x) - \lambda_2 \lVert x-x_0\rVert_2
- \lambda_0 \frac{\lVert x-x_0\rVert_0}{13}.
$$

## Failure Modes

- A single-coordinate direction can miss repairs that require coupled changes.
- The \(L_0\) term is a process heuristic rather than a differentiable norm.
- Sparse directions can improve interpretability while reducing coverage.

## Thesis Interpretation

This variant tests whether L0/L2 pressure applied during optimization is a better
sparse repair mechanism than post-hoc endpoint cleanup.
