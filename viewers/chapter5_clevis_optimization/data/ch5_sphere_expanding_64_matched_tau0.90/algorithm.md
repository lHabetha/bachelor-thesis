# Random Sphere Expanding Baseline

## Intuition

This is the clean gradient-free baseline. Around each blocked start, it samples
uniform random directions on expanding fixed-radius sphere shells in
benchmark-normalized parameter space. If random search needs many more
evaluations or much larger edits than gradient search, the thesis can say the MLP
gradient is doing real optimization work rather than merely finding any nearby
valid point.

## Configuration

| Parameter | Value |
|-----------|-------|
| Radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 64 |
| Max evaluations | 705 (`1 + 11 * 64`) |
| Seed | `26001 + blocked index` |
| Gradient calls | 0 |
| Success criterion | `P(assemblable) >= tau` |
| Selection rule | nearest valid surrogate crossing |
| Early stop | after a full radius if any valid crossing has been found |

## Inputs And Outputs

The optimizer uses the same active `blocked_200_v1` blocked starts and MLP artifact as the
gradient methods. It writes public viewer schema v1 artifacts.

## Mathematical Description

For each radius \(r\), draw unit directions:

$$
u = \\frac{\\epsilon}{\\|\\epsilon\\|_2}, \\quad \\epsilon_i \\sim \\mathcal{N}(0,1)
$$

and evaluate:

$$
x(r,u)_i = x_{0,i} + r u_i \\sigma_i
$$

Only valid candidates receive MLP probabilities at their candidate coordinates;
invalid candidates are recorded with the start probability and cannot be selected
as successes. The first radius containing at least one valid surrogate crossing
can terminate the search, but the selected candidate is always the nearest valid
crossing among all samples evaluated so far. If no crossing exists at any radius,
the fallback is the valid recorded candidate with highest surrogate probability,
using larger normalized distance only as a tie-breaker. If no valid non-start candidate exists, the optimizer records `no_valid_step` and keeps the valid start as the selected final design.

## Failure Modes

- In 13 dimensions, random directions rarely align with the useful repair
  direction, so success can require large radii or many samples.
- The method is stochastic, although deterministic seeds make runs reproducible.
- Some sampled candidates leave the validity-constrained design space.

## Thesis Interpretation

This optimizer measures the dimensionality penalty of not using gradients. It is
the most important negative control for claims that the surrogate gradient is
meaningful.
