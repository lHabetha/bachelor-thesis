# Random Sphere Expanding, 128 Directions

## Intuition

This variant is the expanding random-sphere baseline with 128 random directions
per radius. It tests whether additional unstructured directions improve coverage
or distance without adding coordinate-axis or gradient assumptions.

## Configuration

| Parameter | Value |
|-----------|-------|
| Radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 128 |
| Max evaluations | 1409 (`1 + 11 * 128`) |
| Seed | `26001 + blocked index` |
| Gradient calls | 0 |
| Success criterion | `P(assemblable) >= tau` |
| Selection rule | nearest valid surrogate crossing |
| Early stop | after a full radius if any valid crossing has been found |

## Inputs And Outputs

The optimizer uses the active `blocked_200_v1` starts and writes public viewer schema v1 artifacts.

## Mathematical Description

For each radius \(r\), draw unit directions:

$$
u = \frac{\epsilon}{\|\epsilon\|_2}, \quad \epsilon_i \sim \mathcal{N}(0,1)
$$

and evaluate:

$$
x(r,u)_i = x_{0,i} + r u_i \sigma_i
$$

The selected candidate is the nearest valid surrogate threshold crossing among
all samples evaluated up to the first successful radius.

## Failure Modes

- Random directions can still miss narrow successful regions even with more
  samples.
- Larger sample counts increase compute without adding directed information.
- Some sampled candidates can leave the validity-constrained design space.

## Thesis Interpretation

This row measures whether doubling the default 64 random directions materially
improves the unstructured sphere baseline.
