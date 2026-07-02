# Random Sphere Expanding, 256 Directions

## Intuition

This variant is the expanding random-sphere baseline with 256 random directions
per radius. It tests whether a much denser unstructured sample of each sphere can
close the gap to coordinate-aware search.

## Configuration

| Parameter | Value |
|-----------|-------|
| Radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 256 |
| Max evaluations | 2817 (`1 + 11 * 256`) |
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

- Dense random sampling can still miss narrow or highly structured successful
  regions.
- Compute grows linearly with the number of random directions per radius.
- Some sampled candidates can leave the validity-constrained design space.

## Thesis Interpretation

This row measures whether brute-force random-direction density can compensate
for not testing coordinate axes explicitly.
