# Random Sphere With Coordinate Shrink, 128 Directions

## Intuition

This variant runs the random-sphere plus coordinate-shrink optimizer with 128
random directions per radius. It tests whether additional random crossings give
the shrink stage better candidates without adding explicit coordinate-axis rays
to the sphere search.

## Configuration

| Parameter | Value |
|-----------|-------|
| Sphere radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 128 |
| Sphere seed | `26001 + blocked index` |
| Shrink sweeps | 10 |
| Coordinates per sweep | 13, in clevis parameter feature order |
| Binary-search steps per coordinate | 18 |
| Gradient calls | 0 |
| Success criterion | valid candidate with `P(assemblable) >= tau` |
| Selection rule | nearest valid threshold-preserving candidate after shrink |

## Inputs And Outputs

The optimizer uses the active `blocked_200_v1` starts and writes public viewer schema v1 artifacts.

## Mathematical Description

For each radius \(r\), draw unit directions:

$$
u = \frac{\epsilon}{\|\epsilon\|_2}, \quad \epsilon_i \sim \mathcal{N}(0,1)
$$

and evaluate \(x(r,u)_i = x_{0,i} + r u_i \sigma_i\). After the first successful
sphere radius, coordinate shrink greedily moves each coordinate back toward the
blocked start, using binary search when full restoration would lose validity or
the surrogate threshold.

## Failure Modes

- More random directions can still miss narrow successful regions.
- The greedy shrink order can miss a better joint reduction.
- Shrink preserves the surrogate threshold, so final verification is still
  required.

## Thesis Interpretation

This row tests whether 128 random directions improve the random crossing enough
to change the final shrink result.
