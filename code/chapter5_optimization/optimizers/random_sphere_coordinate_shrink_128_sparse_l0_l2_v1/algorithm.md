# Random Sphere With Coordinate Shrink, 128 Directions, Sparse L0-L2 Selection

## Intuition

This variant uses the same 128-direction random sphere and coordinate-shrink
candidate generator as the matched Chapter 5 sphere-shrink run. It changes only
the final selection rule: among valid surrogate threshold crossings, it first
minimizes the number of active coordinates and then breaks ties by normalized
\(L_2\) distance.

## Configuration

| Parameter | Value |
|-----------|-------|
| Sphere radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 128 |
| Sphere seed | `26001 + blocked index` |
| Shrink sweeps | 10 |
| Coordinates per sweep | 13, in Chapter 5 feature order |
| Binary-search steps per coordinate | 18 |
| Sparse selection | \(1\cdot L_0 \rightarrow 1\cdot L_2\) |
| Gradient calls | 0 |
| Success criterion | valid candidate with `P(assemblable) >= tau` |

## Inputs And Outputs

The optimizer uses the active `blocked_200_v1` starts and writes standard
Chapter 5 schema-v1 artifacts.

## Mathematical Description

For each radius \(r\), draw unit directions:

$$
u = \frac{\epsilon}{\|\epsilon\|_2}, \quad \epsilon_i \sim \mathcal{N}(0,1)
$$

and evaluate \(x(r,u)_i = x_{0,i} + r u_i \sigma_i\). After the first successful
sphere radius, coordinate shrink greedily moves each coordinate back toward the
blocked start, using binary search when full restoration would lose validity or
the surrogate threshold. The selected final candidate is:

$$
\arg\min_{x \in S} \left(L_0(x-x_0), L_2(x-x_0)\right),
$$

where \(S\) is the set of valid candidates with \(P(y=1\mid x)\geq\tau\).

## Failure Modes

- If the random sphere stage misses a crossing, sparse selection has no
  successful candidate to choose.
- Minimizing \(L_0\) can accept a larger Euclidean edit than the original
  nearest-\(L_2\) selection.
- The greedy shrink order can miss a better joint reduction.

## Thesis Interpretation

This row separates the effect of the random-sphere candidate generator from the
choice of final proximity norm.
