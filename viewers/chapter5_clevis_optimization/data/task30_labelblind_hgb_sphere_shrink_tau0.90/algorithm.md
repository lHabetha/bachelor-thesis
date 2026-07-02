# Random Sphere With Coordinate Shrink

## Intuition

This optimizer tests whether the random-sphere baseline can be made more useful
for the thesis goal of small edits. The first stage is gradient-free random
sphere expansion: find any valid candidate near the blocked start whose MLP
probability crosses `tau`. The second stage then pulls that candidate back toward
the original assembly one coordinate at a time, while preserving validity and
`P(assemblable) >= tau`.

## Configuration

| Parameter | Value |
|-----------|-------|
| Sphere radii | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10` |
| Samples per radius | 64 |
| Sphere seed | `26301 + blocked index` |
| Shrink sweeps | 10 |
| Coordinates per sweep | 13, in clevis parameter feature order |
| Binary-search steps per coordinate | 18 |
| Gradient calls | 0 |
| Success criterion | valid candidate with `P(assemblable) >= tau` |
| Selection rule | nearest valid threshold-preserving candidate after shrink |

## Inputs And Outputs

Inputs are the active `blocked_200_v1` starts and the thesis-facing MLP checkpoint. Outputs follow public viewer schema v1: `manifest.json`,
`trajectories.json`, `viewer_data.json`, `statistics.json`, and `algorithm.md`.

The optimizer never uses exact oracle labels for selecting candidates. The exact
formula oracle is attached after selection by the workbench for reporting only.

## Mathematical Description

### Stage 1: random sphere expansion

For each radius \(r\), draw unit directions:

$$
u = \frac{\epsilon}{\|\epsilon\|_2}, \quad \epsilon_i \sim \mathcal{N}(0,1)
$$

and evaluate:

$$
x(r,u)_i = x_{0,i} + r u_i \sigma_i
$$

where \(\sigma_i\) is the active benchmark standard deviation for parameter
\(i\). The search stops after the first radius where any valid sampled candidate
has \(P(x) \ge \tau\). Among all samples evaluated so far, the selected sphere
candidate is the valid threshold crossing with the smallest normalized distance.

If no valid sphere candidate crosses `tau`, the optimizer uses the standard
Chapter 5 fallback: select the valid recorded candidate with highest surrogate
probability, using larger normalized distance only as a tie-breaker. If no valid
non-start candidate exists, it records `no_valid_step` and keeps the valid start
as the selected final design.

### Stage 2: coordinate shrink toward the original assembly

Let \(x_c\) be the current valid threshold-crossing candidate. For 10 sweeps, the
optimizer loops over all 13 coordinates. For coordinate \(j\), it tries to move
only that coordinate from its current value \(x_{c,j}\) back toward the original
value \(x_{0,j}\):

$$
x_j(\alpha) = x_{0,j} + \alpha (x_{c,j} - x_{0,j}), \quad \alpha \in [0,1]
$$

All other coordinates remain fixed during that coordinate probe. The target
\(\alpha=0\) means "restore this coordinate completely to the original blocked
assembly value." The target \(\alpha=1\) means "keep the current optimized
coordinate value."

If fully restoring the coordinate remains valid and keeps \(P(x) \ge \tau\), that
full restoration is accepted. Otherwise, the optimizer performs 18 deterministic
binary-search probes to find the smallest acceptable \(\alpha\), i.e. the
closest-to-original coordinate value that still satisfies:

$$
\text{valid}(x) = \text{true}, \quad P(x) \ge \tau
$$

This is repeated coordinate by coordinate and sweep by sweep. The final selected
candidate is the closest valid threshold-preserving point reached by this shrink
process. "Shrink" therefore means moving back toward the original coordinate
value; it does not necessarily mean numerically decreasing the parameter.

Only valid candidates receive MLP probabilities at their candidate coordinates.
Invalid candidates are recorded with the start probability and cannot be selected
as successes.

## Failure Modes

- Random sphere expansion may fail to find a threshold crossing, especially for
  starts deep in the blocked region.
- The shrink phase preserves the MLP threshold, not the exact formula label.
  Therefore a final point can be a surrogate success but a false oracle success.
- Coordinate-wise shrink is greedy and order-dependent. Restoring one coordinate
  may prevent a later coordinate from being restored as much as a joint
  optimization could allow.
- The method can require many model evaluations because every successful sphere
  candidate can trigger up to 10 sweeps over 13 coordinates with binary search.

## Thesis Interpretation

This optimizer asks whether random search mostly suffers from unnecessary
movement in irrelevant coordinates. If coordinate shrink keeps most random-sphere
rescues while reducing normalized distance, it is evidence that a cheap local
post-processing step can improve gradient-free optimization. If rescue count
drops or false successes rise, it shows that the random-sphere candidate often
depends on coordinated multi-parameter changes that cannot be independently
shrunk without leaving the true assemblable region.


## Run-Specific Model Note

This run pairs the optimizer with a tree ensemble surrogate checkpoint from `results/chapter5_optimization/checkpoints/` as a gradient-free control (`tab:tree-gradient-free-control`). Optimizer mechanics are unchanged; probabilities come from the tree model, not from MLP input gradients.
