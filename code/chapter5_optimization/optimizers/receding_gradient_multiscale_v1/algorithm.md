# Multi-Fixed-Step Gradient Beam Search

## Intuition

The one-shot optimizers ask whether the initial gradient direction is enough.
This optimizer asks the next question: if the surrogate landscape curves, does
recomputing the gradient after each valid move find closer or more reliable
rescues while still using a fixed menu of normalized step sizes?

This is called **multi-fixed-step gradient** in Chapter 5. It is not the planned
adaptive multi-step optimizer: step lengths are chosen from a predefined grid,
and the beam is pruned only after candidate expansion.

## Configuration

| Parameter | Value |
|-----------|-------|
| Step sizes | `0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0` |
| Max depth | 20 expansion depths |
| Beam width | 12 candidates kept after each expansion depth |
| Max evaluations | no fixed cap; bounded by depth, beam width, and bracketing |
| Success criterion | `P(assemblable) >= tau` |
| Selection rule | nearest valid crossing among probes evaluated before early stop |
| Early stop | after a search depth once any valid crossing has been found |
| Oracle use | post-hoc reporting only |

## Inputs And Outputs

The optimizer consumes the canonical Chapter 5 blocked starts and an MLP artifact.
It writes standard schema-v1 Chapter 5 run artifacts. `trajectories.json` contains
the raw fixed-step beam-search probes; the browser viewer shows the
start-to-selected final morph.

## Mathematical Description

At a frontier point \(x_t\), compute the normalized-space gradient direction:

$$
d_t = \\frac{\\nabla_{\\tilde{x}}P(x_t)}{\\|\\nabla_{\\tilde{x}}P(x_t)\\|_2}
$$

For every configured step size \(s\), propose:

$$
x_{t+1,i} = x_{t,i} + s d_{t,i}\\sigma_i
$$

Valid candidates are ranked by probability progress per normalized distance, and
only the best 12 become the next depth's frontier. After each depth, if any
evaluated probe has crossed tau, the
optimizer stops and selects the smallest-normalized-distance valid crossing among
the probes evaluated so far.
Only valid candidates receive MLP probabilities at their candidate coordinates;
invalid candidates are recorded with the start probability and cannot be selected
as successes or added to the next beam frontier.
If no recorded valid candidate crosses tau, the fallback is the valid recorded
candidate with highest surrogate probability, using larger normalized distance
only as a tie-breaker.

## Failure Modes

- The beam may discard a low-probability intermediate that would later become a
  close rescue.
- Gradient recomputation can still follow a locally wrong surrogate direction.
- The method is more expensive than one-shot and should be compared by both
  oracle-confirmed successes and evaluations.
- Validity boundaries can eliminate entire branches.

## Thesis Interpretation

This is the fixed-step gradient comparator. It tests whether recomputing local
MLP gradients beyond the blocked start improves rescue behavior when the
optimizer still uses a fixed step-size menu.
