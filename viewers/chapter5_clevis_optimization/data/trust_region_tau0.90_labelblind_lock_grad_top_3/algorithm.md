# Trust Region Hybrid

## Intuition

This optimizer searches expanding normalized-radius trust regions around the
blocked start. It computes the start gradient once, then at each radius evaluates
that direction when available, all coordinate directions, and a deterministic
random batch. It is a robust middle ground between interpretable
gradient/coordinate moves and blind random sphere search.

## Configuration

| Parameter | Value |
|-----------|-------|
| Radii | `0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5` |
| Deterministic directions | start gradient if finite/non-zero, plus 26 signed axes |
| Random directions | 32 per radius |
| Max evaluations | 532 with gradient (`1 + 9 * 59`), otherwise 523 |
| Early stop | after a full radius if any valid crossing has been found |
| Selection rule | nearest valid surrogate crossing |
| Oracle use | reporting only |

## Inputs And Outputs

Standard clevis repair benchmark inputs and public viewer schema v1 outputs.

## Mathematical Description

For each configured radius \(r\), evaluate candidates on that normalized-radius
shell:

$$
x_i = x_{0,i} + r u_i \\sigma_i
$$

where \(u\) is either the local gradient direction, a coordinate axis, or a
random unit direction. Because radii are processed from small to large, the full
search is an expanding set of shells around the start design. The selected design
is the minimum-distance valid surrogate crossing among all evaluated candidates;
if no crossing exists, the fallback is the valid recorded candidate with highest
surrogate probability, using larger normalized distance only as a tie-breaker. If no valid non-start candidate exists, the optimizer records `no_valid_step` and keeps the valid start as the selected final design.
Only valid candidates receive MLP probabilities at their candidate coordinates;
invalid candidates are recorded with the start probability and cannot be selected
as successes.

## Failure Modes

- It can be evaluation-heavy relative to one-shot.
- Random probes add variance, although seeds make the run reproducible.
- If all useful directions require multi-step curved paths, a pure start-centered
  trust region can still miss them.

## Thesis Interpretation

Trust-region hybrid search asks whether a robust local neighborhood search beats
single-direction gradients without abandoning the minimum-edit objective.



## Run-Specific Model Note

This run uses the thesis-facing active-learning-selected MLP (`row1_uncertainty_disagreement`, dense pool B=1000, T=2500). The §5.3 parameter-lock study freezes selected coordinates during repair; optimizer mechanics otherwise match `sec:semantic-locks` / `tab:gradient-locks`.
