# Coordinate Axis Bracket Search

## Intuition

Some clevis failures may be dominated by one physical parameter: roof span,
splint length, pin length, or a clearance radius. This optimizer tests that
hypothesis by searching each normalized coordinate axis independently in both
directions.

## Configuration

| Parameter | Value |
|-----------|-------|
| Directions | 26 signed coordinate axes |
| Magnitudes | `10^2` through `10^-4`, multipliers `1.0`, `0.5`, `0.2` |
| Evaluations | 547 exactly (`1 + 26 * 21`) |
| Early stop | none; the full axis grid is always evaluated |
| Success criterion | `P(assemblable) >= tau` |
| Selection rule | nearest valid surrogate crossing |
| Oracle use | reporting only |

## Inputs And Outputs

Inputs and outputs use the clevis repair benchmark and public viewer schema v1 run artifacts.

## Mathematical Description

For each coordinate \(i\) and sign \(s \\in \\{-1,+1\\}\), evaluate:

$$
x(m)_i = x_{0,i} + s m \\sigma_i
$$

with all other parameters unchanged. The selected candidate is the smallest
normalized-distance valid axis move that crosses tau. If no axis move crosses
tau, the fallback is the valid recorded candidate with highest surrogate
probability, using larger normalized distance only as a tie-breaker. If no valid non-start candidate exists, the optimizer records `no_valid_step` and keeps the valid start as the selected final design.
Only valid candidates receive MLP probabilities at their candidate coordinates;
invalid candidates are recorded with the start probability and cannot be selected
as successes.

## Failure Modes

- True repairs may require coupled parameter changes and cannot be found by
  one-axis moves.
- Large axis steps frequently violate geometric validity constraints.
- Axis moves are easy to interpret but can be conservative compared with diagonal
  gradient moves.

## Thesis Interpretation

Coordinate search is a diagnostic optimizer: it reveals whether the learned
decision boundary is mostly controlled by a small number of physically meaningful
parameters or by coupled combinations.



## Run-Specific Model Note

This run pairs the optimizer with a tree ensemble surrogate checkpoint from `results/chapter5_optimization/checkpoints/` as a gradient-free control (`tab:tree-gradient-free-control`). Optimizer mechanics are unchanged; probabilities come from the tree model, not from MLP input gradients.
