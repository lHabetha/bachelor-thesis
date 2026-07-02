# One-Shot Gradient Bracket Line Search

## Intuition

This is the simplest possible gradient-based optimizer: compute **one gradient** at the
starting blocked design, then search along that single direction for the smallest step
that makes the surrogate predict "assemblable" (P >= tau).

The search uses a coarse-to-fine bracket strategy: first test 16 logarithmically-spaced
step sizes spanning 8 orders of magnitude to find the approximate crossing point, then
refine with 20 equally-spaced steps between the bracket bounds.

**Tau:** Configurable (default 0.60). The optimizer declares success when the surrogate
probability crosses this threshold. The exact formula oracle is evaluated afterward for
reporting only — it never influences the optimizer's step selection.

## Configuration

| Parameter | Value |
|-----------|-------|
| Gradient evaluations | 1 (at start only) |
| Coarse step magnitudes | 16 (8 orders × 2 multipliers: 1.0 and 0.5) |
| Orders of magnitude | 10^3, 10^2, 10^1, 10^0, 10^-1, 10^-2, 10^-3, 10^-4 |
| Fine bracket steps | 20 (equally spaced between bracket bounds) |
| Total max evaluations | 36 (16 coarse + 20 fine) |
| Success criterion | Surrogate P(assemblable) >= tau |
| Validity enforcement | Every selected success must pass `is_valid()` |
| Fallback (no crossing) | Select the largest valid step tested |
| Distance metric | L2 in median-centered unit-variance normalized space |

## Inputs And Outputs

- **Input:** 200 blocked starts from `benchmark_sets/blocked_200_v1/`
- **Model:** Persisted MLP checkpoint from `results/chapter5_optimization/checkpoints/row1_uncertainty_disagreement_B1000_T2500_best/`
- **Output:** Per-start trajectory JSON (schema v1) + 50-frame viewer export
- **Statistics:** JSON/Markdown summary with subgroup breakdown

## Mathematical Description

Given starting parameters **x₀** (blocked, valid), compute:

$$
g_{\mathrm{raw}} = \nabla_x P(\mathrm{assemblable} \mid x_0)
$$

This is the 13-dimensional gradient obtained by backpropagating through the MLP.

Transform gradient to the benchmark's normalized space (median=0, std=1 over the active benchmark starts):

$$
g_{\mathrm{norm}, i} = g_{\mathrm{raw}, i} \cdot \sigma_i
$$

using the chain rule for

$$
x_i = \tilde{x}_i \cdot \sigma_i + \mu_i
$$

Then normalize the direction in normalized space:

$$
\tilde{d} = \frac{g_{\mathrm{norm}}}{\lVert g_{\mathrm{norm}} \rVert_2}
$$

Convert back to raw parameter space for stepping:

$$
d_{\mathrm{raw}, i} = \tilde{d}_i \cdot \sigma_i
$$

**Coarse search:** For magnitudes m ∈ {1000, 500, 100, 50, 10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005}:

$$
x_{\mathrm{candidate}} = x_0 + m \cdot d_{\mathrm{raw}}
$$

Note: because $\tilde{d}$ is a unit vector in normalized space,
`normalized_distance(x0, x_candidate) = m`.
A step of magnitude 1 moves exactly 1 standard deviation in the gradient direction.

Evaluate `is_valid(x_candidate)` and, if valid, `P(assemblable | x_candidate)`.
Invalid candidates are recorded with the start probability and cannot be selected
as successes.

**Bracket:** Find adjacent magnitudes (m_lo, m_hi) where:
- $P(x_0 + m_{\mathrm{hi}} \cdot d_{\mathrm{raw}}) \ge \tau$ and the candidate is valid
- $P(x_0 + m_{\mathrm{lo}} \cdot d_{\mathrm{raw}}) < \tau$
- $(m_{\mathrm{lo}}, m_{\mathrm{hi}})$ is the **tightest** such bracket (smallest magnitudes)

**Fine search:** Evaluate 20 equally-spaced magnitudes in (m_lo, m_hi).

**Selection:** The smallest magnitude $m^{\star}$ where $P \ge \tau$ and validity passes.
If no valid candidate crosses tau, select the largest valid tested step as the
canonical one-shot fallback. If no valid tested step exists, record `no_valid_step`
and keep the valid start as the selected final design.

## Failure Modes

- **No crossing:** None of the 16 coarse steps reach tau. The optimizer selects
  the largest valid step as a fallback. If there is no valid tested step, it keeps the start with `no_valid_step`. Likely for starts deep in the blocked region.
- **Validity failures:** Large steps can leave the valid design space. These are
  skipped during search.
- **False surrogate success:** The MLP predicts assemblable but the exact formula
  oracle disagrees. This is an expected failure mode that the statistics report.
- **Zero gradient:** The MLP has zero gradient at the start (e.g. saturated sigmoid).
  Rare; logged as `stop_reason=zero_gradient`.

## Thesis Interpretation

This optimizer answers the question: "If you only get ONE gradient evaluation from
the surrogate, how far along that single direction do you need to go to cross the
predicted assemblability boundary?" It is deliberately minimal — no iterative
refinement, no momentum, no adaptive steps. It establishes a lower bound on what
gradient information alone can achieve, and its failure cases motivate more
sophisticated multi-step optimizers.


## Run-Specific Model Note

This run uses the thesis-facing active-learning-selected MLP (`row1_uncertainty_disagreement`, dense pool B=1000, T=2500). The §5.3 parameter-lock study freezes selected coordinates during repair; optimizer mechanics otherwise match `sec:semantic-locks` / `tab:gradient-locks`.
