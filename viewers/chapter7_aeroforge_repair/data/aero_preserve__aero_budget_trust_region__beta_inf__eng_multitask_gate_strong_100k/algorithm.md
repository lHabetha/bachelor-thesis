# Aero-budgeted trust region + shrinkage

Chapter 7 performance-preserving repair (`sec:aeroforge-performance-preserving`).

Trust-region overlap candidates are filtered/ranked by predicted
$R_{\mathrm{aero}}$, optionally under a budget $R_{\mathrm{aero}}^2\le\beta$.
$\beta=\infty$ disables the hard budget but still prefers low predicted drift.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).
