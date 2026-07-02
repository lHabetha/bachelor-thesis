# Aero-penalized receding overlap repair

Chapter 7 performance-preserving repair (`sec:aeroforge-performance-preserving`, `tab:aeroforge-aero-preserve`).

Overlap repair uses the fixed multitask overlap surrogate; an additional term
$\lambda_{\mathrm{aero}} R_{\mathrm{aero}}^2$ penalizes predicted movement in
quick-simulation performance space. Higher $\lambda$ preserves aerodynamics more
strongly but can reduce repair coverage. Median **actual** $R_{\mathrm{aero}}$
after repair (quick sim) is reported in the thesis table, not the in-loop
predicted median shown in run statistics.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).
