# All aero-aware best drift (meta)

Chapter 7 performance-preserving repair (`tab:aeroforge-aero-preserve`).

Per start, picks the verified CAD-clean candidate with lowest **predicted**
$R_{\mathrm{aero}}$ among all performance-aware optimizer runs in the sweep.
Thesis median **actual** $R_{\mathrm{aero}}$ comes from post-repair quick sim
(`T3_drift_actual.csv`).

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).
