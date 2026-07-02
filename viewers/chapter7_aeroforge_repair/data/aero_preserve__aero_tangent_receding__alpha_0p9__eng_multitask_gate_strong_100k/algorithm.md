# Aero-tangent projected receding repair

Chapter 7 performance-preserving repair (`sec:aeroforge-performance-preserving`).

Projects the overlap-reduction gradient away from the performance-surrogate
Jacobian row space. $\alpha=0$ recovers ordinary overlap gradients;
$\alpha\to 1$ moves approximately along constant-performance directions. Does
not guarantee exact invariance of quick-sim outputs after repair.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).
