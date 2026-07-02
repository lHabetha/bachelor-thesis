# Coordinate-grid overlap refinement

Chapter 7 overlap repair (`sec:aeroforge-optimization`).

Gradient-free coordinate search: signed moves on editable ADV drivers, then local
refinement of the best surrogate-clean candidates. Often yields very sparse
one-coordinate repairs but misses more starts and has more false-OK (MLP clean,
CAD overlap remains) cases than gradient-heavy methods
(`tab:aeroforge-overlap-repair`).

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column D: engineered log-Huber overlap model, full feature gradients.
