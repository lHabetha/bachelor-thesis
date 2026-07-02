# Best verified overlap repair (meta)

Chapter 7 overlap repair grid (`tab:aeroforge-overlap-repair-grid`).

For each start, selects the verified CAD-clean candidate with lowest normalized
distance among the optimizer variants in this model column. Reports aggregate
grid statistics; it is not a separate search algorithm.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column B: engineered log-Huber overlap model, detached feature gradients.
