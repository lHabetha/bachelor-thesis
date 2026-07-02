# Trust-region overlap hybrid + shrinkage

Chapter 7 overlap repair (`sec:aeroforge-optimization`).

Trust-region hybrid search followed by rolling changed coordinates back toward the
start while the overlap surrogate stays clean. Targets minimum-edit repairs among
the trust-region family (`tab:aeroforge-overlap-repair`).

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column C: multitask engineered overlap model, full feature gradients.
