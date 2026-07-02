# Trust-region overlap hybrid

Chapter 7 overlap repair (`sec:aeroforge-optimization`).

Expanding normalized trust regions around the start. Each shell evaluates
coordinate directions, overlap-gradient-biased directions where available, and a
deterministic random batch— the same hybrid idea as clevis trust-region search
(`sec:trust-region`). MLP gradients supply coarse directionality; random probes
add robustness.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column B: engineered log-Huber overlap model, detached feature gradients.
