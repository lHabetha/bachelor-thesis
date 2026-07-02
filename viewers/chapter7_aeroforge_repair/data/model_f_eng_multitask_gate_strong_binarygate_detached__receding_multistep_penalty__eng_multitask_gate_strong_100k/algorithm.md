# Receding overlap gradient with proximity penalty

Chapter 7 overlap repair (`sec:aeroforge-optimization`).

Same receding-gradient structure as above, but candidate ranking adds a
normalized $L_2$ proximity penalty so accepted moves stay closer to the
original generated aircraft while the overlap surrogate remains below
`tau_decide`.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column F: multitask engineered model with binary gate, detached feature gradients.
