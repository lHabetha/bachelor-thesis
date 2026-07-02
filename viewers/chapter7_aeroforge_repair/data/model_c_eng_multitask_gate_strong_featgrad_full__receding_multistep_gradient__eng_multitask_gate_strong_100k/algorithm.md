# Receding overlap gradient

Chapter 7 overlap repair (`sec:aeroforge-optimization`).

Recomputes the overlap-surrogate gradient after each accepted clean move, similar
to the clevis receding-gradient family. Trades more surrogate evaluations for
higher repair coverage than one-shot search; edits tend to be denser
(`tab:aeroforge-overlap-repair`).

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column C: multitask engineered overlap model, full feature gradients.
