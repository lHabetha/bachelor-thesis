# One-shot overlap gradient line search

Chapter 7 overlap repair (`sec:aeroforge-optimization`, `tab:aeroforge-overlap-repair-grid`).

Computes one overlap-surrogate gradient at the overlapping start, then searches
along the strongest driver directions for the smallest move whose predicted
overlap volume falls below `tau_decide` (headline rows use 1 mm³). This is the
AeroForge analogue of the clevis one-shot gradient (`sec:one-shot-gradient`):
useful coarse directionality from the MLP, but a single local step cannot recover
from all failure modes.

## Scope and limitations

During search, candidates are scored only by the overlap surrogate; CadQuery
boolean verification is reserved for the selected final designs
(`sec:aeroforge-optimization`). Quick-simulation (VLM) metrics shown in the
viewer are post-repair checks, not in-loop objectives unless noted below. This
study targets external wing--tail overlap only, not full AeroForge validity
(`sec:aeroforge-conclusion`).

## Run-specific surrogate

Surrogate column E: raw log-Huber overlap model on ADV inputs only.
