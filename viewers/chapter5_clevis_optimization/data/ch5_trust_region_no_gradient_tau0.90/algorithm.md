# Trust Region Without Gradient

## Intuition

This optimizer is an ablation of `trust_region_hybrid_v1`. It searches the same
expanding normalized-radius neighborhoods around the blocked start, but removes
the MLP-gradient direction from the candidate set. The remaining candidates are
all signed coordinate axes plus a deterministic random batch at each radius.

## Configuration

| Parameter | Value |
|-----------|-------|
| Radii | `0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5` |
| Deterministic directions | 26 signed coordinate axes |
| Random directions | 32 per radius |
| Max evaluations | 523 (`1 + 9 * 58`) |
| Early stop | after a full radius if any valid crossing has been found |
| Selection rule | nearest valid surrogate crossing |
| Ground-truth use | reporting only |

## Inputs And Outputs

Standard clevis repair benchmark inputs and public viewer schema v1 outputs.

## Mathematical Description

For each configured radius \(r\), evaluate candidates on that normalized-radius
shell:

$$
x_i = x_{0,i} + r u_i \sigma_i
$$

where \(u\) is either a coordinate axis or a random unit direction. Because radii
are processed from small to large, the search is an expanding set of shells
around the start design. The selected design is the minimum-distance valid
surrogate crossing among all evaluated candidates; if no crossing exists, the
fallback is the valid recorded candidate with highest surrogate probability.

## Failure Modes

- Removing the gradient direction may lose a useful locally directed probe.
- The random directions are deterministic, but still only sample a finite subset
  of the sphere.
- Narrow successful regions can be missed if no coordinate or random direction
  lands in them at the tested radii.

## Thesis Interpretation

This variant measures how much the trust-region result depends on the single
start-gradient direction versus the richness of the coordinate and random
neighborhood itself.
