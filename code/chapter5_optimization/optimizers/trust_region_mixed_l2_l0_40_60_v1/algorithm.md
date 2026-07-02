# Trust Region Mixed L2-L0

## Intuition

This optimizer keeps the trust-region candidate generator unchanged. It still
evaluates the start-gradient direction, all signed coordinate directions, and
random directions on expanding radii. The sparsity change is only in the
successful-candidate selection score:

\[
0.4\,L_2 + 0.6\,\frac{L_0}{13}.
\]

\(L_2\) is the normalized Euclidean distance from the blocked start. \(L_0\)
counts active coordinates and is divided by the 13 editable parameters so that
the two terms are on a comparable scale.

## Configuration

| Parameter | Value |
|-----------|-------|
| Candidate generator | unchanged trust-region hybrid |
| Proximity score | `0.4 * L2 + 0.6 * (L0 / 13)` |
| Random directions per radius | `32` |
| Success criterion | valid candidate with `P(assemblable) >= tau` |

## Failure Modes

- The mixed score can prefer a sparser but farther candidate.
- The result depends on which candidates the original trust-region search records.
- The 40/60 weighting is a representative proximity preference, not a tuned
  optimum.
