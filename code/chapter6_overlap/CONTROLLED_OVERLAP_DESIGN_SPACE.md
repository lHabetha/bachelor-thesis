# Controlled-Overlap Design Space (Chapter 6)

**Status:** Frozen.
**Purpose:** Define which dummy-clevis validity constraints are kept hard and
which are relaxed for the overlap-aware dataset, active-learning surrogate, and
overlap optimizer (thesis Chapter 6).

## High-Level Policy

The overlap extension intentionally relaxes selected inter-part validity constraints so the
dummy clevis can contain controlled geometric overlap. The goal is to create a
meaningful supervised overlap target and then optimize overlap away.

The relaxed design space is **not** allowed to become arbitrary broken geometry.
The sampler must still reject:

- degenerate solids;
- negative offsets;
- holes breaking out of their own parent part where the constraint remains hard;
- wrong-side / mirrored / pass-through geometry;
- part-swapped geometries;
- cases where the bracket cavity becomes a negative distance.

Important downstream consequence:

The assemblability optimizer in the combined zig-zag pipeline will also operate
under these more relaxed constraints. Therefore, after overlap repair works, the
combined pipeline must guard against the assemblability optimizer reintroducing
overlap. The first implementation priority is still overlap repair alone; handle
the zig-zag issue only after the overlap optimizer is working.

## Constraint Decisions

### Group A - Sliding Fits

- **A1 - Relax entirely.**
  Original:
  ```text
  main_pin_radius + eps_fit <= main_hole_radius
  ```
  New relaxed meaning: the main pin may be larger than the bracket through-hole.
  This intentionally creates possible `bracket-main_pin` overlap.

- **A2 - Relax entirely.**
  Original:
  ```text
  splint_radius + eps_fit <= cross_hole_radius
  ```
  New relaxed meaning: the splint may be larger than the pin cross-hole. This
  intentionally creates possible `main_pin-splint` overlap.

- **A3 - Keep hard.**
  Original:
  ```text
  SPLINT_HEAD_RATIO * splint_radius >= cross_hole_radius + eps_head_slip
  ```
  Meaning: the splint head must still retain the splint. This is a functional
  retention rule, not an overlap-generation rule.

### Group B - Holes Stay Inside Their Parent Solid

- **B1 - Keep hard.**
  Original:
  ```text
  2 * main_hole_radius + 2 * eps_depth_wall <= depth
  ```
  Meaning: the main hole must not break out of the bracket in X.

- **B2 - Modify.**
  Original:
  ```text
  main_hole_offset_from_open_end + main_hole_radius <= 0.80 * leg_length
  ```
  New relaxed rule:
  ```text
  main_hole_offset_from_open_end + main_hole_radius <= 0.99 * leg_length
  ```
  Meaning: the main hole may move much higher in the leg, almost to the top of
  the leg. This makes it possible for the main pin head / pin geometry to
  overlap into the roof region through D7-related geometry.

- **B3 - Keep hard.**
  Original:
  ```text
  main_hole_offset_from_open_end - main_hole_radius >= eps_leg_wall
  ```
  Meaning: the main hole must not break the open leg tip.

- **B4 - Keep hard.**
  Original:
  ```text
  cross_hole_distance_from_free_end >= cross_hole_radius + eps_pin_wall
  ```
  Meaning: the cross-hole must not break the main pin's `-Y` free tip.

- **B5 - Keep hard.**
  Original:
  ```text
  cross_hole_distance_from_free_end + cross_hole_radius + eps_pin_wall
      <= main_pin_length - head_len
  ```
  With fixed head length:
  ```text
  cross_hole_distance_from_free_end + cross_hole_radius + eps_pin_wall
      <= 0.95 * main_pin_length
  ```
  Meaning: the cross-hole must not eat into the `+Y` nail-head.

### Group C - Bracket Cavity

- **C1 - Relax, but keep positive cavity.**
  Original:
  ```text
  outer_span > 2 * wall_thickness + eps_access
  ```
  New relaxed rule:
  ```text
  outer_span > wall_thickness + eps_access
  ```
  Meaning: the two bracket legs may move much closer than before, but the inner
  cavity must not become a negative distance. The frame should remain a
  meaningful bracket-like object, not an inverted or self-swapped geometry.

### Group D - Parts Can Physically Be In The Assembled Pose

- **D1 - Keep hard.**
  Original:
  ```text
  main_pin_length > outer_span + 2 * cross_hole_distance_from_free_end
                    + 2 * eps_access
  ```
  Meaning: the pin must still be long enough that the cross-hole center sits
  outside the `-Y` wall.

- **D2 - Keep hard.**
  Original:
  ```text
  main_pin_length >= outer_span - 2 * wall_thickness
                     + 2 * cross_hole_distance_from_free_end
                     + 2 * cross_hole_radius
                     + 2 * eps_access
  ```
  Meaning: the cross-hole rim must still clear the `-Y` wall's inner face.

- **D3 - Keep hard.**
  Original:
  ```text
  splint_length >= 2 * main_pin_radius + 2 * eps_retention
  ```
  Meaning: the splint must still be long enough to span the pin and retain it.

- **D4 - Relax entirely.**
  Original:
  ```text
  hole_z + splint_length/2 + splint_head_len
      + main_hole_radial_slack + eps_head_clear
      <= wall_thickness/2
  ```
  Fixed-head equivalent:
  ```text
  0.55 * splint_length + main_hole_radial_slack + eps_head_clear
      <= wall_thickness + leg_length - main_hole_offset_from_open_end
  ```
  New relaxed meaning: the splint head may clip into the roof. This
  intentionally creates possible `bracket-splint` overlap.

- **D5 - Relax entirely.**
  Original:
  ```text
  cross_y + hr <= -outer_span/2 - eps_access
  ```
  Equivalent:
  ```text
  main_pin_length >= outer_span + 2 * cross_hole_distance_from_free_end
                     + 2 * hr + 2 * eps_access
  where hr = SPLINT_HEAD_RATIO * splint_radius
  ```
  New relaxed meaning: the splint head may extend into the `-Y` bracket leg
  wall. This intentionally creates possible `bracket-splint` overlap.

- **D6 - Relax entirely.**
  Original:
  ```text
  SPLINT_HEAD_RATIO * splint_radius + eps_fit <= main_hole_radius
  ```
  New relaxed meaning: the splint head disc may be too large for the bracket
  main-hole bore and may overlap bracket material. This is related to D5 but
  should still be tracked explicitly.

- **D7 - Relax entirely.**
  Original:
  ```text
  hole_z + 2 * main_pin_radius + main_hole_radial_slack + eps_head_clear
      <= wall_thickness/2
  ```
  Equivalent:
  ```text
  main_hole_offset_from_open_end + main_hole_radius + main_pin_radius
      + eps_head_clear <= wall_thickness + leg_length
  ```
  New relaxed meaning: the main pin nail-head may intersect the roof underside.
  This intentionally creates possible `bracket-main_pin` overlap.

### Group E - Non-Degenerate Dimensions

- **E1 - Keep hard.**
  Original:
  ```text
  all linear dimensions >= 0.5 mm
  ```
  Meaning: no degenerate solids.

- **E2 - Keep hard.**
  Original:
  ```text
  overhang_span_y >= 0
  main_hole_offset_from_open_end >= 0
  cross_hole_distance_from_free_end >= 0
  ```
  Meaning: no negative offsets.

- **E3 - Keep hard.**
  Original:
  ```text
  exploded_gap > 0
  ```
  Meaning: preview/export separation remains positive.

## Summary Table

| Constraint | Relaxation decision | Notes |
|---|---|---|
| A1 | Relax entirely | Allow pin shaft vs. bracket hole overlap. |
| A2 | Relax entirely | Allow splint shaft vs. cross-hole overlap. |
| A3 | Keep hard | Retention must remain meaningful. |
| B1 | Keep hard | Main hole must stay inside bracket depth. |
| B2 | Modify | Use `0.99 * leg_length` instead of `0.80 * leg_length`. |
| B3 | Keep hard | Main hole must not break open leg tip. |
| B4 | Keep hard | Cross-hole must not break pin free tip. |
| B5 | Keep hard | Cross-hole must not eat into nail-head. |
| C1 | Modify / relax | Use `outer_span > wall_thickness + eps_access`. |
| D1 | Keep hard | Cross-hole center still outside `-Y` wall. |
| D2 | Keep hard | Cross-hole rim still clears `-Y` wall inner face. |
| D3 | Keep hard | Splint must span pin. |
| D4 | Relax entirely | Allow splint head vs. roof overlap. |
| D5 | Relax entirely | Allow splint head vs. `-Y` wall overlap. |
| D6 | Relax entirely | Allow splint head vs. bracket bore/material overlap. |
| D7 | Relax entirely | Allow pin nail-head vs. roof overlap. |
| E1 | Keep hard | No degenerate dimensions. |
| E2 | Keep hard | No negative offsets. |
| E3 | Keep hard | Preview gap stays positive. |

## Implementation Notes

- The relaxed sampler should record every original constraint margin and every
  relaxed/modified margin as diagnostics.
- The labeler should not assume that a failed original constraint necessarily
  means visible mesh overlap. SDF labels decide the actual continuous overlap
  value.
- The overlap optimizer should first focus only on reducing overlap under this
  relaxed design space.
- The combined zig-zag pipeline must later account for the fact that the
  assemblability optimizer can exploit the relaxed constraints and reintroduce
  overlap unless guarded by overlap checks or a combined objective.
