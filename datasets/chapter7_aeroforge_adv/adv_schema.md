# ADV schema — AeroForge 43-key build parameters

Thesis anchor: Appendix AeroForge ADV tables (`tab:appendix-aeroforge-adv-kept`,
`tab:appendix-aeroforge-adv-removed`); `sec:aeroforge-sampling`.

Machine-readable key list: `adv_schema.json`.

Each sample in `advs/{sample_idx:06d}.json` contains:

| Field | Type | Description |
|-------|------|-------------|
| `sample_idx` | int | 0-based index (0 … 99,999) |
| `seed` | int | `seed_base + sample_idx` (see `manifest.json`) |
| `family` | str | Weighted generator family name |
| `adv` | object | Exactly 43 keys listed below |

## Kept keys (43)

Airframe-only subset used for overlap labeling and repair. Numeric drivers are
sampled per family; categoricals draw from fixed domains.

**Fuselage / layout:** `length`, `max_width`, `max_height`, `wingspan`,
`wing_position`, `wing_height_ratio`, `tail_position`, `design_name`,
`wall_thickness`, `end_cap_percent`

**Main wing:** `aspect_ratio`, `taper`, `sweep`, `dihedral`, `twist`,
`root_incidence`, `airfoil_source`, `root_naca_code`, `tip_naca_code`,
`root_csv_filepath`, `tip_csv_filepath`

**Tail type:** `tail_type`, `v_tail_angle`

**Horizontal stabilizer:** `hstab_semispan`, `hstab_aspect_ratio`, `hstab_taper`,
`hstab_sweep`, `hstab_dihedral`, `hstab_root_incidence`, `hstab_airfoil_source`,
`hstab_root_naca_code`, `hstab_tip_naca_code`, `hstab_root_csv_filepath`,
`hstab_tip_csv_filepath`

**Vertical stabilizer:** `vstab_height`, `vstab_aspect_ratio`, `vstab_taper`,
`vstab_sweep`, `vstab_airfoil_source`, `vstab_root_naca_code`,
`vstab_tip_naca_code`, `vstab_root_csv_filepath`, `vstab_tip_csv_filepath`

## Model input notes

- Six `*_csv_filepath` keys are present in ADV JSON but **dropped** from surrogate
  inputs (always null in the 100k pool).
- Categoricals are one-hot encoded → **39 raw numeric inputs**; engineered spacing
  features → **45 inputs** for engineered models.
- Overlap label: CadQuery fast cut metric; binary overlap if volume > **1 mm³**.

Defined in `code/chapter7_aeroforge/overlap_search/core.py` (`ADV_KEYS`).
