# Parameter Lock Robustness Study

Study ID: `parameter_locks_ch5_reduced_v1`

This suite reruns Chapter 5 optimizers while selected parameters are forced to remain
equal to their blocked-start values. Each constrained run is paired with its
unconstrained baseline.

## Run Summary

| Run | Constraint | Oracle OK | False OK | Mean Dist | Drop vs Baseline | Recoverability |
|-----|------------|-----------|----------|-----------|------------------|----------------|
| `one_shot_gradient_bracket_v1_tau0.60_labelblind` | `none` | 146 | 7 | 0.3010 | n/a | n/a |
| `penalized_proximity_tau0.75_labelblind` | `none` | 158 | 0 | 0.2931 | n/a | n/a |
| `sphere_shrink_tau0.90_labelblind` | `none` | 179 | 0 | 0.5607 | n/a | n/a |
| `trust_region_tau0.90_labelblind` | `none` | 200 | 0 | 0.6255 | n/a | n/a |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_grad_bottom_10` | `grad_bottom_10` | 176 | 9 | 0.3787 | -30 | 100.0% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_grad_bottom_11` | `grad_bottom_11` | 190 | 9 | 0.5189 | -44 | 99.3% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_grad_bottom_12` | `grad_bottom_12` | 194 | 6 | 0.4255 | -48 | 100.0% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_grad_top_3` | `grad_top_3` | 100 | 7 | 0.4007 | 46 | 64.4% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_lock_main_bracket_user` | `lock_main_bracket_user` | 145 | 6 | 0.3934 | 1 | 83.6% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_lock_pin_user` | `lock_pin_user` | 135 | 8 | 0.3050 | 11 | 91.8% |
| `one_shot_gradient_bracket_v1_tau0.60_labelblind_lock_lock_splint_user` | `lock_splint_user` | 150 | 8 | 0.3085 | -4 | 99.3% |
| `penalized_proximity_tau0.75_labelblind_lock_grad_bottom_10` | `grad_bottom_10` | 177 | 0 | 0.3303 | -19 | 96.2% |
| `penalized_proximity_tau0.75_labelblind_lock_grad_bottom_11` | `grad_bottom_11` | 179 | 0 | 0.3850 | -21 | 94.9% |
| `penalized_proximity_tau0.75_labelblind_lock_grad_bottom_12` | `grad_bottom_12` | 159 | 0 | 0.3090 | -1 | 84.8% |
| `penalized_proximity_tau0.75_labelblind_lock_grad_top_3` | `grad_top_3` | 105 | 0 | 0.3386 | 53 | 64.6% |
| `penalized_proximity_tau0.75_labelblind_lock_lock_main_bracket_user` | `lock_main_bracket_user` | 137 | 0 | 0.3105 | 21 | 81.0% |
| `penalized_proximity_tau0.75_labelblind_lock_lock_pin_user` | `lock_pin_user` | 147 | 0 | 0.3067 | 11 | 92.4% |
| `penalized_proximity_tau0.75_labelblind_lock_lock_splint_user` | `lock_splint_user` | 168 | 0 | 0.3010 | -10 | 100.0% |
| `sphere_shrink_tau0.90_labelblind_lock_grad_bottom_10` | `grad_bottom_10` | 200 | 0 | 0.4388 | -21 | 100.0% |
| `sphere_shrink_tau0.90_labelblind_lock_grad_bottom_11` | `grad_bottom_11` | 200 | 0 | 0.4505 | -21 | 100.0% |
| `sphere_shrink_tau0.90_labelblind_lock_grad_bottom_12` | `grad_bottom_12` | 200 | 0 | 0.4762 | -21 | 100.0% |
| `sphere_shrink_tau0.90_labelblind_lock_grad_top_3` | `grad_top_3` | 128 | 0 | 0.9128 | 51 | 70.4% |
| `sphere_shrink_tau0.90_labelblind_lock_lock_main_bracket_user` | `lock_main_bracket_user` | 161 | 0 | 0.6078 | 18 | 86.0% |
| `sphere_shrink_tau0.90_labelblind_lock_lock_pin_user` | `lock_pin_user` | 174 | 0 | 0.6061 | 5 | 92.2% |
| `sphere_shrink_tau0.90_labelblind_lock_lock_splint_user` | `lock_splint_user` | 198 | 0 | 0.5011 | -19 | 98.9% |
| `trust_region_tau0.90_labelblind_lock_grad_bottom_10` | `grad_bottom_10` | 200 | 0 | 0.5935 | 0 | 100.0% |
| `trust_region_tau0.90_labelblind_lock_grad_bottom_11` | `grad_bottom_11` | 200 | 0 | 0.6265 | 0 | 100.0% |
| `trust_region_tau0.90_labelblind_lock_grad_bottom_12` | `grad_bottom_12` | 200 | 0 | 0.6843 | 0 | 100.0% |
| `trust_region_tau0.90_labelblind_lock_grad_top_3` | `grad_top_3` | 195 | 0 | 1.4810 | 5 | 97.5% |
| `trust_region_tau0.90_labelblind_lock_lock_main_bracket_user` | `lock_main_bracket_user` | 200 | 0 | 0.7642 | 0 | 100.0% |
| `trust_region_tau0.90_labelblind_lock_lock_pin_user` | `lock_pin_user` | 200 | 0 | 0.6810 | 0 | 100.0% |
| `trust_region_tau0.90_labelblind_lock_lock_splint_user` | `lock_splint_user` | 200 | 0 | 0.6260 | 0 | 100.0% |

## Parameter Criticality

| Parameter | Locked Starts | Mean Success Drop | Mean Distance Delta | False OK |
|-----------|---------------|-------------------|---------------------|----------|
| `main_pin_length` | 2080 | 0.0668 | 0.1330 | 19 |
| `leg_length` | 2220 | 0.0586 | 0.1367 | 23 |
| `overhang_span_y` | 3032 | 0.0053 | 0.1010 | 23 |
| `main_hole_offset_from_open_end` | 3068 | -0.0476 | 0.0663 | 28 |
| `outer_span` | 3200 | -0.0537 | 0.0457 | 29 |
| `wall_thickness` | 3200 | -0.0581 | 0.0358 | 30 |
| `depth` | 3200 | -0.0581 | 0.0358 | 30 |
| `main_hole_radius` | 3200 | -0.0622 | 0.0247 | 32 |
| `main_pin_radius` | 3200 | -0.0622 | 0.0247 | 32 |
| `splint_length` | 3200 | -0.0788 | 0.0247 | 32 |
| `cross_hole_radius` | 3200 | -0.0809 | 0.0146 | 32 |
| `splint_radius` | 3200 | -0.0809 | 0.0146 | 32 |
| `cross_hole_distance_from_free_end` | 2400 | -0.0942 | 0.0230 | 24 |
