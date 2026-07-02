# Chapter 5 Optimizer Comparison

Current Chapter 5 comparison rows. Fixed-step gradient and fixed-step penalty rows use the depth-20/no-fixed-cap rerun. Random-sphere rows include the matched-seed 32/64/128/256 direction-count sweep.

| Run | Optimizer | Tau | Constraint | Oracle OK | Surr. OK | False OK | No Crossing | Invalid | Mean Dist | Median Dist | Mean Active | Mean L1 |
|-----|-----------|-----|------------|-----------|----------|----------|-------------|---------|-----------|-------------|-------------|---------|
| trust_region_tau0.75_labelblind | trust_region_hybrid_v1 | 0.75 | none | 200 | 200 | 0 | 0 | 0 | 0.5770 | 0.5000 | 6.0150 | 0.8768 |
| ch5_trust_region_no_gradient_tau0.75 | trust_region_no_gradient_v1 | 0.75 | none | 200 | 200 | 0 | 0 | 0 | 0.5857 | 0.5000 | 2.4250 | 0.6579 |
| coordinate_axis_tau0.75_labelblind | coordinate_axis_bracket_v1 | 0.75 | none | 200 | 200 | 0 | 0 | 0 | 0.5857 | 0.5000 | 1.0000 | 0.5857 |
| trust_region_tau0.90_labelblind | trust_region_hybrid_v1 | 0.9 | none | 200 | 200 | 0 | 0 | 0 | 0.6255 | 0.5000 | 5.9550 | 0.9354 |
| ch5_trust_region_no_gradient_tau0.90 | trust_region_no_gradient_v1 | 0.9 | none | 200 | 200 | 0 | 0 | 0 | 0.6407 | 0.5000 | 2.3800 | 0.7197 |
| coordinate_axis_tau0.90_labelblind | coordinate_axis_bracket_v1 | 0.9 | none | 200 | 200 | 0 | 0 | 0 | 0.6408 | 0.5000 | 1.0000 | 0.6408 |
| ch5_sphere_shrink_256_matched_tau0.90 | random_sphere_coordinate_shrink_256_v1 | 0.9 | none | 195 | 192 | 0 | 8 | 0 | 0.5016 | 0.4683 | 5.5750 | 0.9770 |
| ch5_sphere_expanding_256_matched_tau0.90 | random_sphere_expanding_256_v1 | 0.9 | none | 195 | 192 | 0 | 8 | 0 | 0.8187 | 1.0000 | 12.9750 | 2.3648 |
| ch5_sphere_shrink_128_matched_tau0.90 | random_sphere_coordinate_shrink_128_v1 | 0.9 | none | 185 | 179 | 0 | 21 | 0 | 0.5654 | 0.5000 | 6.1450 | 1.2006 |
| ch5_sphere_expanding_128_matched_tau0.90 | random_sphere_expanding_128_v1 | 0.9 | none | 185 | 179 | 0 | 21 | 0 | 0.8452 | 1.0000 | 12.9950 | 2.4574 |
| ch5_sphere_shrink_64_matched_tau0.90 | random_sphere_coordinate_shrink_64_matched_seed_v1 | 0.9 | none | 173 | 159 | 0 | 41 | 0 | 0.5257 | 0.5000 | 6.6500 | 1.1582 |
| ch5_sphere_expanding_64_matched_tau0.90 | random_sphere_expanding_v1 | 0.9 | none | 173 | 159 | 0 | 41 | 0 | 0.8085 | 1.0000 | 12.9900 | 2.3603 |
| ch5d20_multifixed_penalty_lam0_10_tau0.75 | multifixed_penalty_direction_frontier_lam0_10_v1 | 0.75 | none | 167 | 162 | 0 | 38 | 0 | 0.3381 | 0.3538 | 11.9450 | 0.9090 |
| ch5d20_multifixed_penalty_lam0_50_tau0.75 | multifixed_penalty_direction_frontier_lam0_50_v1 | 0.75 | none | 167 | 162 | 0 | 38 | 0 | 0.3397 | 0.3538 | 11.9350 | 0.9144 |
| ch5d20_multifixed_penalty_lam0_30_tau0.75 | multifixed_penalty_direction_frontier_lam0_30_v1 | 0.75 | none | 167 | 163 | 0 | 37 | 0 | 0.3414 | 0.3538 | 11.9400 | 0.9190 |
| ch5d20_multifixed_penalty_lam0_10_tau0.90 | multifixed_penalty_direction_frontier_lam0_10_v1 | 0.9 | none | 167 | 160 | 0 | 40 | 0 | 0.3650 | 0.3899 | 12.0550 | 0.9756 |
| ch5d20_multifixed_penalty_lam0_20_tau0.75 | multifixed_penalty_direction_frontier_lam0_20_v1 | 0.75 | none | 166 | 163 | 1 | 37 | 0 | 0.3399 | 0.3538 | 11.9450 | 0.9128 |
| task34_adaptive_penalty_lam0_10_tau0.75 | adaptive_multistep_penalty_lam0_10_v1 | 0.75 | none | 165 | 158 | 0 | 42 | 0 | 0.3300 | 0.3473 | 12.1200 | 0.8876 |
| task34_adaptive_penalty_lam0_05_tau0.75 | adaptive_multistep_penalty_lam0_05_v1 | 0.75 | none | 165 | 158 | 0 | 42 | 0 | 0.3314 | 0.3473 | 12.1150 | 0.8915 |
| task34_adaptive_gradient_tau0.75 | adaptive_multistep_gradient_v1 | 0.75 | none | 165 | 158 | 0 | 42 | 0 | 0.3320 | 0.3473 | 12.1200 | 0.8932 |
| task34_adaptive_penalty_lam0_10_tau0.90 | adaptive_multistep_penalty_lam0_10_v1 | 0.9 | none | 165 | 157 | 0 | 43 | 0 | 0.3554 | 0.3750 | 12.2250 | 0.9503 |
| task34_adaptive_gradient_tau0.90 | adaptive_multistep_gradient_v1 | 0.9 | none | 165 | 157 | 0 | 43 | 0 | 0.3575 | 0.3834 | 12.2250 | 0.9559 |
| task34_adaptive_penalty_lam0_30_tau0.75 | adaptive_multistep_penalty_lam0_30_v1 | 0.75 | none | 164 | 158 | 0 | 42 | 0 | 0.3280 | 0.3473 | 12.1200 | 0.8836 |
| task34_adaptive_penalty_lam0_20_tau0.75 | adaptive_multistep_penalty_lam0_20_v1 | 0.75 | none | 164 | 158 | 0 | 42 | 0 | 0.3290 | 0.3473 | 12.1250 | 0.8853 |
| ch5d20_multifixed_gradient_tau0.75 | receding_gradient_multiscale_v1 | 0.75 | none | 164 | 159 | 0 | 41 | 0 | 0.4504 | 0.5000 | 12.0050 | 1.2092 |
| ch5d20_multifixed_gradient_tau0.90 | receding_gradient_multiscale_v1 | 0.9 | none | 164 | 157 | 0 | 43 | 0 | 0.4805 | 0.5000 | 12.1150 | 1.2832 |
| ch5_sphere_shrink_32_matched_tau0.90 | random_sphere_coordinate_shrink_32_v1 | 0.9 | none | 163 | 148 | 0 | 52 | 0 | 0.5718 | 0.5000 | 6.9150 | 1.3005 |
| ch5_sphere_expanding_32_matched_tau0.90 | random_sphere_expanding_32_v1 | 0.9 | none | 163 | 148 | 0 | 52 | 0 | 0.9200 | 1.0000 | 12.9800 | 2.6856 |
| task34_adaptive_penalty_lam0_50_tau0.75 | adaptive_multistep_penalty_lam0_50_v1 | 0.75 | none | 161 | 155 | 0 | 45 | 0 | 0.3216 | 0.3473 | 12.1100 | 0.8625 |
| ch5d20_multifixed_penalty_lam0_10_tau0.60 | multifixed_penalty_direction_frontier_lam0_10_v1 | 0.6 | none | 161 | 165 | 7 | 35 | 0 | 0.3204 | 0.3316 | 11.6600 | 0.8652 |
| ch5d20_multifixed_gradient_tau0.60 | receding_gradient_multiscale_v1 | 0.6 | none | 158 | 162 | 7 | 38 | 0 | 0.4196 | 0.5000 | 11.8300 | 1.1308 |
| task34_adaptive_penalty_lam0_10_tau0.60 | adaptive_multistep_penalty_lam0_10_v1 | 0.6 | none | 158 | 163 | 8 | 37 | 0 | 0.3130 | 0.3282 | 11.7650 | 0.8454 |
| task34_adaptive_gradient_tau0.60 | adaptive_multistep_gradient_v1 | 0.6 | none | 157 | 162 | 8 | 38 | 0 | 0.3148 | 0.3282 | 11.7650 | 0.8505 |
| one_shot_gradient_bracket_v1_tau0.70_labelblind | one_shot_gradient_bracket_v1 | 0.7 | none | 151 | 144 | 0 | 49 | 7 | 0.3108 | 0.3095 | 11.9400 | 0.8372 |
| one_shot_gradient_bracket_v1_tau0.75_ch5 | one_shot_gradient_bracket_v1 | 0.75 | none | 151 | 143 | 0 | 50 | 7 | 0.3164 | 0.3286 | 11.9900 | 0.8512 |
| one_shot_gradient_bracket_v1_tau0.80_labelblind | one_shot_gradient_bracket_v1 | 0.8 | none | 151 | 143 | 0 | 50 | 7 | 0.3234 | 0.3286 | 12.0450 | 0.8684 |
| one_shot_gradient_bracket_v1_tau0.90_labelblind | one_shot_gradient_bracket_v1 | 0.9 | none | 151 | 140 | 0 | 53 | 7 | 0.3408 | 0.3476 | 12.0900 | 0.9114 |
| one_shot_gradient_bracket_v1_tau0.60_labelblind | one_shot_gradient_bracket_v1 | 0.6 | none | 146 | 148 | 7 | 45 | 7 | 0.3010 | 0.3095 | 11.6950 | 0.8129 |
| one_shot_gradient_bracket_v1_tau0.50_labelblind | one_shot_gradient_bracket_v1 | 0.5 | none | 137 | 150 | 17 | 43 | 7 | 0.2903 | 0.2905 | 11.2950 | 0.7861 |
