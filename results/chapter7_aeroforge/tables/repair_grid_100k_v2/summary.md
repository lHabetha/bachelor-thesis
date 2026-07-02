# Chapter 7 optimization grid — digest

Generated 2026-07-01T17:57:43+00:00

## T1 — Verified-OK recovery (optimizer x model, tau_decide=1.0)

| optimizer | a | b | c | d | e | f |
| --- | --- | --- | --- | --- | --- | --- |
| one_shot_gradient_line_search | 81 | 75 | 86 | 80 | 83 | 81 |
| receding_multistep_gradient | 92 | 89 | 92 | 91 | 88 | 92 |
| receding_multistep_penalty | 92 | 89 | 92 | 91 | 87 | 92 |
| full_coordinate_grid_refine | 62 | 63 | 62 | 63 | 53 | 62 |
| trust_region_hybrid | 80 | 77 | 81 | 79 | 73 | 80 |
| trust_region_hybrid_shrinkage | 76 | 70 | 68 | 70 | 64 | 76 |
| all | 94 | 90 | 93 | 91 | 90 | 94 |

_Reading: higher is better; each cell is the count (of 100) of CAD-verified clean repairs at the headline operating point._

## T2 — False-OK (surrogate false positives)

| optimizer | a | b | c | d | e | f |
| --- | --- | --- | --- | --- | --- | --- |
| one_shot_gradient_line_search | 16 | 15 | 14 | 14 | 17 | 16 |
| receding_multistep_gradient | 8 | 6 | 8 | 7 | 12 | 8 |
| receding_multistep_penalty | 8 | 6 | 8 | 7 | 13 | 8 |
| full_coordinate_grid_refine | 27 | 27 | 27 | 27 | 35 | 27 |
| trust_region_hybrid | 15 | 15 | 16 | 17 | 24 | 15 |
| trust_region_hybrid_shrinkage | 19 | 22 | 29 | 26 | 33 | 19 |
| all | 6 | 6 | 7 | 7 | 10 | 6 |

_Reading: lower is better; variant f (binary gate) should shrink these vs variant a._

## T5 — 'All' frontier (best of 6 per start)

| variant | union_recovery | best_L2_avg | best_L0_avg | top_winner |
| --- | --- | --- | --- | --- |
| a | 94 | 0.1894 | 2.5213 | full_coordinate_grid_refine |
| b | 90 | 0.1874 | 2.4111 | full_coordinate_grid_refine |
| c | 93 | 0.1513 | 2.3226 | full_coordinate_grid_refine |
| d | 91 | 0.1460 | 2.3187 | full_coordinate_grid_refine |
| e | 90 | 0.1559 | 2.6333 | full_coordinate_grid_refine |
| f | 94 | 0.1891 | 2.5319 | full_coordinate_grid_refine |

_Reading: union_recovery = #starts some optimizer repaired; the winner shows which optimizer earns its keep._

## T6 — Aero performance preservation (full-VLM quick sim, over successful repairs)

| optimizer | a_dL/D | b_dL/D | c_dL/D | d_dL/D | e_dL/D | f_dL/D |
| --- | --- | --- | --- | --- | --- | --- |
| one_shot_gradient_line_search | -505.6200 | 5.0783 | -479.4879 | 3.6273 | -498.7757 | -505.6200 |
| receding_multistep_gradient | -447.8332 | -463.2642 | -417.7430 | -453.4589 | -468.3483 | -447.8332 |
| receding_multistep_penalty | -447.1771 | -463.2996 | -417.6923 | -453.6029 | -473.8847 | -447.1771 |
| full_coordinate_grid_refine | -675.9137 | -665.0726 | -675.9137 | -665.0726 | -789.5782 | -675.9137 |
| trust_region_hybrid | -495.5334 | -342.5433 | -508.4562 | -529.3530 | -564.3834 | -495.5334 |
| trust_region_hybrid_shrinkage | -541.4124 | -375.7369 | -614.5135 | 135.8692 | 4.9271 | -540.1628 |
| all | -439.2916 | -458.8729 | -444.5493 | -453.8390 | -460.0569 | -439.2917 |

_Reading: mean Delta(L/D) = final - start over CAD-verified-clean repairs (both designs VLM-ok). Near 0 = repair preserved aerodynamics; strongly negative = overlap was fixed by hurting L/D. See `tables/T6_quicksim.csv` for CD0/CD/drag and medians; weight is not in the VLM quick path._


See `tables/*.csv` for the full P1-P7 tau/p* sweeps, T3 proximity, T4 feature-grad deltas, T6 aero, and CF.
