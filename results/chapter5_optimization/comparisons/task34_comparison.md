# Chapter 5 Adaptive Multi-Step Optimizer Analysis

Selected step-size rule: bold-driver backtracking with validity-aware shrinking, progress-based expansion, and post-crossing bracketing.

## Adaptive Tau Sweep

| Variant | Tau | Verified OK | False OK | Mean L2 | Mean L1 | Active coords | Run ID |
|---|---:|---:|---:|---:|---:|---:|---|
| Adaptive gradient | 0.60 | 157/200 | 8 | 0.3148 | 0.8505 | 11.77 | `task34_adaptive_gradient_tau0.60` |
| Adaptive gradient | 0.75 | 165/200 | 0 | 0.3320 | 0.8932 | 12.12 | `task34_adaptive_gradient_tau0.75` |
| Adaptive gradient | 0.90 | 165/200 | 0 | 0.3575 | 0.9559 | 12.22 | `task34_adaptive_gradient_tau0.90` |
| Adaptive penalty lambda=0.10 | 0.60 | 158/200 | 8 | 0.3130 | 0.8454 | 11.77 | `task34_adaptive_penalty_lam0_10_tau0.60` |
| Adaptive penalty lambda=0.10 | 0.75 | 165/200 | 0 | 0.3300 | 0.8876 | 12.12 | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Adaptive penalty lambda=0.10 | 0.90 | 165/200 | 0 | 0.3554 | 0.9503 | 12.22 | `task34_adaptive_penalty_lam0_10_tau0.90` |

## Penalty Sweep At Tau 0.75

| Variant | Tau | Verified OK | False OK | Mean L2 | Mean L1 | Active coords | Run ID |
|---|---:|---:|---:|---:|---:|---:|---|
| Adaptive penalty lambda=0.05 | 0.75 | 165/200 | 0 | 0.3314 | 0.8915 | 12.12 | `task34_adaptive_penalty_lam0_05_tau0.75` |
| Adaptive penalty lambda=0.10 | 0.75 | 165/200 | 0 | 0.3300 | 0.8876 | 12.12 | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Adaptive penalty lambda=0.20 | 0.75 | 164/200 | 0 | 0.3290 | 0.8853 | 12.12 | `task34_adaptive_penalty_lam0_20_tau0.75` |
| Adaptive penalty lambda=0.30 | 0.75 | 164/200 | 0 | 0.3280 | 0.8836 | 12.12 | `task34_adaptive_penalty_lam0_30_tau0.75` |
| Adaptive penalty lambda=0.50 | 0.75 | 161/200 | 0 | 0.3216 | 0.8625 | 12.11 | `task34_adaptive_penalty_lam0_50_tau0.75` |

## Sparse Adaptive Penalized Selection

| Variant | Tau | Verified OK | False OK | Mean L2 | Mean L1 | Active coords | Run ID |
|---|---:|---:|---:|---:|---:|---:|---|
| Adaptive penalty original L2 | 0.75 | 165/200 | 0 | 0.3300 | 0.8876 | 12.12 | `task34_adaptive_penalty_lam0_10_tau0.75` |
| Adaptive penalty L1->L2 | 0.75 | 165/200 | 0 | 0.2934 | 0.6756 | 7.59 | `task34_adaptive_penalty_sparse_l1_l2_tau0.75` |
| Adaptive penalty L0->L2 | 0.75 | 165/200 | 0 | 0.2934 | 0.6756 | 7.59 | `task34_adaptive_penalty_sparse_l0_l2_tau0.75` |
| Adaptive penalty L0->L1->L2 | 0.75 | 165/200 | 0 | 0.2934 | 0.6756 | 7.59 | `task34_adaptive_penalty_sparse_l0_l1_l2_tau0.75` |

## Reduced Adaptive Lock Matrix

| Variant | Tau | Verified OK | False OK | Mean L2 | Mean L1 | Active coords | Run ID |
|---|---:|---:|---:|---:|---:|---:|---|
| Adaptive gradient / lock_splint_user | 0.75 | 171/200 | 0 | 0.3410 | 0.8587 | 9.42 | `task34_adaptive_gradient_tau0.75_lock_lock_splint_user` |
| Adaptive gradient / lock_pin_user | 0.75 | 157/200 | 0 | 0.3489 | 0.8273 | 9.36 | `task34_adaptive_gradient_tau0.75_lock_lock_pin_user` |
| Adaptive gradient / lock_main_bracket_user | 0.75 | 170/200 | 0 | 0.4410 | 0.7713 | 6.66 | `task34_adaptive_gradient_tau0.75_lock_lock_main_bracket_user` |
| Adaptive gradient / grad_top_3 | 0.75 | 135/200 | 0 | 0.5035 | 1.1733 | 9.48 | `task34_adaptive_gradient_tau0.75_lock_grad_top_3` |
| Adaptive gradient / grad_bottom_12 | 0.75 | 200/200 | 0 | 0.4370 | 0.4370 | 1.00 | `task34_adaptive_gradient_tau0.75_lock_grad_bottom_12` |
| Adaptive gradient / grad_bottom_11 | 0.75 | 199/200 | 0 | 0.5132 | 0.7233 | 1.99 | `task34_adaptive_gradient_tau0.75_lock_grad_bottom_11` |
| Adaptive gradient / grad_bottom_10 | 0.75 | 187/200 | 0 | 0.3961 | 0.6787 | 2.97 | `task34_adaptive_gradient_tau0.75_lock_grad_bottom_10` |
| Adaptive penalty lambda=0.10 / lock_splint_user | 0.75 | 171/200 | 0 | 0.3390 | 0.8539 | 9.42 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_lock_splint_user` |
| Adaptive penalty lambda=0.10 / lock_pin_user | 0.75 | 156/200 | 0 | 0.3452 | 0.8196 | 9.36 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_lock_pin_user` |
| Adaptive penalty lambda=0.10 / lock_main_bracket_user | 0.75 | 154/200 | 0 | 0.3960 | 0.6879 | 6.65 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_lock_main_bracket_user` |
| Adaptive penalty lambda=0.10 / grad_top_3 | 0.75 | 124/200 | 0 | 0.4519 | 1.0484 | 9.46 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_grad_top_3` |
| Adaptive penalty lambda=0.10 / grad_bottom_12 | 0.75 | 189/200 | 0 | 0.4144 | 0.4144 | 1.00 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_grad_bottom_12` |
| Adaptive penalty lambda=0.10 / grad_bottom_11 | 0.75 | 192/200 | 0 | 0.4880 | 0.6854 | 1.99 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_grad_bottom_11` |
| Adaptive penalty lambda=0.10 / grad_bottom_10 | 0.75 | 183/200 | 0 | 0.3873 | 0.6628 | 2.97 | `task34_adaptive_penalty_lam0_10_tau0.75_lock_grad_bottom_10` |

## Tuning Summary

- Round 1 used a 32-start subgroup-balanced subset.
- Round 2 used all 200 starts through the lightweight tuning harness.
- `bold_backtrack` tied the best full-start verified count at 165/200 with zero false successes and lower mean L2 than the other top-coverage variants.

## Additional Diagnostics

- Subgroup breakdown CSV: `comparisons_task34/task34_subgroup_breakdown.csv`.
- Step-size and evaluation diagnostics CSV: `analysis/task34_adaptive_multistep/task34_step_diagnostics.csv`.

Trade-off plot written to `results/chapter5_optimization/comparisons/task34_optimizer_tradeoff.png`.
