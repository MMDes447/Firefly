# Temporal-options pilot: execution check

The first online pilot runs end to end on the existing lighting and blind environment. Each learned policy used seed 0, 2,200 primitive training hours and 496 optimizer updates. This is a short execution check, not a converged benchmark or evidence of superiority.

Evaluation below covers the same 49 test days (539 daytime hours) for every controller. Energy figures are for these days only. Policies have different comfort outcomes, so energy alone does not rank them fairly.

| Controller | Energy (kWh) | Mean illuminance shortfall | Any-seat glare hours | Manager calls | Blind moves |
|---|---:|---:|---:|---:|---:|
| hourly | 11.43 | 0.0991 | 1.86% | 539 | 124 |
| fixed3 | 12.33 | 0.0999 | 5.38% | 196 | 77 |
| duration | 12.85 | 0.1104 | 0.56% | 459 | 108 |
| hold3 | 17.76 | 0.0907 | 6.31% | 196 | 59 |
| rule_based | 24.22 | 0.0978 | 1.86% | 539 | 69 |
| fixed30 | 20.11 | 0.2474 | 0.19% | 539 | 49 |

Illuminance shortfall is the mean of max(0, 1 − illuminance / 500 lx), across workplane points and hours. Glare means the environment’s DGPs proxy exceeds 0.40 at any seat. Blind movement counts follow the original environment’s command convention. A manager call need not move the blinds.

The learned-duration policy executed 379 one-hour options and 80 two-hour options on the test days. Its mean option length was 1.174 hours. These results do not establish that longer options are preferable.

Validation completed:

- 18 tests passed for duration-aware reward/cost accounting, boundaries, execution behavior, baseline parity, train/test separation, checkpoints and learning targets.
- Reloading the learned-duration checkpoint reproduced every validation and test metric exactly.
- Original environment and validator source checksums remained unchanged.
- The existing full-year validator reproduced v1 open/reference-75°/rule energy of 92.8 / 592.4 / 176.2 kWh. The original whole-window values are 92.07 / 588.90 / 173.90 kWh; the existing strip-model differences remain.

The local dimming controller is fixed and uses the current candidate-state daylight model. Learned low-level control, comfort-aware early termination, recurrent policies and explicit cost-budget enforcement are not implemented in this first pilot. HVAC is outside this experiment.

See [the implementation and experiment guide](../../TEMPORAL_RL.md) for the option definitions, comparisons, held-out-week protocol and commands. The original local run also saved full metrics and inference checkpoints in this directory; those generated files are excluded from Git. Rerunning the documented command creates a new local run directory.
