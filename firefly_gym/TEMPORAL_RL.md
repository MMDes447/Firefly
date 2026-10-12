# Online temporal-options pilot

This is the first implementation for testing hierarchy and temporal abstraction
on `FireflyOfficeHeightTilt-v1`. It controls blinds and electric lighting. HVAC is
outside this experiment. The original environment, physics, reward weights,
comfort thresholds and daily terminal semantics are unchanged.

## Hierarchy and execution

The learned manager selects a blind height, slat tilt and duration. Each option
keeps the blind command fixed while the local controller adjusts electric-light
dimming every primitive hour. The manager sees the original 10 observations,
including the exact next-hour facade-irradiance forecast. It receives a new
observation after its option ends.

There are 6 height commands and 6 tilt commands. With durations 1, 2 and 3 hours,
the manager has 108 actions. All six tilt commands remain selectable with the
blinds retracted: their optical effects are identical there, but the original
environment remembers the tilt index and uses that memory in subsequent movement
penalties. Removing these aliases would change the action/reward semantics.

The local controller uses the existing 625 lx equivalent ceiling-sensor dimming
law. It reads the simulator's **current candidate-state daylight**, then calculates
the dimmer command. This is a model-assisted local controller, not a controller
using only the manager's ten observations. All comparisons use the same model
access. The low-level controller is fixed; it is not learned in this pilot.

An option ends after its chosen number of hours or at the end of the day/episode.
In a two-day base episode, an option also stops before the skipped night, without
marking the whole episode terminal. The supplied trainer uses one-day episodes.
There is no early comfort interruption or learned termination function yet.

## Comparisons

| Variant | Manager decisions | Local dimming |
|---|---|---|
| `hourly` | Every hour | Recomputed each hour |
| `fixed2` | Every 2 hours, shortened at day end | Recomputed each hour |
| `fixed3` | Every 3 hours, shortened at day end | Recomputed each hour |
| `duration` | Learns 1, 2 or 3 hours along with blind setting | Recomputed each hour |
| `hold3` | Every 3 hours, shortened at day end | Calculated once and held with the blinds |
| `rule_based` | Original facade threshold of 120 W/m² | Original hourly dimming law |
| `fixed30` | Fully down at 30°, throughout each day | Original hourly dimming law |

The default run trains `hourly`, `fixed3`, `duration` and `hold3`, and evaluates the
two reference controllers. Include `fixed2` to complete the fixed-duration grid.
`hold3` versus `fixed3` tests primitive-command persistence against hourly local
feedback at the same manager clock. `fixed3` versus `duration` changes duration
selection. `hourly` uses the same fixed dimming controller; it is **not** a fully
learned flat joint blind/dimmer policy or the earlier SDAR method.

The manager is a small feedforward Double DQN, trained from newly collected
simulator interaction with an online replay buffer. Hidden layers are 64 × 64;
output size changes with the action set. No offline dataset is used for learning.
The underlying physical simulator is still the precomputed Radiance year.

For an option lasting `tau` actual hours, learning uses:

```text
R = r_0 + gamma*r_1 + ... + gamma**(tau-1)*r_(tau-1)
target = R + gamma**tau * (1 - terminated) * Q_target(next_obs, argmax Q(next_obs))
```

`gamma = 0.99` is per primitive hour. A shortened option uses its actual duration.
The base daily terminal masks the bootstrap. The target helper also supports
bootstrapping through truncation, but the current base never reports truncation.
Do not plug this wrapper into ordinary DQN/PPO using a fixed discount per wrapper
step: the returned Gym reward is an undiscounted sum. The provided learner uses
`info['discounted_reward']` and `info['discount']` instead.

Costs remain separate in `info['costs']`, summed over the option. Discounted costs
and every primitive reward/info record are available too. Evaluation divides cost
totals by primitive hours, not by the number of manager decisions. It reports
energy, workplane shortfall, glare, blind movement, dimmer changes and decision
counts. Blind movement units follow the base reward convention; they do not
represent measured motor energy.

**The learner optimizes the existing reward and logs the CMDP costs. It does not
yet enforce cost budgets.** Glare uses the environment's DGPs approximation.

## Run from the repository root

Dependencies: NumPy, Gymnasium, PyTorch; pytest for checks. The existing
`env_v2_data` directory is sufficient. No Radiance rebuild is needed.

```bash
# Accounting, boundaries, controller parity and learning-target tests
python -m pytest firefly_gym/tests/test_temporal.py -q

# Short execution check, 2,200 primitive training hours per learned policy
python -m firefly_gym.train_temporal --train-steps 2200 --warmup-hours 220 \
  --seeds 0 --out /tmp/firefly-temporal-smoke

# Initial multi-seed experiment, with the full fixed-duration grid
python -m firefly_gym.train_temporal --train-steps 22000 --seeds 0 1 2 \
  --variants hourly fixed2 fixed3 duration hold3
```

The output directory must be new. Without `--out`, a timestamped directory is
created under `firefly_gym/results/`. It contains `metrics.json` and one `.pt`
inference checkpoint per learned policy/seed. Checkpoints do not contain replay or
optimizer state, so they are not training-resume files. To repeat evaluation:

```bash
python -m firefly_gym.train_temporal \
  --evaluate-checkpoint /tmp/firefly-temporal-smoke/duration_seed0.pt \
  --out /tmp/firefly-temporal-reevaluation
```

The default `closed_state='model'` uses generated tilt BSDFs. The existing
`--closed-state measured` selector instead uses the reference `m75` entry at 75°.
Its historical name is retained; it does not establish that the original BSDF was
measured. A checkpoint's saved setting takes precedence during reevaluation.

## Evaluation protocol and interpretation

Each cycle of seven week-long blocks uses five training weeks, one validation
week and one test week. This gives 267 training days, 49 validation days and 49
test days. Days are sampled only from training during learning. All variants with
the same seed receive the same sampled days. The replay buffer holds only those
training transitions. Each policy receives the requested **primitive-hour**
budget, with optimizer-update opportunities and exploration decay also scheduled
in primitive hours. Replay capacity counts macro transitions.

Validation and test both evaluate every held-out day with a frozen greedy policy.
The runner performs no hyperparameter selection or best-checkpoint selection.
Use validation results to design subsequent experiments, then freeze settings
before a final test evaluation. Inspecting test results repeatedly during method
development would turn that test split into another validation set.

Energy totals in the report cover only the named split; they are not annual
energy figures. Reported glare percentage means at least one seat exceeds the
threshold, which differs from the worst individual seat's percentage printed by
the original validator. Daily resets remove overnight action carryover and impose
no terminal blind-reset penalty, matching the base environment.

This is a reproducible starting scaffold, not evidence that hierarchy or adaptive
duration wins. Important next experiments are recurrent managers, a strong flat
update/hold baseline, and comfort-aware early termination with matched feedback
and decision/movement budgets. A constrained learner is also needed before
claiming cost-budget enforcement. Those changes are not implemented here.

There is only one weather year, so held-out weeks test within-year generalization.
The light response is instantaneous and weather is action-independent; movement
penalties and temporal commitment provide the current intertemporal effects.
Duration selection has no extra decision penalty. A preference for one-hour
options can therefore be a legitimate result, and fewer manager decisions must
not be equated with fewer physical blind movements.
