# FIREFLY: office lighting, blind control and temporal options

Gymnasium environments for joint blind and electric-light control, built from the
FIREFLY Radiance office model. The online temporal-options pilot adds a learned
manager for blind settings and 1–3-hour durations, with hourly local dimming.
HVAC is outside the current scope.

The reference simulation dataset is published separately on
[Zenodo](https://doi.org/10.5281/zenodo.20344638). This repository contains our
environment, data-building, validation and learning code. Reference archives,
weather files, simulation matrices, runtime arrays and trained checkpoints are
excluded from version control.

## Start here

- [Environment and Radiance workflow](firefly_gym/README.md)
- [Temporal-options implementation and experiment guide](firefly_gym/TEMPORAL_RL.md)
- [Initial short-run results](firefly_gym/results/temporal_smoke_20261010/SUMMARY.md)
- [Repository instructions and model details](AGENTS.md)

## Installation and data

Use Python 3.10 or later. The implementation was checked with Python 3.12,
NumPy 1.26.4, Gymnasium 1.2.3 and PyTorch 2.5.1.

```bash
python -m pip install numpy gymnasium torch pytest
```

A fresh clone does **not** contain the data needed to run the environments.

- For `FireflyOffice-v0`, obtain the reference office results separately and use
  `firefly_gym/build_office_env_data.py` as described in the environment guide.
- For `FireflyOfficeHeightTilt-v1` and the options pilot, place the existing
  `env_v2_data/` directory under `firefly_gym/`, or rebuild it using the
  `radiance_strips` workflow. Rebuilding requires Radiance and the reference
  inputs. The per-strip daylight matrices are not distributed here; recompute
  them or explicitly select the documented approximation.

Keep license-restricted weather files local. Treat the downloaded
`Implementation_Radiance/` reference directory as read-only.

## Run the pilot

From the repository root, after preparing the v1 runtime data:

```bash
python -m pytest firefly_gym/tests/test_temporal.py -q
python -m firefly_gym.train_temporal --train-steps 2200 --warmup-hours 220 --seeds 0
```

The runner compares hourly control, fixed-duration options, learned durations and
holding both actuator commands, alongside the rule-based and fixed-30° references.
It saves metrics and inference checkpoints locally. Data-dependent tests are
skipped when runtime data is missing; a passing code-only test run does not
validate the lighting simulation.

This is an initial research pilot. The low-level dimming controller is fixed and
uses the current candidate-state daylight model. Costs are logged separately;
explicit constraint-budget enforcement and comfort-aware early termination are
not implemented. The short-run results do not establish superiority or a safety
guarantee.
