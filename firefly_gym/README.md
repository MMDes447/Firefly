# firefly_gym — Gymnasium env for the FIREFLY Radiance office

Joint **venetian-blind + dimmable-lighting** control of the hyperspectral Radiance reference
office (Innsbruck) in `Implementation_Radiance/office_with_results`.

## Quick start

```bash
pip install numpy gymnasium
cd firefly_gym
# 1) build the env data from the Radiance matrices (~5 s, ~13 MB .npz)
python build_office_env_data.py ../Implementation_Radiance/office_with_results --out office_env_data.npz
# 2) check the env reproduces the Radiance results
python validate_against_radiance.py office_env_data.npz
```

```python
import gymnasium as gym, firefly_office_env  # registers FireflyOffice-v0
env = gym.make("FireflyOffice-v0", data_path="office_env_data.npz", episode_days=1)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step([1.0, 0.3])   # [shade, dimming]
info["costs"]   # {"illuminance": ..., "glare": ..., "glare_violation": ...}  -> CMDP costs
```

## What one step is

| | |
|---|---|
| Time step | 1 daytime hour (EPW hours 8–18 → 11 steps/day, 4015/year) |
| Action (continuous) | `a[0]` shade ∈ [0,1] → nearest shade state (now: open, closed 75°); `a[1]` dimming ∈ [0,1] of 4 × 98 W |
| Action (discrete) | `Discrete(n_states × n_dim_levels)` |
| Observation | sin/cos hour, sin/cos day, façade irradiance now + next hour, ceiling photosensor (daylight + electric), previous shade, previous dimming |
| Physics | `E_wp = E_wp_day[s,t] + u·E_wp_elec100` (441 points); `E_ceil` likewise; `P = u·392 W` |
| Glare | evalglare DGP from the repo (3 seats) where available, otherwise simplified DGP from seat eye illuminance |
| Reward | −(w_E·u + w_V·workplane shortfall below 500 lx + w_G·(DGP−0.4)+/0.1 + w_S·shade switch) — see `RewardConfig` |

## Validation (full year, env vs Radiance `results/`)

| Strategy | Energy env / Radiance | Light-on hours env / Radiance |
|---|---|---|
| Blinds always open | 92.07 / 92.07 kWh | 759 / 759 |
| Blinds always closed | 588.90 / 588.90 kWh | 3989 / 3989 |
| Rule-based (façade > 120 W/m²) | 173.90 / 173.90 kWh | 1657 / 1657 |

Workplane illuminance matches `Innsbruck_day_*_wp_ill.mtx` to 1e-7 relative error; sDA and UDI
match exactly. Seat eye illuminance from the `comp*.hsr` images agrees with evalglare to ~1.5 %.

## Adding more slat angles (the main extension)

Only the BSDF `T` changes with blind position — `V`, `D` and the sky matrix are reused, so
**no expensive Radiance run is needed** beyond making the BSDF:

1. Create a Klems BSDF XML for each slat angle (LBNL WINDOW, or `genBSDF` on a slat model).
2. Spectralise it exactly like `_office_step01.sh` step 1b:
   ```bash
   rmtxop -ff -c `rcalc -e '$1=0;$2=$1;$3=0' spectra/BauIng_518_smult_780-380.txt` \
       window/BSDF/Bauing_ILM_Lamelle_+45.xml > mtx/blinds+45.mtx
   ```
3. `python build_office_env_data.py ROOT --state slat45=mtx/blinds+45.mtx --state ...`

Order the states from most open to most closed so `a[0]` is monotone. For new states,
glare falls back to the simplified DGP (evalglare frames only exist for open / closed 75°).

## Known limits

* Hourly, daytime-only, one TMY year (IWEC Innsbruck). Multi-day episodes skip nights.
* No thermal model: solar gains / HVAC are not simulated.
* Occupancy is "always occupied" during 8–18, as in the repo.
* Electric light at the seats' eyes is not in the glare/eye-illuminance terms.

---

# v1: blind HEIGHT + SLAT TILT (`firefly_office_env_v2.py`, id `FireflyOfficeHeightTilt-v1`)

The original model only has two window states (blinds fully up / fully down at 75°).
v1 adds partial blind heights and slat angles.

## How it works

* **Height** – the window is split into 5 horizontal strips (top → bottom). Each strip has its own
  Radiance view matrices (workplane, ceiling sensor, seat eye illuminance) and daylight matrix, so a
  partly lowered blind is exact three-phase superposition: `E = Σ_z V_z · T_state(z) · D_z · s(t)`.
  Height level k covers the top k strips (0 %, 20 %, …, 100 %).
* **Tilt** – slat BSDFs for 0/15/30/45/60/75° from a genBSDF external-venetian-blind model
  (80 mm curved slats, 70 mm pitch, reflectance 0.68) calibrated to the measured `+75` file
  (hemispherical-transmittance correlation 0.99, RMSE 0.007), layered with the measured glazing
  (`XXX`) by the Klems matrix method. `closed_state="measured"` swaps in the measured 75° file.
* **Glare** – simplified DGP (Wienold 2009) from seat eye illuminance: agrees with the repo's
  evalglare DGP > 0.40 classification in 98.4 % of hours.

## Quick start

```python
import firefly_office_env_v2          # registers FireflyOfficeHeightTilt-v1
env = gym.make("FireflyOfficeHeightTilt-v1", data_dir="env_v2_data", episode_days=1)
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step([0.6, 0.4, 0.2])   # 60 % down, 30° tilt, 20 % dimming
```
`python validate_v2.py env_v2_data` reproduces the original results and prints an
energy/glare table over all height × tilt combinations.

## Rebuilding / higher quality (e.g. on LEO5)

`radiance_strips/run_strips.sh` recreates everything from `Implementation_Radiance/office`
(copy the folder's files plus `radiance_io.py` into `office/`, put the IWEC EPW in `weatherdata/`
and link `office_with_results/atmos_data`). Raise `-c` (daylight matrix) and `-ad` (view
matrices) to reduce Monte-Carlo noise; set `N=` for a different number of strips.

## Caveats (v1)

* Slat geometry is a calibrated generic model, not the manufacturer's data. For exact slat
  angles, drop manufacturer / WINDOW BSDFs into `gen_tilts.py`'s place.
* Full evalglare DGP only exists for the two original states; all states use DGPs.
* View matrices were computed with reduced settings (2 CPU cores): annual workplane daylight
  within 1 % of the original matrices, hourly per-point noise ~2 % median.

## Online temporal abstraction and options

The [temporal-options pilot](TEMPORAL_RL.md) adds an online manager over blind
settings and 1–3-hour durations, with hourly local dimming. It includes matched
hourly/fixed-duration comparisons, a command-hold ablation, held-out weeks,
duration-aware Double DQN targets, and separate comfort-cost reporting.
See the linked guide for execution commands, accounting details and limitations.
