# AGENTS.md: FIREFLY office daylighting + RL environment

This file is for coding agents (Codex and similar). It explains what the repo is, how the
physics works, where each file is, which numbers count as ground truth, and what to avoid.
Read all of it before you change any code.

---

## 1. What this repo is (short version)

Research code by Mohammad Esmaeili (PhD, University of Innsbruck, safe RL for building energy
management). It has two parts:

1. **`Implementation_Radiance/`** is the published **FIREFLY reference Radiance model** (Zenodo,
   DOI 10.5281/zenodo.20344638). It is a hyperspectral (20 bands, 780→380 nm) Radiance 6.1a
   three-phase simulation of a reference **office** in Innsbruck and a reference **hall** in
   Stuttgart. It was run on the LEO5 cluster. **This is third-party reference data. Treat it as
   read-only.**
2. **`firefly_gym/`** is our code. It turns the office's Radiance matrices into **Gymnasium
   environments** for joint control of the **external venetian blinds** and the **dimmable electric
   lighting**. The environments are designed as **CMDPs**: reward = energy and comfort, and
   constraint costs (under-illumination, glare) are reported separately in `info["costs"]`.
   - `FireflyOffice-v0`: 2 blind states (retracted / fully down at 75°) + dimming. It reproduces
     the Radiance results exactly.
   - `FireflyOfficeHeightTilt-v1`: blind **height** (5 strips) × **slat tilt** (0–75°) + dimming.

Each env step is a table lookup plus a linear electric-light term. There is no Radiance at
runtime, so the envs run at ~10⁴–10⁵ steps/s.

---

## 2. Directory map

```text
firefly/                                   (source repository; large data stays local)
├── README.md                              project entry point and data setup
├── .gitignore                             excludes reference data, arrays and checkpoints
├── AGENTS.md                              ← this file
├── Implementation_Radiance.zip            6.3 GB  original Zenodo download (duplicate; ignore)
├── Claude outputs/                        stale delivery zips from an earlier session (ignore)
├── Implementation_Radiance/               READ-ONLY reference model
│   ├── README.md
│   ├── office.zip, office_with_results.zip (3.9 GB), hall.zip, hall_with_results.zip (2.5 GB)
│   ├── office/                            inputs only (geometry, materials, BSDFs, pts, scripts)
│   │   ├── _office_step00_all.sh … _office_step20.sh   the Radiance pipeline (see §4)
│   │   ├── base_model.rad / base_model_night.rad
│   │   ├── objects/ materials/ sources/ spectra/ pts/ misc/ weatherdata/
│   │   └── window/window_kf.rad, window/BSDF/Bauing_ILM_Lamelle_{XXX,+75}.xml
│   └── office_with_results/               same + every intermediate matrix and result
│       ├── mtx/          (~1.9 GB; the matrices the envs are built from, see §5)
│       ├── mtx/seat{1,2,3}/comp000..144.hsr   per-Klems-patch fisheye images (145 per seat)
│       ├── results/      KPI files + anim/seat*/{blinds,noblinds}_evg.txt (hourly evalglare)
│       ├── atmos_data/   aerosol tables gensdaymtx needs (200 files)
│       └── weatherdata/AUT_Innsbruck.111200_IWEC.epw   (license-restricted; do not redistribute)
└── firefly_gym/                           OUR CODE
    ├── README.md                          user-facing docs for both envs
    ├── radiance_io.py                     readers: Radiance .mtx/.vec, flat .hsr; photopic weights
    ├── build_office_env_data.py           v0 data builder (office_with_results → office_env_data.npz)
    ├── firefly_office_env.py              FireflyOffice-v0 (+ rule_based_action baseline)
    ├── validate_against_radiance.py       v0 full-year check against Radiance results
    ├── firefly_office_env_v2.py           FireflyOfficeHeightTilt-v1 (filename says v2, id says v1)
    ├── validate_v2.py                     v1 full-year check + height×tilt energy/glare table
    ├── temporal_options.py               time-extended blind options with hourly dimming
    ├── train_temporal.py                 online semi-Markov Double DQN pilot and evaluation
    ├── TEMPORAL_RL.md                    experiment definitions, commands and limitations
    ├── tests/test_temporal.py            accounting, boundary and learning-target tests
    ├── env_v2_data/                       v1 runtime data (≈138 MB, already built)
    │   ├── meta.npz
    │   └── wp/s{1..5}_{open,t00,t15,t30,t45,t60,t75,m75}.npy   (40 files, float16)
    ├── radiance_strips/                   how env_v2_data was produced (needs Radiance)
    │   ├── run_strips.sh                  end-to-end rebuild script
    │   ├── make_strips.py                 splits the window into N horizontal strips
    │   ├── gen_tilts.py, bsdf_tools.py    genBSDF slat model + Klems layering with glazing
    │   ├── build_v2_data.py               strip matrices → env_v2_data/
    │   ├── strip{1..5}.rad, window_strips5.rad, seats_eye.pts
    │   ├── bsdf/Bauing_model_tilt{00..75}.xml     generated system BSDFs
    │   └── mtx/  tilt*.mtx, wp_strip*.mtx, ceil_strip*.vec, seat_strip*.mtx (no daylight_strip*)
    └── results/firefly_blind_study.html   earlier analysis report (height × tilt study)
```

`firefly_gym/office_env_data.npz` (v0 data) is **not** in the repo. Build it (see §7).

---

## 3. The physical model

### Office room
- About 5.6 × 4.9 m, ceiling ≈ 2.7 m. One glazed façade on the **−x side**, made of two panes
  (≈ 3.9 m × 1.66 m in total). Model coordinates are absolute metres with an offset: window
  z ∈ [19.255, 20.915] and seat eye height z = 19.3. Workplane is 0.75 m above the floor.
- **Shading:** *external* venetian blinds. The measured BSDFs are
  `Bauing_ILM_Lamelle_XXX.xml` (retracted, glazing only) and `Bauing_ILM_Lamelle_+75.xml`
  (fully down, slats closed at 75°).
- **Lighting:** 4 × 98 W fluorescent 840 luminaires (Ludwig Leuchten 2722-001), so P_max = 392 W.
  Dimming is linear (perfect dimming efficacy, dims to 0).
- **Climate:** Innsbruck IWEC EPW. Only daytime hours are simulated: EPW hours 8–18, which gives
  **11 steps/day × 365 = 4015 steps/year**. Index: `t = (day_of_year − 1)·11 + (hour − 8)`.
  EPW hour h means the interval ending at h:00, and the env encodes time as `hour − 0.5`.
- **Sensors:** 441-point (21×21) workplane grid (`pts/workplane.pts`). One downward-looking
  ceiling photosensor (`ceil_sensor.pts`), which is the closed-loop dimming sensor. One exterior
  vertical façade irradiance sensor (`facade_sensor.pts`), which drives the rule-based blind
  control (threshold 120 W/m²). Three seats with fisheye views (seat1/2/3) for glare.

### Three-phase method (the core equation)
For each spectral channel c and each daytime hour t:

```
E(t) = V · T_state · D · s(t)
  V        view matrix, sensors ← 145 Klems window patches     (wp_sensor_kf.mtx, ceil_sensor_kf.vec, seat images)
  T_state  window-system BSDF incl. glazing spectrum, 145×145   (noblinds.mtx, blinds+75.mtx, tilt*.mtx)
  D        daylight matrix, window ← 2306 Reinhart m4 sky patches (daylight_r4.mtx)
  s(t)     spectral sky vector for hour t                         (Innsbruck_daytime_sky4.mtx, 2306×4015×20)
  D·s      precomputed: Innsbruck_daytime_kf.mtx (145×4015×20)
```
Photopic quantities (lux) come from contracting the 20 spectral channels with
`radiance_io.PHOTOPIC_W`, which sums to 179 lm/W. This matches `rmtxop -c Y` to ~1e-8.
**Only T depends on the blind state.** That is why a new slat angle needs only a new BSDF and no
new expensive Radiance run.

### Blind height (v1): strip superposition
The window is split into N = 5 horizontal strips, strip 1 at the top. Each strip z has its own
V_z and D_z:
```
E(t) = Σ_z V_z · T_{state(z)} · D_z · s(t)
```
Height level k ∈ {0..5} means the top k strips get the chosen slat state and the rest stay open.
The sum of the strips reproduces the whole-window result: annual workplane daylight −0.6 % and
sDA identical. Per-strip D_z differs from the whole-window D by only ~0.1 % (the façade is flat),
so `D_z ≈ D` is an acceptable approximation (`APPROX_D=1` in `build_v2_data.py`).

### Slat tilt (v1)
Tilt BSDFs at 0/15/30/45/60/75° come from a **genBSDF** blind model: 80 mm curved slats
(r = −0.08), 70 mm pitch, ρ = 0.68, with genblinds angle = −tilt. This model is **layered with the
measured `XXX` glazing** by the Klems matrix method (`bsdf_tools.layer`). It was calibrated
against the measured `+75` file: hemispherical Tf correlation 0.99, RMSE 0.007.
State names: `open`, `t00`…`t75` (model), `m75` (measured 75°). `closed_state="measured"` swaps
`m75` in for `t75`.
Spectralisation follows `_office_step01.sh` step 1b exactly:
`rmtxop -ff -c <glazing spectrum> file.xml > mtx/...`.

### Glare
- The Radiance repo computes **evalglare DGP** on hourly fisheye HDR frames (steps 10–11), but
  only for `open` and `closed 75°`.
- v1 uses **simplified DGP (DGPs)** from vertical eye illuminance: `DGPs = 6.22e-5·Ev + 0.184`.
  This agrees with evalglare on the DGP > 0.40 classification in 98.4 % of 12,045 seat-hours.
- v0 uses the evalglare value where it exists. Otherwise it falls back to `5.87e-5·Ev + 0.16`
  (the textbook Wienold coefficients; see §9).
- In v0, seat Ev comes from the comp*.hsr images: Ev = π × mean radiance over the fisheye disk.
  This is within ~1.5 % of evalglare's Ev. In v1, seat Ev comes from `rxfluxmtx` point sensors
  (`seats_eye.pts`).
- The glare threshold is DGP 0.40 ("disturbing").

---

## 4. The Radiance pipeline (`Implementation_Radiance/office/_office_stepNN.sh`)

| Step | What it does | Main output |
|---|---|---|
| 01 | Normalise glazing/lamp spectra to V(λ); spectralise both BSDFs | `mtx/noblinds.mtx`, `mtx/blinds+75.mtx` |
| 02 | Daytime-only EPW (8–18); spectral sky matrices (all, sun-only) | `Innsbruck_daytime_sky4.mtx`, `Innsbruck_sunonly_sky4.mtx` |
| 03 | Daylight matrix D (window → sky) | `daylight_r4.mtx` |
| 04 | D·s and T·D·s for both states (all-sky and sun-only) | `Innsbruck_daytime_kf.mtx`, `Innsbruck_day_{blinds,noblinds}.mtx` |
| 05 | Seat view matrices as 145 fisheye component images | `mtx/seat*/comp*.hsr` |
| 06 / 07 | Workplane V (full / direct-only) | `wp_sensor_kf.mtx`, `wp_direct_kf.mtx` |
| 08 | Annual workplane illuminance; sDA300/50 %, UDI100–3000 | `*_wp_ill.mtx`, `results/*.sDA*`, `*.UDI*` |
| 09 | Sun-only direct workplane illuminance; ASE proxy | `results/*.ASE_1000lx_250h` |
| 10 / 11 | Hourly HDR frames per seat/state; evalglare; share of hours with DGP > 0.4 | `results/anim/seat*/…_evg.txt` |
| 12 | Night-time spectral render seat 2; mEDI | `results/night_seat2.*` |
| 13 / 14 / 15 | Electric-only illuminance: workplane, ceiling sensor, workplane average | `electric_wp_ill.mtx`, `electric_ceil_sensor_ill.txt`, `electric_wp_ill_avg.txt` |
| 16 | Ceiling-sensor daylight V | `ceil_sensor_kf.vec` |
| 17 | Closed-loop dimming to 625 lx-equivalent (500 lx / 0.8 MF); energy & hours | `electric_light_levels_*.txt`, `results/electric_light_*` |
| 18 / 19 | Façade sensor vector; hourly exterior vertical irradiance; blinds when > 120 W/m² | `Innsbruck_daytime_facade_ext_irrad.txt`, `Innsbruck_daytime_useblinds.txt` |
| 20 | Rule-based combined control: energy & hours | `results/electric_light_annual_control.energy` |

**Dimming law** (steps 17/20, reused by the env baselines):
`TGI = 625 · E_ceil_elec100 / mean(E_wp_elec100)`, `u = clip((TGI − E_ceil_day)/E_ceil_elec100, 0, 1)`,
energy per hour = `u · 0.392 kWh`.
Constants: E_wp_elec100 mean = 1274.42 lx, E_ceil_elec100 = 418.08 lx.

---

## 5. Data files the envs use

### v0: `office_env_data.npz` (built by `build_office_env_data.py`, ~13 MB)
| key | shape | meaning |
|---|---|---|
| `state_names` | (S,) | `["open", "closed75", …extra --state]` |
| `E_wp_day` | (S, 4015, 441) | daylight workplane illuminance, lx |
| `E_ceil_day` | (S, 4015) | daylight at ceiling sensor, lx |
| `Ev_seat_day` | (S, 3, 4015) | vertical eye illuminance per seat (NaN if comp*.hsr missing) |
| `dgp_evalglare` | (S, 3, 4015) | evalglare DGP (NaN for states without frames) |
| `facade_irr` | (4015,) | exterior vertical irradiance, W/m² |
| `E_wp_elec100`, `E_ceil_elec100` | (441,), scalar | electric light at 100 % |
| `P_lights_max_W` | scalar | 392 |
| `day_of_year`, `hour` | (4015,) | 1..365, 8..18 |
| `wp_xy`, `seats`, `photopic_w` | | metadata |

### v1: `env_v2_data/`
- `wp/s{z}_{state}.npy`: (4015, 441) **float16 in units of 10 lx**. Multiply by
  `meta["wp_scale"]` (= 10). Workplane for strip z alone.
- `meta.npz`: `n_strips`, `state_names` (open, t00…t75, m75), `tilts_deg`, `strip_edges_z`,
  `ceil_day` (N, S, 4015), `ev_day` (N, S, 3, 4015), `dgp_evalglare_full` (2, 3, 4015, only open
  and m75 for the whole window), `facade_irr`, `E_wp_elec100`, `E_ceil_elec100`, `P_lights_max_W`,
  `day_of_year`, `hour`, `wp_xy`, `seats`, `wp_scale`.
- Per-strip values are **additive**. Sum over strips with each strip's own state.

### Reading Radiance files
Always use `radiance_io.read_rmtx` (header + ascii/float/double body → `(NROWS, NCOLS, NCOMP)`)
and `radiance_io.read_hsr` (flat `Radiance_spectra` pictures, 20 mantissas + shared exponent).
The spectral channel order is **780 → 380 nm**.

---

## 6. Environment APIs

### `FireflyOffice-v0` (`firefly_office_env.py`, class `FireflyOfficeEnv`)
- **ctor:** `data_path, episode_days=1, day_range=(1,365), action_type="continuous"|"discrete",
  n_dim_levels=11, reward=RewardConfig(), seats=(0,1,2)`
- **action (continuous):** `Box(0,1,(2,))`. `a[0]` is the shade position, rounded to the nearest
  of S states (ordered open → closed). `a[1]` is dimming u.
  **(discrete):** `Discrete(S·n_dim_levels)`, decoded as `divmod(a, n_dim_levels)`.
- **obs (9):** sin/cos hour, sin/cos day, façade irr now/500, façade irr next hour/500 (perfect
  1-h forecast, 0 at end of day), ceiling-sensor reading/1000 (under the *previous* action),
  prev shade (normalised), prev dim.
- **reward:** `−(w_energy·u + w_visual·shortfall + w_glare·glare_excess/0.1 + w_switch·[shade changed])`.
  Defaults: 1, 2, 2, 0.05, target 500 lx, DGP 0.40. `shortfall` = mean over the 441 points of
  `clip(1 − E/500, 0, 1)`. `glare_excess` = max over seats of `(DGP − 0.40)+`.
- **info:** `energy_kWh, E_wp_mean, E_wp_min, E_ceil, frac_wp_below_target, dgp[3], facade_irr,
  shade_state, dim, day_of_year, hour, costs{illuminance, glare, glare_violation}`.
- **baseline:** `rule_based_action(env)` is the repo strategy: close when façade > 120 W/m²,
  then closed-loop dimming.

### `FireflyOfficeHeightTilt-v1` (`firefly_office_env_v2.py`, class `FireflyOfficeHeightTiltEnv`)
- **ctor:** `data_dir="env_v2_data", episode_days, day_range,
  action_type="continuous"|"multidiscrete", n_dim_levels=11, closed_state="model"|"measured", reward`
- **action (continuous):** `Box(0,1,(3,))` = [height → k = round(a·5), tilt → nearest of 6, dimming u].
  **(multidiscrete):** `MultiDiscrete([6, 6, n_dim_levels])`.
- **obs (10):** same as v0, but "prev shade" is replaced by prev height and prev tilt.
- **reward:** like v0, but the switch term is replaced by
  `w_move·(|Δk| + |Δtilt_index|·[k>0])` with `w_move = 0.02`.
- **info:** adds `height_level, height_frac, tilt_deg, Ev_seats`. `dgp` is always DGPs here.
- **helpers:** `env.daylight(k, j, t)` returns `(wp[441], ceil, ev[3])`, daylight only.
  `rule_based_action(env)` is fully down at 75° above 120 W/m².

### Episodes (both)
- `reset(options={"start_day": d})` pins the start day. Otherwise the start day is drawn uniformly
  in `day_range`.
- Episode length = `11·episode_days` steps.
- `terminated=True` at the end of the horizon and `truncated` is always False. Nights are
  skipped, so consecutive days are not continuous in time.

---

## 7. How to run

```bash
pip install numpy gymnasium            # the only runtime deps; there is no requirements.txt yet
cd firefly_gym

# v0: build data once (reads ~50 MB of matrices + 435 .hsr images; seconds)
python build_office_env_data.py ../Implementation_Radiance/office_with_results --out office_env_data.npz
python validate_against_radiance.py office_env_data.npz

# v1: data already present
python validate_v2.py env_v2_data
```

```python
import gymnasium as gym
import firefly_office_env, firefly_office_env_v2      # importing registers the ids
env = gym.make("FireflyOfficeHeightTilt-v1", data_dir="env_v2_data", episode_days=1)
obs, _ = env.reset(seed=0)
obs, r, term, trunc, info = env.step([0.6, 0.4, 0.2])  # 60 % down, 30°, 20 % dimming
info["costs"]   # CMDP constraint signals
```

Rebuilding the v1 data from scratch needs **Radiance ≥ 6.1** (`rxfluxmtx`, `genBSDF`,
`gensdaymtx`, `rmtxop`). Check with `which rxfluxmtx`. Do not assume it is installed. The
workflow:
1. Copy `Implementation_Radiance/office` to a scratch dir.
2. Copy `radiance_strips/*` and `radiance_io.py` into it.
3. Put the EPW in `weatherdata/` and link `office_with_results/atmos_data`.
4. Run `NP=<cores> N=5 sh run_strips.sh`.

The scripts use **cwd-relative paths** (`mtx/strips/…`, `pts/…`) and must be run from inside that
prepared copy. The per-strip daylight matrices (`daylight_strip*.mtx`) are **not** in the repo.
Either recompute them (step 5 of `run_strips.sh`) or set `APPROX_D=1`. Raise `-c` and `-ad` on
LEO5 for less Monte-Carlo noise.

---

## 8. Ground-truth numbers (regression targets)

Any change to env physics or data must still reproduce these.

**Radiance reference** (`office_with_results/results/`) vs **v0 env**: identical to the printed
precision.

| Strategy | Lighting energy kWh/yr | Light-on hours |
|---|---|---|
| Blinds always open | 92.07 | 759 |
| Blinds always closed (75°) | 588.90 | 3989 |
| Rule-based (façade > 120 W/m²) | 173.90 | 1657 |

| Metric | Open | Closed 75° |
|---|---|---|
| sDA 300 lx / 50 % | 98.19 % | 0 % |
| UDI 100–3000 | 98.19 % | 4.54 % |
| ASE proxy 1000 lx / 250 h | 26.08 % | 0 % |
| Share of hours with DGP > 0.40, seat1 / seat2 / seat3 | 7.8 / 12.5 / 5.1 % | 0 / 0 / 0 % |

Other: v0 workplane illuminance matches `Innsbruck_day_*_wp_ill.mtx` to ~1e-7 relative error.
mEDI at seat 2 (night) = 117.4.

**v1 checks:**
- With `closed_state="measured"`, v1 reproduces the table above to within the strip-matrix noise
  (annual workplane daylight −0.6 %, sDA identical).
- Hourly per-point noise is ~2 % median (reduced `-ad`, 2 cores).
- DGPs vs evalglare agreement is 98.4 %.

**Key study finding** (`results/firefly_blind_study.html`): fully down at **30° tilt ≈ 177 kWh/yr
with ~0.1 % glare hours**, vs rule-based 174–176 kWh with ~1 % glare hours.

---

## 9. Known quirks and inconsistencies (understand before "fixing")

1. **Naming:** `firefly_office_env_v2.py` and `env_v2_data/` implement gym id
   **`FireflyOfficeHeightTilt-v1`**. If you rename one, update all of them, plus both READMEs and
   the validators.
2. **Two DGPs formulas:** v0's fallback uses `5.87e-5·Ev + 0.16` (standard Wienold). v1 uses the
   recalibrated `6.22e-5·Ev + 0.184`. The v0 fallback is only hit for extra `--state` entries.
   Unify on the calibrated one if you touch this, and document the change.
3. In v1, `w_move` ignores tilt changes while the blind is fully up (k = 0). This is intentional:
   tilt is meaningless there.
4. `FireflyOfficeEnv.dgp_known` is computed but never used.
5. **Time-limit semantics:** the end of the horizon is reported as `terminated`, not
   `truncated`. For bootstrapped RL over a continuing year, you may want `truncated=True` instead.
   Ask before changing this, because it changes learning targets.
6. **Electric light is not included** in seat eye illuminance or glare. Also missing: no thermal
   model (solar gains/HVAC), occupancy is always on 8–18, a single TMY year, and the tilt BSDFs
   come from a calibrated generic slat model, not manufacturer data.
7. The upstream READMEs in `Implementation_Radiance/` **swap some spectra descriptions**
   (e.g. `Fluorescent_840.csv` is labelled as CIE CMFs). Trust the filenames and step 01, not
   the README text.
8. `radiance_strips/mtx/` is flat in the repo, but the scripts write and read `mtx/strips/`.
9. `Claude outputs/` and `Implementation_Radiance.zip` are leftovers and duplicates. Do not use
   them.

---

## 10. Rules for agents working here

- **Never modify anything under `Implementation_Radiance/`.** It is the published reference
  dataset. Write derived files under `firefly_gym/` or a scratch dir.
- **Never `cat`, print or load whole huge files into context.** That includes `.mtx` (up to
  740 MB), `.hsr`, `.zip`, `.npy` and `.xml` BSDFs. Use `radiance_io` and print shapes or
  summaries. The sky matrices alone are 740 MB each in float32.
- **Do not put data into version control.** That covers `*.npz`, `*.npy`, `*.mtx`, `*.hsr`,
  `*.zip`, the EPW and `env_v2_data/`. If you initialise git, add a `.gitignore` for these first.
- **Keep the CMDP structure:** constraint signals stay in `info["costs"]` and stay separate from
  the reward. Safe-RL algorithms (Lagrangian PPO, safety filters/shields, CBF-style projections)
  depend on that.
- **After any change to physics, data building or reward/cost code**, run
  `validate_against_radiance.py` and/or `validate_v2.py` and confirm the §8 numbers. Report any
  deviation, don't hide it.
- Keep the envs fast. No per-step file I/O, and vectorise with NumPy.
- New shading states are added by **new BSDFs only**: spectralise with `rmtxop -ff -c …` as in
  step 01, then pass `--state name=path` (v0) or add the state to `STATES` in `build_v2_data.py`
  (v1). Order states from most open to most closed so continuous actions are monotone.
- Units: illuminance in lux, irradiance in W/m², energy in kWh, power in W, angles in degrees.
  Workplane target is 500 lx (EN 12464-1), and the dimming target is 625 lx (500 / 0.8
  maintenance factor).
- Ask before large refactors, renaming gym ids, or changing reward weights or default thresholds.
  Experiments and papers depend on them.

---

## 11. Likely next tasks (context, not a to-do list)

- Train and evaluate safe-RL agents on v1 (energy vs glare/illuminance constraints) against the
  rule-based and "fully down at 30°" baselines.
- Add `requirements.txt`, a `.gitignore`, tests that assert the §8 numbers, and a vectorised
  batch env.
- Rebuild v1 data on LEO5 with higher `-ad` / `-c`, and add manufacturer slat BSDFs.
- Use the hall model (`hall_with_results`: skylit hall, Stuttgart, 64 × 54 W LEDs, one skylight
  BSDF, no shading control) as a second, lighting-only environment.
- Couple to a thermal model (solar gains) for joint HVAC + shading control.
