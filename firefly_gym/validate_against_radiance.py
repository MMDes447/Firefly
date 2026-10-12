"""Roll the env for a full year with the three strategies of the repo and compare the KPIs
with the numbers Radiance wrote to office_with_results/results/.

    python validate_against_radiance.py office_env_data.npz
"""
import sys
import time

import numpy as np

from firefly_office_env import FireflyOfficeEnv, rule_based_action

REPO = {  # office_with_results/results/*
    "always_open":   {"kWh": 92.0741913, "light_hours": 759},
    "always_closed": {"kWh": 588.903826, "light_hours": 3989},
    "rule_based":    {"kWh": 173.898418, "light_hours": 1657},
}


def run_year(env, policy):
    obs, _ = env.reset(options={"start_day": 1})
    kwh, hours, glare, done, n = 0.0, 0, np.zeros(3), False, 0
    while not done:
        a = policy(env)
        obs, r, done, _, info = env.step(a)
        kwh += info["energy_kWh"]
        hours += info["dim"] > 0
        glare += np.array(info["dgp"]) > 0.4
        n += 1
    return kwh, hours, glare / n, n


def main(path):
    env = FireflyOfficeEnv(path, episode_days=365)
    e = env
    tgi = lambda s: 625 * e.E_ceil_el / e.E_wp_el.mean()
    fixed = lambda s: (lambda env: np.array(
        [s, np.clip((tgi(s) - env.E_ceil_day[s, env.t]) / env.E_ceil_el, 0, 1)], np.float32))
    for name, pol in [("always_open", fixed(0)), ("always_closed", fixed(1)),
                      ("rule_based", rule_based_action)]:
        t0 = time.time()
        kwh, hrs, gl, n = run_year(env, pol)
        dt = time.time() - t0
        ref = REPO[name]
        print(f"{name:14s} energy {kwh:8.2f} kWh (Radiance {ref['kWh']:8.2f}) | "
              f"light-on hours {hrs:5d} (Radiance {ref['light_hours']:5d}) | "
              f"DGP>0.4 share seat1/2/3 {np.round(gl, 4)} | {n/dt:,.0f} steps/s")

    # annual daylight metrics straight from the lookup tables (steps 8)
    for s, nm in enumerate(env.state_names[:2]):
        E = env.E_wp_day[s]                          # 4015 x 441
        sda = np.mean((E > 300).sum(0) > 4015 / 2) * 100
        udi = np.mean(((E > 100) & (E < 3000)).sum(0) > 4015 / 2) * 100
        print(f"{nm:10s} sDA300/50% {sda:6.2f}%   UDI100-3000 {udi:6.2f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "office_env_data.npz")
