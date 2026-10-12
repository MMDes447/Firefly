"""Full-year checks for FireflyOfficeHeightTilt-v1.

    python validate_v2.py env_v2_data
"""
import sys, time
import numpy as np
from firefly_office_env_v2 import FireflyOfficeHeightTiltEnv, rule_based_action

REPO = {"open": 92.07, "closed_m75": 588.90, "rule_m75": 173.90}


def run_year(env, policy):
    env.reset(options={"start_day": 1})
    kwh = hours = n = 0; glare = np.zeros(3); done = False
    while not done:
        _, _, done, _, info = env.step(policy(env))
        kwh += info["energy_kWh"]; hours += info["dim"] > 0
        glare += np.array(info["dgp"]) > 0.4; n += 1
    return kwh, hours, glare / n * 100, n


def fixed(k, j):
    def pol(e):
        _, ceil, _ = e.daylight(k, j)
        tgi = 625 * e.E_ceil_el / float(e.E_wp_el.mean())
        return np.array([k / e.N, j / (len(e.tilts) - 1), np.clip((tgi - ceil) / e.E_ceil_el, 0, 1)], np.float32)
    return pol


def main(d):
    em = FireflyOfficeHeightTiltEnv(d, episode_days=365, closed_state="measured")
    N = em.N
    print("== consistency with the original whole-window Radiance results (measured 75 deg BSDF)")
    for name, pol, ref in [("blinds up", fixed(0, 0), REPO["open"]), ("fully down, 75 deg", fixed(N, 5), REPO["closed_m75"]),
                           ("rule-based", rule_based_action, REPO["rule_m75"])]:
        t0 = time.time(); kwh, hrs, g, n = run_year(em, pol)
        print(f"  {name:20s} {kwh:7.1f} kWh (original {ref:6.1f})  DGPs>0.4 seats {np.round(g,1)} %   {n/(time.time()-t0):,.0f} steps/s")
    e = FireflyOfficeHeightTiltEnv(d, episode_days=365)
    print("== new action space (model BSDFs): annual lighting energy [kWh] / worst-seat glare hours [%]")
    print("   height " + "".join(f"{t:>14d}deg" for t in e.tilts))
    for k in range(N + 1):
        row = []
        for j in range(len(e.tilts)):
            kwh, _, g, _ = run_year(e, fixed(k, j))
            row.append(f"{kwh:8.0f}/{g.max():5.1f}%")
            if k == 0:
                row = [row[0]] + ["       (same)"] * (len(e.tilts) - 1); break
        print(f"   {k/N:5.0%}  " + " ".join(f"{r:>16s}" for r in row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "env_v2_data")
