"""Turn the Radiance three-phase results of the FIREFLY office model into a compact
data file for the Gymnasium environment.

Three-phase method (per spectral channel c, per daytime hour t):
    E(t) = V · T_state · (D · s(t))
      V      : view matrix, sensors <- 145 Klems window patches     (wp_sensor_kf.mtx, ceil_sensor_kf.vec, seat images)
      T_state: window BSDF incl. glazing spectrum, 145 x 145        (noblinds.mtx, blinds+75.mtx, ... any new state)
      D·s(t) : daylight matrix x spectral sky vector, 145 x 4015    (Innsbruck_daytime_kf.mtx)

Everything except T is independent of the blind position, so adding more slat angles only
needs new BSDF XMLs (spectralised exactly like _office_step01.sh, step 1b).

Usage
-----
    python build_office_env_data.py /path/to/Implementation_Radiance/office_with_results \
        --out office_env_data.npz
    # extra shading states (create them first with rmtxop, see README):
    python build_office_env_data.py ROOT --state slat45=mtx/blinds+45.mtx
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from radiance_io import PHOTOPIC_W, read_hsr, read_rmtx

N_HOURS = 4015          # 365 days x 11 daytime hours (EPW hours 8..18)
HOURS = np.arange(8, 19)
SEATS = ("seat1", "seat2", "seat3")


def photopic_product(Vw: np.ndarray, T: np.ndarray, DS: np.ndarray) -> np.ndarray:
    """sum_c w_c * (V_c @ T_c @ DS_c)  ->  (n_sensors, n_hours)."""
    out = np.zeros((Vw.shape[0], DS.shape[1]), np.float64)
    for c in range(DS.shape[2]):
        if PHOTOPIC_W[c] == 0:
            continue
        out += PHOTOPIC_W[c] * ((Vw[:, :, c] @ T[:, :, c]) @ DS[:, :, c])
    return out


def seat_eye_vector(seat_dir: str) -> np.ndarray | None:
    """Vertical eye illuminance coefficients (145 x 20) from the 145 -vth fisheye component images.
    For a hemispherical (-vth) projection pixel area is proportional to projected solid angle,
    so Ev = pi * mean radiance over the image disk.  Validated to ~1.5% against evalglare."""
    files = [os.path.join(seat_dir, f"comp{k:03d}.hsr") for k in range(145)]
    if not all(os.path.exists(f) for f in files):
        return None
    first = read_hsr(files[0])
    H, W = first.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    disk = np.hypot((xx + 0.5) / (W / 2) - 1, (yy + 0.5) / (H / 2) - 1) <= 1
    ev = np.empty((145, first.shape[2]))
    for k, f in enumerate(files):
        img = first if k == 0 else read_hsr(f)
        ev[k] = np.pi * img[disk].mean(0)
    return ev


def read_evalglare(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    vals = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("dgp"):
                vals.append(float(line.split(":")[1].split()[0]))
    a = np.array(vals)
    return a if len(a) == N_HOURS else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="path to Implementation_Radiance/office_with_results")
    ap.add_argument("--out", default="office_env_data.npz")
    ap.add_argument("--state", action="append", default=[],
                    help="extra shading state name=relative/path/to/T.mtx (spectral 145x145x20)")
    args = ap.parse_args()
    R = lambda *p: os.path.join(args.root, *p)
    t0 = time.time()

    states = [("open", "mtx/noblinds.mtx", "noblinds"), ("closed75", "mtx/blinds+75.mtx", "blinds")]
    for s in args.state:
        name, path = s.split("=", 1)
        states.append((name, path, None))

    print("reading D*s(t) (145 x 4015 x 20) ...")
    DS = read_rmtx(R("mtx", "Innsbruck_daytime_kf.mtx")).astype(np.float64)
    V_wp = read_rmtx(R("mtx", "wp_sensor_kf.mtx")).astype(np.float64)       # 441 x 145 x 20
    V_ceil = read_rmtx(R("mtx", "ceil_sensor_kf.vec")).astype(np.float64)   # 1 x 145 x 20

    seat_vecs = {}
    for s in SEATS:
        ev = seat_eye_vector(R("mtx", s))
        seat_vecs[s] = ev
        print(f"  {s}: eye-illuminance vector {'OK' if ev is not None else 'MISSING (comp*.hsr not found)'}")

    S = len(states)
    E_wp = np.zeros((S, N_HOURS, 441), np.float32)
    E_ceil = np.zeros((S, N_HOURS), np.float32)
    Ev = np.full((S, 3, N_HOURS), np.nan, np.float32)
    dgp = np.full((S, 3, N_HOURS), np.nan, np.float32)

    for i, (name, path, evg_tag) in enumerate(states):
        T = read_rmtx(R(path)).astype(np.float64)
        E_wp[i] = photopic_product(V_wp, T, DS).T
        E_ceil[i] = photopic_product(V_ceil, T, DS)[0]
        for j, s in enumerate(SEATS):
            if seat_vecs[s] is not None:
                Ev[i, j] = photopic_product(seat_vecs[s][None], T, DS)[0]
            if evg_tag:
                g = read_evalglare(R("results", "anim", s, f"{evg_tag}_evg.txt"))
                if g is not None:
                    dgp[i, j] = g
        print(f"  state {name:10s}: mean workplane illuminance {E_wp[i].mean():8.1f} lx")

    # --- sanity check against Radiance's own rmtxop output --------------------------------
    for i, tag in [(0, "noblinds"), (1, "blinds")]:
        ref_p = R("mtx", f"Innsbruck_day_{tag}_wp_ill.mtx")
        if os.path.exists(ref_p):
            ref = read_rmtx(ref_p)[..., 0].T
            rel = np.abs(E_wp[i] - ref) / np.maximum(ref, 1.0)
            print(f"  check vs {os.path.basename(ref_p)}: max rel. error {rel.max():.2e}")

    facade = np.loadtxt(R("mtx", "Innsbruck_daytime_facade_ext_irrad.txt")).astype(np.float32)
    E_wp_el = read_rmtx(R("mtx", "electric_wp_ill.mtx"))[:, 0, 0]
    E_ceil_el = float(np.loadtxt(R("mtx", "electric_ceil_sensor_ill.txt")))
    wp_pts = np.loadtxt(R("pts", "workplane.pts"))

    np.savez_compressed(
        args.out,
        state_names=np.array([s[0] for s in states]),
        E_wp_day=E_wp, E_ceil_day=E_ceil, Ev_seat_day=Ev, dgp_evalglare=dgp,
        facade_irr=facade, E_wp_elec100=E_wp_el.astype(np.float32), E_ceil_elec100=E_ceil_el,
        P_lights_max_W=4 * 98.0, day_of_year=np.repeat(np.arange(1, 366), 11),
        hour=np.tile(HOURS, 365), wp_xy=wp_pts[:, :2].astype(np.float32),
        seats=np.array(SEATS), photopic_w=PHOTOPIC_W,
    )
    print(f"wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
