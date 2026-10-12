"""Build env data for blind HEIGHT x TILT control from per-strip three-phase matrices.

E(t) = sum_z  V_z · T_{state(z)} · D_z · s(t)       z = window strip (1 = top)
A blind lowered to height h covers the top h strips with the slat state; lower strips are open.

Outputs (in OUT dir):
  meta.npz                 small arrays (ceiling sensor, seat eye illuminance, facade, electric, time)
  wp/s{z}_{state}.npy      workplane illuminance per strip & window state, float16 in units of 10 lx
"""
import os, sys, numpy as np
_here = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_here, os.path.join(_here, '..')]
from radiance_io import read_rmtx, PHOTOPIC_W

N = 5
O = (sys.argv[1] if len(sys.argv) > 1 else '../office_with_results').rstrip('/') + '/'   # for evalglare, facade, electric results
OUT = sys.argv[2] if len(sys.argv) > 2 else 'env_v2_data'
os.makedirs(f'{OUT}/wp', exist_ok=True)

STATES = [('open', 'mtx/noblinds.mtx')] + [(f't{a:02d}', f'mtx/tilt{a:02d}.mtx') for a in (0, 15, 30, 45, 60, 75)] \
         + [('m75', 'mtx/blinds+75.mtx')]
T = {n: read_rmtx(p).astype(np.float64) for n, p in STATES}
S = None if os.environ.get('APPROX_D') else read_rmtx('mtx/Innsbruck_daytime_sky4.mtx')                     # 2306 x 4015 x 20 (float32)

ceil = np.zeros((N, len(STATES), 4015), np.float32)
ev = np.zeros((N, len(STATES), 3, 4015), np.float32)
for z in range(1, N + 1):
    if os.environ.get('APPROX_D'):   # development shortcut: every strip sees the whole-window D
        DS = read_rmtx(O + 'mtx/Innsbruck_daytime_kf.mtx').astype(np.float64)
    else:
        D = read_rmtx(f'mtx/strips/daylight_strip{z}.mtx').astype(np.float64)     # 145 x 2306 x 20
        DS = np.stack([D[:, :, c] @ S[:, :, c].astype(np.float64) for c in range(20)], -1)  # 145 x 4015 x 20
    Vw = read_rmtx(f'mtx/strips/wp_strip{z}.mtx').astype(np.float64) * PHOTOPIC_W      # 441 x 145 x 20
    Vc = read_rmtx(f'mtx/strips/ceil_strip{z}.vec').astype(np.float64)[0] * PHOTOPIC_W
    Ve = read_rmtx(f'mtx/strips/seat_strip{z}.mtx').astype(np.float64) * PHOTOPIC_W    # 3 x 145 x 20
    for i, (n, _) in enumerate(STATES):
        TDS = np.einsum('kjc,jtc->ktc', T[n], DS)
        wp = np.einsum('pkc,ktc->tp', Vw, TDS)
        np.save(f'{OUT}/wp/s{z}_{n}.npy', (wp / 10.0).astype(np.float16))
        ceil[z - 1, i] = np.einsum('kc,ktc->t', Vc, TDS)
        ev[z - 1, i] = np.einsum('skc,ktc->st', Ve, TDS)
    print('strip', z, 'done', flush=True)

dgp = np.full((2, 3, 4015), np.nan, np.float32)      # evalglare: [open, measured 75] x seats (full-window states only)
for i, tag in enumerate(['noblinds', 'blinds']):
    for s in range(3):
        dgp[i, s] = [float(l.split(':')[1].split()[0]) for l in open(O + f'results/anim/seat{s+1}/{tag}_evg.txt')]

np.savez_compressed(
    f'{OUT}/meta.npz', n_strips=N, state_names=np.array([n for n, _ in STATES]),
    tilts_deg=np.array([0, 15, 30, 45, 60, 75]),
    strip_edges_z=np.linspace(20.915, 19.255, N + 1), ceil_day=ceil, ev_day=ev,
    dgp_evalglare_full=dgp,
    facade_irr=np.loadtxt(O + 'mtx/Innsbruck_daytime_facade_ext_irrad.txt').astype(np.float32),
    E_wp_elec100=read_rmtx(O + 'mtx/electric_wp_ill.mtx')[:, 0, 0].astype(np.float32),
    E_ceil_elec100=float(np.loadtxt(O + 'mtx/electric_ceil_sensor_ill.txt')),
    P_lights_max_W=392.0, day_of_year=np.repeat(np.arange(1, 366), 11), hour=np.tile(np.arange(8, 19), 365),
    wp_xy=np.loadtxt('pts/workplane.pts')[:, :2].astype(np.float32),
    seats=np.array(['seat1', 'seat2', 'seat3']), wp_scale=10.0)
print('done')
