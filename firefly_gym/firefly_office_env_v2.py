"""FireflyOffice-v1: venetian blind HEIGHT + SLAT TILT + dimmable lighting.

The window is split into N horizontal strips (default 5, strip 1 = top). Each strip has its
own Radiance view matrix V_z and daylight matrix D_z, so a partially lowered blind is exact
superposition in the three-phase method:

    E(t) = sum_z  V_z · T_{state(z)} · D_z · s(t)

A blind lowered to height level k (0..N) covers the top k strips with the chosen slat tilt;
the strips below stay open (glazing only).

Action (continuous, default): Box(3) in [0,1]
    a[0] blind height  -> k = round(a[0]*N) strips covered (0 = retracted, N = fully down)
    a[1] slat tilt     -> nearest of the tilt set (0, 15, 30, 45, 60, 75 deg; 75 = closed)
    a[2] dimming       -> fraction of 4 x 98 W
Action (multidiscrete): MultiDiscrete([N+1, n_tilts, n_dim_levels])

Glare: simplified DGP (Wienold 2009) from vertical eye illuminance at the three seats,
DGPs = 6.22e-5 * Ev + 0.184 (checked against the repo's 12,045 evalglare results: 98.4 %
agreement on DGP > 0.40).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RewardConfig:
    w_energy: float = 1.0
    w_visual: float = 2.0
    w_glare: float = 2.0
    w_move: float = 0.02          # per strip of height change + per 15 deg of tilt change
    target_lux: float = 500.0
    glare_dgp: float = 0.40


class FireflyOfficeHeightTiltEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, data_dir: str = "env_v2_data", episode_days: int = 1,
                 day_range: tuple[int, int] = (1, 365), action_type: str = "continuous",
                 n_dim_levels: int = 11, closed_state: str = "model",
                 reward: RewardConfig | None = None, render_mode: str | None = None):
        m = np.load(os.path.join(data_dir, "meta.npz"))
        self.N = int(m["n_strips"])
        names = [str(s) for s in m["state_names"]]
        self.tilts = [int(t) for t in m["tilts_deg"]]
        tilt_names = [f"t{t:02d}" for t in self.tilts]
        if closed_state == "measured":        # use the measured 75-deg BSDF for the closed position
            tilt_names[-1] = "m75"
        self.state_idx = {n: i for i, n in enumerate(names)}
        self.open_i = self.state_idx["open"]
        self.tilt_i = [self.state_idx[n] for n in tilt_names]
        used = sorted({self.open_i, *self.tilt_i})
        sc = float(m["wp_scale"])
        # WP[z, state, t, p] (float16 on disk, units of 10 lx) -> keep only used states
        self.WP = {}
        for z in range(self.N):
            for i in used:
                self.WP[z, i] = np.load(os.path.join(data_dir, "wp", f"s{z+1}_{names[i]}.npy"))
        self.wp_scale = sc
        self.ceil = m["ceil_day"]; self.ev = m["ev_day"]
        self.facade = m["facade_irr"]; self.E_wp_el = m["E_wp_elec100"]
        self.E_ceil_el = float(m["E_ceil_elec100"]); self.P_max = float(m["P_lights_max_W"])
        self.doy = m["day_of_year"]; self.hour = m["hour"]
        self.T_total = len(self.facade)

        self.episode_days, self.day_range = episode_days, day_range
        self.cfg = reward or RewardConfig()
        self.render_mode = render_mode
        self.action_type, self.n_dim_levels = action_type, n_dim_levels
        nT = len(self.tilts)
        if action_type == "continuous":
            self.action_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        elif action_type == "multidiscrete":
            self.action_space = spaces.MultiDiscrete([self.N + 1, nT, n_dim_levels])
        else:
            raise ValueError(action_type)
        lo = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], np.float32)
        hi = np.array([1, 1, 1, 1, 3, 3, 20, 1, 1, 1], np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    # -------------------------------------------------------------- physics
    def decode_action(self, a):
        if self.action_type == "continuous":
            a = np.clip(np.asarray(a, np.float64), 0, 1)
            return int(np.rint(a[0] * self.N)), int(np.rint(a[1] * (len(self.tilts) - 1))), float(a[2])
        k, j, d = (int(x) for x in a)
        return k, j, d / (self.n_dim_levels - 1)

    def strip_states(self, k, j):
        return [self.tilt_i[j] if z < k else self.open_i for z in range(self.N)]

    def daylight(self, k, j, t=None):
        """Daylight-only workplane (441), ceiling sensor and seat eye illuminance for height k, tilt j."""
        t = self.t if t is None else t
        st = self.strip_states(k, j)
        wp = sum(self.WP[z, s][t].astype(np.float32) for z, s in enumerate(st)) * self.wp_scale
        ceil = float(sum(self.ceil[z, s, t] for z, s in enumerate(st)))
        ev = sum(self.ev[z, s, :, t] for z, s in enumerate(st))
        return wp, ceil, ev

    def _obs(self):
        t = self.t
        h = (self.hour[t] - 0.5) / 24 * 2 * np.pi
        dy = (self.doy[t] - 1) / 365 * 2 * np.pi
        nxt = self.facade[t + 1] if (t + 1 < self.T_total and self.doy[t + 1] == self.doy[t]) else 0.0
        _, ceil, _ = self.daylight(self.prev[0], self.prev[1])
        ceil += self.prev[2] * self.E_ceil_el
        return np.array([np.sin(h), np.cos(h), np.sin(dy), np.cos(dy), self.facade[t] / 500,
                         nxt / 500, ceil / 1000, self.prev[0] / self.N,
                         self.prev[1] / (len(self.tilts) - 1), self.prev[2]], np.float32)

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        lo, hi = self.day_range
        d = options.get("start_day")
        if d is None:
            d = int(self.np_random.integers(lo, max(lo, hi - self.episode_days + 1) + 1))
        self.t = (d - 1) * 11
        self.t_end = min(self.t + 11 * self.episode_days, self.T_total)
        self.prev = (0, 0, 0.0)
        return self._obs(), {"day_of_year": int(self.doy[self.t]), "hour": int(self.hour[self.t])}

    def step(self, action):
        k, j, u = self.decode_action(action)
        c = self.cfg
        wp, ceil, ev = self.daylight(k, j)
        E_wp = wp + u * self.E_wp_el
        dgp = 6.22e-5 * ev + 0.184
        shortfall = float(np.mean(np.clip(1 - E_wp / c.target_lux, 0, 1)))
        glare = float(np.max(np.clip(dgp - c.glare_dgp, 0, None)))
        move = abs(k - self.prev[0]) + (abs(j - self.prev[1]) if k > 0 else 0)
        reward = -(c.w_energy * u + c.w_visual * shortfall + c.w_glare * glare / 0.1 + c.w_move * move)
        info = {"day_of_year": int(self.doy[self.t]), "hour": int(self.hour[self.t]),
                "height_level": k, "height_frac": k / self.N, "tilt_deg": self.tilts[j], "dim": u,
                "energy_kWh": u * self.P_max / 1000, "E_wp_mean": float(E_wp.mean()),
                "E_wp_min": float(E_wp.min()), "E_ceil": ceil + u * self.E_ceil_el,
                "Ev_seats": ev.astype(float).tolist(), "dgp": dgp.astype(float).tolist(),
                "facade_irr": float(self.facade[self.t]),
                "costs": {"illuminance": shortfall, "glare": glare, "glare_violation": float(glare > 0)}}
        self.prev = (k, j, u)
        self.t += 1
        term = self.t >= self.t_end
        obs = self._obs() if not term else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), term, False, info


def rule_based_action(env, facade_threshold=120.0, target_wp=625.0):
    """Repo strategy: fully lower + close (75 deg) above 120 W/m2, dim to the ceiling-sensor target."""
    e = env.unwrapped
    k = e.N if e.facade[e.t] > facade_threshold else 0
    j = len(e.tilts) - 1
    _, ceil, _ = e.daylight(k, j)
    tgi = target_wp * e.E_ceil_el / float(e.E_wp_el.mean())
    u = float(np.clip((tgi - ceil) / e.E_ceil_el, 0, 1))
    return np.array([k / e.N, 1.0, u], np.float32)


gym.register(id="FireflyOfficeHeightTilt-v1", entry_point=FireflyOfficeHeightTiltEnv)
