"""Gymnasium environment for joint venetian-blind + dimmable-lighting control of the
FIREFLY reference office (Innsbruck), built on the hyperspectral Radiance three-phase results
in Implementation_Radiance/office_with_results.

One step = one daytime hour (EPW hours 8..18, i.e. 11 steps per day, 4015 per year).
Each step is a table lookup + a linear electric-light term, so it runs at >10^5 steps/s
while matching Radiance's rmtxop output to ~1e-7.

Physics per step t, shade state s, dimming level u in [0, 1]:
    E_wp  = E_wp_day[s, t] + u * E_wp_elec100          (441 workplane points, lux)
    E_ceil= E_ceil_day[s, t] + u * E_ceil_elec100      (ceiling photosensor, lux)
    P     = u * 392 W                                  (4 luminaires x 98 W, linear dimming)
    DGP_k = evalglare DGP from the repo if available for (s, seat k), else
            simplified DGP = 5.87e-5 * Ev_k + 0.16 from the seat's eye illuminance.

Safe-RL friendly: `info["costs"]` holds the constraint signals (under-illumination, glare)
separately from the energy reward, so the env can be used as a CMDP.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RewardConfig:
    w_energy: float = 1.0        # per step at full power (normalised: u)
    w_visual: float = 2.0        # mean relative shortfall below target on the workplane
    w_glare: float = 2.0         # max over seats of (DGP - 0.40)_+ / 0.1
    w_switch: float = 0.05       # per shade-state change
    target_lux: float = 500.0    # maintained illuminance (EN 12464-1 office)
    glare_dgp: float = 0.40      # "disturbing" threshold used in the repo (step 11)


class FireflyOfficeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        data_path: str = "office_env_data.npz",
        episode_days: int = 1,
        day_range: tuple[int, int] = (1, 365),
        action_type: str = "continuous",      # "continuous" | "discrete"
        n_dim_levels: int = 11,               # only for action_type="discrete"
        reward: RewardConfig | None = None,
        seats: tuple[int, ...] = (0, 1, 2),
        render_mode: str | None = None,
    ):
        d = np.load(data_path, allow_pickle=False)
        self.state_names = [str(s) for s in d["state_names"]]
        self.E_wp_day = d["E_wp_day"]            # (S, 4015, 441)
        self.E_ceil_day = d["E_ceil_day"]        # (S, 4015)
        self.Ev = d["Ev_seat_day"]               # (S, 3, 4015)  may contain NaN
        self.dgp_tab = d["dgp_evalglare"]        # (S, 3, 4015)  may contain NaN
        self.facade = d["facade_irr"]            # (4015,) W/m2, exterior vertical
        self.E_wp_el = d["E_wp_elec100"]         # (441,)
        self.E_ceil_el = float(d["E_ceil_elec100"])
        self.P_max = float(d["P_lights_max_W"])
        self.doy = d["day_of_year"]
        self.hour = d["hour"]
        self.S = len(self.state_names)
        self.seats = list(seats)

        # glare per (state, seat, hour): evalglare where available, else simplified DGP from Ev
        dgps = 5.87e-5 * np.nan_to_num(self.Ev, nan=0.0) + 0.16
        self.dgp = np.where(np.isnan(self.dgp_tab), dgps, self.dgp_tab).astype(np.float32)
        self.dgp_known = ~(np.isnan(self.dgp_tab) & np.isnan(self.Ev))

        self.episode_days = episode_days
        self.day_range = day_range
        self.cfg = reward or RewardConfig()
        self.render_mode = render_mode
        self.action_type = action_type
        self.n_dim_levels = n_dim_levels

        if action_type == "continuous":
            # a[0]: shade position in [0,1] -> nearest of S states; a[1]: dimming level in [0,1]
            self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        elif action_type == "discrete":
            self.action_space = spaces.Discrete(self.S * n_dim_levels)
        else:
            raise ValueError(action_type)

        # obs: sin/cos hour, sin/cos day, facade irr now, facade irr next hour (forecast),
        #      ceiling-sensor reading (daylight under previous shade + electric), prev shade, prev dim
        hi = np.array([1, 1, 1, 1, 3, 3, 20, 1, 1], dtype=np.float32)
        lo = np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0], dtype=np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def decode_action(self, action):
        if self.action_type == "continuous":
            a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
            s = int(np.rint(a[0] * (self.S - 1)))
            u = float(a[1])
        else:
            s, k = divmod(int(action), self.n_dim_levels)
            u = k / (self.n_dim_levels - 1)
        return s, u

    def daylight(self, s: int, t: int | None = None):
        """Daylight-only quantities for shade state s at hour index t (default: current)."""
        t = self.t if t is None else t
        return {"E_wp": self.E_wp_day[s, t], "E_ceil": float(self.E_ceil_day[s, t]),
                "dgp": self.dgp[s, self.seats, t]}

    def _obs(self):
        t = self.t
        h = (self.hour[t] - 0.5) / 24.0 * 2 * np.pi
        dy = (self.doy[t] - 1) / 365.0 * 2 * np.pi
        nxt = self.facade[t + 1] if (t + 1 < len(self.facade) and self.doy[t + 1] == self.doy[t]) else 0.0
        ceil = self.E_ceil_day[self.prev_s, t] + self.prev_u * self.E_ceil_el
        return np.array([np.sin(h), np.cos(h), np.sin(dy), np.cos(dy),
                         self.facade[t] / 500.0, nxt / 500.0, ceil / 1000.0,
                         self.prev_s / max(self.S - 1, 1), self.prev_u], dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        lo, hi = self.day_range
        start_day = options.get("start_day")
        if start_day is None:
            start_day = int(self.np_random.integers(lo, max(lo, hi - self.episode_days + 1) + 1))
        self.t0 = (start_day - 1) * 11
        self.t_end = min(self.t0 + 11 * self.episode_days, len(self.facade))
        self.t = self.t0
        self.prev_s, self.prev_u = 0, 0.0
        return self._obs(), {"day_of_year": int(self.doy[self.t]), "hour": int(self.hour[self.t])}

    def step(self, action):
        s, u = self.decode_action(action)
        t, c = self.t, self.cfg

        E_wp = self.E_wp_day[s, t] + u * self.E_wp_el
        E_ceil = float(self.E_ceil_day[s, t] + u * self.E_ceil_el)
        dgp = self.dgp[s, self.seats, t]
        energy_kwh = u * self.P_max / 1000.0

        shortfall = float(np.mean(np.clip(1.0 - E_wp / c.target_lux, 0.0, 1.0)))
        glare_excess = float(np.max(np.clip(dgp - c.glare_dgp, 0.0, None)))
        switched = float(s != self.prev_s)

        reward = -(c.w_energy * u + c.w_visual * shortfall
                   + c.w_glare * glare_excess / 0.1 + c.w_switch * switched)

        info = {
            "day_of_year": int(self.doy[t]), "hour": int(self.hour[t]),
            "shade_state": self.state_names[s], "dim": u,
            "energy_kWh": energy_kwh, "E_wp_mean": float(E_wp.mean()),
            "E_wp_min": float(E_wp.min()), "E_ceil": E_ceil,
            "frac_wp_below_target": float(np.mean(E_wp < c.target_lux)),
            "dgp": dgp.astype(float).tolist(), "facade_irr": float(self.facade[t]),
            "costs": {"illuminance": shortfall, "glare": glare_excess,
                      "glare_violation": float(glare_excess > 0)},
        }
        self.prev_s, self.prev_u = s, u
        self.t += 1
        terminated = self.t >= self.t_end
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            return f"t={self.t} day={self.doy[min(self.t, len(self.doy)-1)]} prev_shade={self.prev_s} prev_dim={self.prev_u:.2f}"


# ---------------------------------------------------------------------- baselines
def rule_based_action(env: FireflyOfficeEnv, facade_threshold: float = 120.0,
                      target_wp: float = 625.0):
    """The control strategy of the repo (steps 17-20): close blinds when exterior vertical
    irradiance > 120 W/m2; dim lights so that the ceiling sensor sees the fraction of 625 lx
    (500 lx / 0.8 maintenance factor) that corresponds to the workplane at full output."""
    e = env.unwrapped
    s = int(e.facade[e.t] > facade_threshold) if e.S == 2 else (e.S - 1 if e.facade[e.t] > facade_threshold else 0)
    tgi = target_wp * e.E_ceil_el / float(e.E_wp_el.mean())
    u = float(np.clip((tgi - e.E_ceil_day[s, e.t]) / e.E_ceil_el, 0.0, 1.0))
    return np.array([s / max(e.S - 1, 1), u], dtype=np.float32)


gym.register(id="FireflyOffice-v0", entry_point=FireflyOfficeEnv)
