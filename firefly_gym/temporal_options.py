"""Time-extended blind settings with hourly electric-light execution.

The wrapped environment still advances in primitive daytime hours. A wrapper step
returns an undiscounted reward sum; SMDP learners must use discounted_reward and
discount from info instead of applying a fixed discount per manager decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class BlindOption:
    height: int
    tilt_index: int
    hours: int


def dimming_action(env, height: int, tilt_index: int, target_lux: float = 625.0):
    """Existing model-assisted ceiling-sensor law, evaluated for the chosen blinds.

    This reads candidate-state daylight from the simulator, not just the manager's
    observation. Use the same executor/model access across compared managers.
    """
    e = env.unwrapped
    _, ceiling, _ = e.daylight(height, tilt_index)
    target_ceiling = target_lux * e.E_ceil_el / float(e.E_wp_el.mean())
    dim = np.clip((target_ceiling - ceiling) / e.E_ceil_el, 0.0, 1.0)
    return np.array([height / e.N, tilt_index / (len(e.tilts) - 1), dim], np.float32)


class TemporalOptions(gym.Wrapper):
    """Select (blind height, tilt, duration); execute hourly feedback or a hold.

    All original height/tilt commands remain available, including tilt values
    while retracted: the base environment retains those in its action memory.
    Options end at day boundaries even in a multi-day base episode.
    """

    def __init__(self, env, durations=(1, 2, 3), gamma=0.99, executor="feedback"):
        super().__init__(env)
        if env.unwrapped.action_type != "continuous":
            raise ValueError("TemporalOptions requires a continuous-action base environment")
        durations = tuple(durations)
        if not durations or any(not isinstance(d, Integral) or d < 1 for d in durations):
            raise ValueError("durations must be positive integer numbers of hours")
        if len(set(durations)) != len(durations):
            raise ValueError("durations must not contain duplicates")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if executor not in ("feedback", "hold"):
            raise ValueError("executor must be feedback or hold")
        self.durations = tuple(int(d) for d in durations)
        self.gamma = float(gamma)
        self.executor = executor
        e = env.unwrapped
        self.options = tuple(BlindOption(k, j, d) for k in range(e.N + 1)
                             for j in range(len(e.tilts)) for d in self.durations)
        self.action_space = gym.spaces.Discrete(len(self.options))
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._needs_reset = False
        return obs, info

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("Call reset before starting another episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid option index: {action}")
        option = self.options[int(action)]
        e = self.env.unwrapped
        start_day = int(e.doy[e.t])
        held_action = dimming_action(e, option.height, option.tilt_index)
        rewards, records = [], []
        terminated = truncated = False
        reason = "duration"
        for elapsed in range(option.hours):
            primitive_action = (held_action if self.executor == "hold" or elapsed == 0 else
                                dimming_action(e, option.height, option.tilt_index))
            prev_k, prev_j, prev_u = e.prev
            obs, reward, terminated, truncated, info = self.env.step(primitive_action)
            # Count actual commands separately from manager decision frequency.
            movement = abs(option.height - prev_k)
            if option.height > 0:
                movement += abs(option.tilt_index - prev_j)
            info = dict(info)
            info["blind_movement_units"] = movement
            info["blind_moved"] = int(movement > 0)
            info["dimmer_change"] = abs(float(info["dim"]) - prev_u)
            rewards.append(float(reward))
            records.append(info)
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                self._needs_reset = True
                break
            if int(e.doy[e.t]) != start_day:
                reason = "day_boundary"
                break
        duration = len(rewards)
        weights = self.gamma ** np.arange(duration)
        cost_names = records[0]["costs"]
        info = {
            "option_index": int(action),
            "planned_duration": option.hours,
            "duration": duration,
            "end_reason": reason,
            "discount": self.gamma ** duration,
            "discounted_reward": float(np.dot(weights, rewards)),
            "costs": {key: sum(r["costs"][key] for r in records) for key in cost_names},
            "discounted_costs": {
                key: float(np.dot(weights, [r["costs"][key] for r in records]))
                for key in cost_names
            },
            "energy_kWh": sum(r["energy_kWh"] for r in records),
            "primitive_rewards": rewards,
            "primitive_infos": records,
        }
        return obs, sum(rewards), terminated, truncated, info


def week_split():
    """Deterministic season-spanning split: 5 train, 1 validation, 1 test weeks.

    These are seven-day blocks of the supplied weather year, not ISO weeks.
    The final incomplete week belongs to training. Adjacent weeks still come
    from the same weather year; this does not test another climate/year.
    """
    split = {"train": [], "validation": [], "test": []}
    for day in range(1, 366):
        slot = ((day - 1) // 7) % 7
        group = "train" if slot < 5 else "validation" if slot == 5 else "test"
        split[group].append(day)
    return split


class HourlyMetrics:
    """Aggregate primitive hours, never average costs over unequal options."""

    def __init__(self):
        self.hours = self.decisions = self.blind_moves = self.dimmer_changes = 0
        self.reward = self.energy = self.movement = self.dimmer_variation = 0.0
        self.costs = {"illuminance": 0.0, "glare": 0.0, "glare_violation": 0.0}
        self.durations = {}
        self.planned_durations = {}

    def add(self, info):
        self.decisions += 1
        duration = info["duration"]
        self.hours += duration
        self.reward += sum(info["primitive_rewards"])
        self.energy += info["energy_kWh"]
        self.durations[duration] = self.durations.get(duration, 0) + 1
        planned = info["planned_duration"]
        self.planned_durations[planned] = self.planned_durations.get(planned, 0) + 1
        for key in self.costs:
            self.costs[key] += info["costs"][key]
        for record in info["primitive_infos"]:
            self.blind_moves += record["blind_moved"]
            self.movement += record["blind_movement_units"]
            self.dimmer_variation += record["dimmer_change"]
            self.dimmer_changes += record["dimmer_change"] > 1e-6

    def summary(self):
        if not self.hours:
            raise ValueError("No primitive hours were recorded")
        return {
            "primitive_hours": self.hours,
            "manager_decisions": self.decisions,
            "mean_option_hours": self.hours / self.decisions,
            "energy_kWh": self.energy,
            "energy_kWh_per_hour": self.energy / self.hours,
            "reward_per_hour": self.reward / self.hours,
            "costs_per_hour": {key: value / self.hours for key, value in self.costs.items()},
            "any_seat_glare_hours_pct": 100 * self.costs["glare_violation"] / self.hours,
            "blind_moves": self.blind_moves,
            "blind_movement_units": self.movement,
            "dimmer_changes": self.dimmer_changes,
            "dimmer_total_variation": self.dimmer_variation,
            "realized_duration_counts": self.durations,
            "planned_duration_counts": self.planned_durations,
        }
