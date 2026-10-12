"""Online semi-Markov Double DQN pilot. Run from the repository root:

python -m firefly_gym.train_temporal --train-steps 22000 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import gymnasium
import numpy as np
import torch
from torch import nn

from .firefly_office_env_v2 import FireflyOfficeHeightTiltEnv
from .temporal_options import HourlyMetrics, TemporalOptions, week_split


VARIANTS = {
    "hourly": ((1,), "feedback"),
    "fixed2": ((2,), "feedback"),
    "fixed3": ((3,), "feedback"),
    "duration": ((1, 2, 3), "feedback"),
    "hold3": ((3,), "hold"),
}


@dataclass
class TrainConfig:
    primitive_steps: int = 22000
    gamma: float = 0.99
    hidden: int = 64
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_hours: int = 1100
    update_every_hours: int = 4
    target_every_hours: int = 1100
    epsilon_final: float = 0.10


class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=64):
        super().__init__()
        # Gym's Discrete.n can be a NumPy scalar; keep checkpoint metadata plain.
        obs_size, n_actions, hidden = int(obs_size), int(n_actions), int(hidden)
        self.layers = nn.Sequential(nn.Linear(obs_size, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, n_actions))

    def forward(self, obs):
        return self.layers(obs)


def smdp_target(rewards, discounts, terminated, next_values):
    """rewards already contain sum_j gamma**j * r_j; discounts are gamma**tau.

    A time-limit truncation keeps its bootstrap. A genuine terminal does not.
    The current FIREFLY base reports its daily horizon as a terminal.
    """
    return rewards + discounts * (1.0 - terminated) * next_values


class ReplayBuffer:
    def __init__(self, capacity, obs_size):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.obs = np.empty((capacity, obs_size), np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.discounts = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, np.float32)

    def add(self, obs, action, reward, next_obs, discount, terminated):
        i = self.position
        self.obs[i], self.next_obs[i] = obs, next_obs
        self.actions[i], self.rewards[i] = action, reward
        self.discounts[i], self.terminated[i] = discount, terminated
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, rng):
        idx = rng.integers(self.size, size=count)
        return tuple(torch.from_numpy(array[idx]) for array in
                     (self.obs, self.actions, self.rewards, self.next_obs,
                      self.discounts, self.terminated))


def greedy_action(model, observation):
    with torch.no_grad():
        return int(model(torch.as_tensor(observation)).argmax().item())


def train(base, variant, seed, config, training_days):
    if config.primitive_steps <= 0 or config.primitive_steps % 11:
        raise ValueError("primitive_steps must be a positive multiple of 11")
    if base.episode_days != 1:
        raise ValueError("The pilot trainer requires one-day base episodes")
    if not training_days or not set(training_days) <= set(week_split()["train"]):
        raise ValueError("Training must use only the declared training split")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # Day sampling is independent of how many actions/exploration draws a variant makes.
    day_rng = np.random.default_rng(seed + 10000)
    schedule = day_rng.choice(training_days, size=config.primitive_steps // 11).tolist()
    durations, executor = VARIANTS[variant]
    env = TemporalOptions(base, durations, config.gamma, executor)
    model = QNetwork(env.observation_space.shape[0], env.action_space.n, config.hidden)
    target = QNetwork(env.observation_space.shape[0], env.action_space.n, config.hidden)
    target.load_state_dict(model.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity, env.observation_space.shape[0])
    stats = HourlyMetrics()
    steps = updates = 0
    losses = []
    for episode, day in enumerate(schedule):
        obs, _ = env.reset(seed=seed if episode == 0 else None, options={"start_day": day})
        done = False
        while not done:
            epsilon = config.epsilon_final + (1 - config.epsilon_final) * max(
                0.0, 1.0 - steps / (0.8 * config.primitive_steps))
            action = (int(rng.integers(env.action_space.n)) if rng.random() < epsilon else
                      greedy_action(model, obs))
            next_obs, _, terminated, truncated, info = env.step(action)
            replay.add(obs, action, info["discounted_reward"], next_obs,
                       info["discount"], terminated)
            old_steps = steps
            steps += info["duration"]
            stats.add(info)
            # Match update opportunities in primitive hours, not macro decisions.
            first = (old_steps // config.update_every_hours + 1) * config.update_every_hours
            for update_hour in range(first, steps + 1, config.update_every_hours):
                if update_hour < config.warmup_hours or replay.size < config.batch_size:
                    continue
                ob, ac, re, no, di, te = replay.sample(config.batch_size, rng)
                with torch.no_grad():
                    next_actions = model(no).argmax(1, keepdim=True)
                    next_values = target(no).gather(1, next_actions).squeeze(1)
                    values = smdp_target(re, di, te, next_values)
                estimates = model(ob).gather(1, ac[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(estimates, values)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                losses.append(float(loss.item()))
                updates += 1
            if old_steps // config.target_every_hours != steps // config.target_every_hours:
                target.load_state_dict(model.state_dict())
            obs, done = next_obs, terminated or truncated
        if (episode + 1) % 100 == 0:
            print(f"  {variant}, seed {seed}: {steps}/{config.primitive_steps} hours", flush=True)
    model.eval()
    return model, {
        **stats.summary(),
        "gradient_updates": updates,
        "mean_last_100_losses": float(np.mean(losses[-100:])) if losses else None,
        "sampled_training_days": schedule,
    }


def evaluate(base, days, *, variant="hourly", model=None, baseline=None, gamma=0.99):
    durations, executor = VARIANTS[variant]
    env = TemporalOptions(base, durations, gamma, executor)
    index = {(o.height, o.tilt_index, o.hours): i for i, o in enumerate(env.options)}
    if baseline is not None and (model is not None or variant != "hourly"):
        raise ValueError("Reference baselines use the hourly executor without a learned model")
    stats = HourlyMetrics()
    for day in days:
        obs, _ = env.reset(options={"start_day": int(day)})
        done = False
        while not done:
            if baseline == "rule_based":
                # Retain the original rule's tilt command, including when retracted.
                k = base.N if base.facade[base.t] > 120 else 0
                action = index[k, len(base.tilts) - 1, 1]
            elif baseline == "fixed30":
                action = index[base.N, base.tilts.index(30), 1]
            elif baseline is not None:
                raise ValueError(baseline)
            elif model is not None:
                action = greedy_action(model, obs)
            else:
                raise ValueError("Provide a model or reference baseline")
            obs, _, terminated, truncated, info = env.step(action)
            stats.add(info)
            done = terminated or truncated
    return stats.summary()


def save_policy(path, model, *, variant, seed, config, closed_state):
    """Inference checkpoint; optimizer/replay state is deliberately not a resume file."""
    torch.save({"state_dict": model.state_dict(), "variant": variant, "seed": seed,
                "train_config": asdict(config), "closed_state": closed_state,
                "obs_size": model.layers[0].in_features,
                "n_actions": model.layers[-1].out_features}, path)


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainConfig(**checkpoint["train_config"])
    model = QNetwork(checkpoint["obs_size"], checkpoint["n_actions"], config.hidden)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "env_v2_data")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--train-steps", type=int, default=22000, help="Primitive hours PER policy/seed; multiple of 11")
    parser.add_argument("--warmup-hours", type=int, default=1100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=["hourly", "fixed3", "duration", "hold3"])
    parser.add_argument("--closed-state", choices=("model", "measured"), default="model")
    parser.add_argument("--evaluate-checkpoint", type=Path, help="Evaluate a saved policy without any training")
    args = parser.parse_args()
    config = TrainConfig(primitive_steps=args.train_steps, warmup_hours=args.warmup_hours)
    if not args.evaluate_checkpoint:
        if args.train_steps <= 0 or args.train_steps % 11:
            parser.error("--train-steps must be a positive multiple of 11")
        if args.warmup_hours < config.batch_size * 3 or args.warmup_hours >= args.train_steps:
            parser.error("--warmup-hours must be at least 192 and below --train-steps")
        if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must be distinct nonnegative integers")
        if len(set(args.variants)) != len(args.variants):
            parser.error("--variants must not contain duplicates")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = None
    if args.evaluate_checkpoint:
        model, checkpoint = load_policy(args.evaluate_checkpoint)
        config = TrainConfig(**checkpoint["train_config"])
        args.closed_state = checkpoint["closed_state"]
    base = FireflyOfficeHeightTiltEnv(str(args.data_dir), episode_days=1, closed_state=args.closed_state)
    output = args.out or Path(__file__).parent / "results" / (
        "temporal_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    split = week_split()
    report = {
        "purpose": "Initial online temporal-options pilot; not evidence of superiority or safety",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config), "closed_state": args.closed_state,
        "reward_config": asdict(base.cfg), "data_dir": str(args.data_dir.resolve()),
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__),
                     "gymnasium": gymnasium.__version__},
        "split": split, "runs": {}, "baselines": {},
        "observation": "Original 10 values, including perfect next-hour facade forecast",
        "executor": "Model-assisted ceiling-sensor dimming to 625 lx equivalent",
        "cost_learning": "Logged separately; no learned constraint enforcement",
        "boundary": "Daily terminal as reported by the original environment",
    }

    def write_report():
        (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    for name in ("rule_based", "fixed30"):
        report["baselines"][name] = {
            group: evaluate(base, split[group], baseline=name, gamma=config.gamma)
            for group in ("validation", "test")
        }
    write_report()
    if checkpoint is not None:
        variant = checkpoint["variant"]
        durations, executor = VARIANTS[variant]
        env = TemporalOptions(base, durations, gamma=config.gamma, executor=executor)
        if env.action_space.n != checkpoint["n_actions"]:
            raise ValueError("Checkpoint option count does not match this environment")
        report["evaluated_checkpoint"] = str(args.evaluate_checkpoint.resolve())
        report["runs"][f"{variant}_seed{checkpoint['seed']}"] = {
            group: evaluate(base, split[group], variant=variant, model=model, gamma=config.gamma)
            for group in ("validation", "test")
        }
        write_report()
    else:
        for seed in args.seeds:
            for variant in args.variants:
                print(f"Training {variant}, seed {seed}", flush=True)
                model, training = train(base, variant, seed, config, split["train"])
                result = {"training": training, "variant_settings": {
                    "durations": list(VARIANTS[variant][0]), "executor": VARIANTS[variant][1]},
                    **{group: evaluate(base, split[group], variant=variant, model=model, gamma=config.gamma)
                       for group in ("validation", "test")}}
                key = f"{variant}_seed{seed}"
                save_policy(output / f"{key}.pt", model, variant=variant, seed=seed,
                            config=config, closed_state=args.closed_state)
                report["runs"][key] = result
                write_report()
                test = result["test"]
                print(f"  held-out: {test['energy_kWh']:.2f} kWh / {test['primitive_hours']} hours; "
                      f"glare {test['any_seat_glare_hours_pct']:.2f}%; "
                      f"{test['manager_decisions']} manager decisions", flush=True)
    base.close()
    print(f"Saved {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
