"""Accounting, boundary and execution tests; no Radiance rebuild required."""
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from firefly_gym.firefly_office_env_v2 import FireflyOfficeHeightTiltEnv, rule_based_action
from firefly_gym.temporal_options import HourlyMetrics, TemporalOptions, dimming_action, week_split
from firefly_gym.train_temporal import (
    QNetwork, ReplayBuffer, TrainConfig, greedy_action, load_policy, save_policy,
    smdp_target, train,
)


DATA = Path(__file__).parents[1] / "env_v2_data"


@pytest.fixture(scope="module")
def base():
    if not (DATA / "meta.npz").exists():
        pytest.skip("FIREFLY runtime data unavailable")
    env = FireflyOfficeHeightTiltEnv(str(DATA), episode_days=1)
    yield env
    env.close()


def option_id(env, height, tilt, hours):
    return next(i for i, o in enumerate(env.options)
                if (o.height, o.tilt_index, o.hours) == (height, tilt, hours))


def test_option_matches_manual_primitive_execution(base):
    wrapped = TemporalOptions(base, gamma=0.8)
    wrapped.reset(options={"start_day": 170})
    obs, total, term, trunc, info = wrapped.step(option_id(wrapped, 3, 2, 3))
    base.reset(options={"start_day": 170})
    rewards, records = [], []
    for _ in range(3):
        expected_obs, reward, _, _, record = base.step(dimming_action(base, 3, 2))
        rewards.append(reward)
        records.append(record)
    np.testing.assert_array_equal(obs, expected_obs)
    assert not term and not trunc
    assert total == pytest.approx(sum(rewards))
    assert info["discounted_reward"] == pytest.approx(sum(0.8**j * r for j, r in enumerate(rewards)))
    assert info["discount"] == pytest.approx(0.8**3)
    assert info["energy_kWh"] == pytest.approx(sum(r["energy_kWh"] for r in records))
    for key in records[0]["costs"]:
        assert info["costs"][key] == pytest.approx(sum(r["costs"][key] for r in records))
        assert info["discounted_costs"][key] == pytest.approx(
            sum(0.8**j * r["costs"][key] for j, r in enumerate(records)))


def test_daily_terminal_shortens_option_and_requires_reset(base):
    wrapped = TemporalOptions(base, durations=(3,), gamma=0.9)
    wrapped.reset(options={"start_day": 365})
    for _ in range(3):
        _, _, term, _, _ = wrapped.step(0)
        assert not term
    obs, _, term, trunc, info = wrapped.step(0)
    assert term and not trunc
    assert info["planned_duration"] == 3 and info["duration"] == 2
    assert info["discount"] == pytest.approx(0.9**2)
    assert info["end_reason"] == "terminated"
    np.testing.assert_array_equal(obs, np.zeros(10))
    with pytest.raises(gym.error.ResetNeeded):
        wrapped.step(0)


def test_option_cannot_cross_skipped_night_in_multiday_episode(base):
    original_days = base.episode_days
    try:
        base.episode_days = 2
        wrapped = TemporalOptions(base, durations=(3,))
        wrapped.reset(options={"start_day": 1})
        for _ in range(3):
            wrapped.step(0)
        _, _, term, trunc, info = wrapped.step(0)
        assert not term and not trunc
        assert info["duration"] == 2 and info["end_reason"] == "day_boundary"
        assert base.doy[base.t] == 2
        _, _, term, _, info = wrapped.step(0)
        assert not term and info["duration"] == 3
        assert {r["day_of_year"] for r in info["primitive_infos"]} == {2}
    finally:
        base.episode_days = original_days


def test_time_limit_shortens_option_without_fabricating_a_terminal(base):
    wrapped = TemporalOptions(gym.wrappers.TimeLimit(base, max_episode_steps=2), durations=(3,))
    wrapped.reset(options={"start_day": 1})
    obs, _, term, trunc, info = wrapped.step(0)
    assert trunc and not term
    assert info["duration"] == 2 and info["end_reason"] == "truncated"
    assert info["discount"] == pytest.approx(0.99**2)
    assert np.any(obs != 0)


def test_feedback_adjusts_lighting_but_hold_keeps_primitive_command(base):
    # Find three hours with a varying local dimmer response, independent of data order.
    selection = None
    for day in range(1, 366):
        base.reset(options={"start_day": day})
        dims = []
        for _ in range(3):
            a = dimming_action(base, base.N, 2)
            dims.append(float(a[2]))
            base.step(a)
        if np.ptp(dims) > 0.01:
            selection = day, dims
            break
    assert selection is not None
    day, expected = selection
    feedback = TemporalOptions(base, durations=(3,), executor="feedback")
    feedback.reset(options={"start_day": day})
    _, _, _, _, fb = feedback.step(option_id(feedback, base.N, 2, 3))
    hold = TemporalOptions(base, durations=(3,), executor="hold")
    hold.reset(options={"start_day": day})
    _, _, _, _, held = hold.step(option_id(hold, base.N, 2, 3))
    np.testing.assert_allclose([r["dim"] for r in fb["primitive_infos"]], expected)
    np.testing.assert_allclose([r["dim"] for r in held["primitive_infos"]], expected[0])


def test_hourly_rule_controller_reproduces_original(base):
    base.reset(options={"start_day": 170})
    expected = []
    for _ in range(11):
        _, reward, _, _, info = base.step(rule_based_action(base))
        expected.append((reward, info["energy_kWh"], info["costs"]))
    wrapped = TemporalOptions(base, durations=(1,))
    wrapped.reset(options={"start_day": 170})
    for reward, energy, costs in expected:
        height = base.N if base.facade[base.t] > 120 else 0
        _, actual_reward, _, _, info = wrapped.step(option_id(wrapped, height, 5, 1))
        assert actual_reward == pytest.approx(reward)
        assert info["energy_kWh"] == pytest.approx(energy)
        assert info["costs"] == pytest.approx(costs)


def test_metrics_weight_hours_and_separate_decisions_from_moves(base):
    wrapped = TemporalOptions(base, durations=(1, 3))
    wrapped.reset(options={"start_day": 100})
    stats = HourlyMetrics()
    records = []
    for hours in (1, 3):
        _, _, _, _, info = wrapped.step(option_id(wrapped, 5, 2, hours))
        stats.add(info)
        records.extend(info["primitive_infos"])
    summary = stats.summary()
    assert summary["primitive_hours"] == 4
    assert summary["manager_decisions"] == 2
    assert summary["blind_moves"] == 1
    assert summary["blind_movement_units"] == 7
    assert summary["mean_option_hours"] == 2
    assert summary["costs_per_hour"]["illuminance"] == pytest.approx(
        np.mean([r["costs"]["illuminance"] for r in records]))


@pytest.mark.parametrize("durations", [(), (0,), (1.5,), (1, 1), (-1,)])
def test_invalid_durations_rejected(base, durations):
    with pytest.raises(ValueError):
        TemporalOptions(base, durations=durations)


def test_invalid_actions_rejected(base):
    env = TemporalOptions(base)
    env.reset()
    for action in (-1, env.action_space.n, 1.5):
        with pytest.raises(ValueError):
            env.step(action)


def test_smdp_target_uses_actual_duration_and_terminal_mask():
    # The second transition is terminal. The third represents a truncation:
    # terminated remains zero, so its final observation retains a bootstrap.
    rewards = torch.tensor([1 + 0.9 * 2 + 0.9**2 * 3, -2.0, 4.0])
    discounts = torch.tensor([0.9**3, 0.9**2, 0.9])
    targets = smdp_target(rewards, discounts, torch.tensor([0., 1., 0.]),
                          torch.tensor([10., 100., 5.]))
    torch.testing.assert_close(targets, torch.tensor([12.52, -2., 8.5]))


def test_split_is_disjoint_complete_and_keeps_weeks_together():
    split = week_split()
    sets = [set(days) for days in split.values()]
    assert set.union(*sets) == set(range(1, 366))
    assert sum(map(len, sets)) == 365
    assert len(split["validation"]) == len(split["test"]) == 49
    for first in range(1, 366, 7):
        block = set(range(first, min(first + 7, 366)))
        assert sum(block <= group for group in sets) == 1


def test_replay_copies_observations_and_wraps():
    replay = ReplayBuffer(2, 2)
    obs = np.array([1., 2.], np.float32)
    replay.add(obs, 0, -1, obs, .9, False)
    obs[:] = 99
    np.testing.assert_array_equal(replay.obs[0], [1., 2.])
    replay.add(obs, 1, -2, obs, .81, True)
    replay.add(obs, 2, -3, obs, .729, False)
    assert replay.size == 2 and replay.position == 1
    assert set(replay.actions) == {1, 2}
    assert replay.sample(3, np.random.default_rng(0))[0].shape == (3, 2)


def test_checkpoint_round_trip(tmp_path):
    model = QNetwork(10, np.int64(108))
    path = tmp_path / "policy.pt"
    save_policy(path, model, variant="duration", seed=7,
                config=TrainConfig(), closed_state="model")
    loaded, metadata = load_policy(path)
    obs = np.linspace(-1, 1, 10, dtype=np.float32)
    torch.testing.assert_close(loaded(torch.from_numpy(obs)), model(torch.from_numpy(obs)))
    assert greedy_action(loaded, obs) == greedy_action(model, obs)
    assert metadata["variant"] == "duration" and metadata["seed"] == 7


def test_training_excludes_heldout_days_and_counts_primitive_budget(base):
    torch.set_num_threads(1)
    config = TrainConfig(primitive_steps=22, batch_size=2, warmup_hours=0,
                         target_every_hours=11, hidden=8)
    with pytest.raises(ValueError, match="training split"):
        train(base, "duration", 0, config, week_split()["test"])
    _, first = train(base, "hourly", 0, config, week_split()["train"])
    _, second = train(base, "duration", 0, config, week_split()["train"])
    assert first["primitive_hours"] == second["primitive_hours"] == 22
    assert first["sampled_training_days"] == second["sampled_training_days"]
    assert first["gradient_updates"] == second["gradient_updates"] == 5
    assert np.isfinite(second["mean_last_100_losses"])
