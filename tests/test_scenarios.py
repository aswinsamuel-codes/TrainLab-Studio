import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import trainlab
from trainlab.core.registry import get_scenario

BUILTIN = ["roadster_gap_jump", "roadster_launch_sprint", "roadster_space_dock", "roadster_track_race"]


def test_builtins_registered():
    names = trainlab.list_scenarios()
    for name in BUILTIN:
        assert name in names


@pytest.mark.parametrize("name", BUILTIN)
def test_check_env(name):
    env = get_scenario(name)()
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("name", BUILTIN)
def test_seeded_reset_is_deterministic(name):
    env = trainlab.make(name)
    a, _ = env.reset(seed=123)
    b, _ = env.reset(seed=123)
    np.testing.assert_allclose(a, b)
    env.close()


@pytest.mark.parametrize("name", BUILTIN)
def test_every_level_runs_with_finite_rewards(name):
    env = trainlab.make(name)
    u = env.unwrapped
    for level in range(len(u.CURRICULUM)):
        u.set_level(level)
        env.reset(seed=level)
        for _ in range(30):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            assert isinstance(info["reward_terms"], dict)
            if term or trunc:
                env.reset()
    env.close()


def test_rgb_render():
    env = trainlab.make("roadster_gap_jump", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (480, 640, 3) and frame.std() > 1
    env.close()


def test_car_drives_and_jump_scenario_crashes_without_thrusters():
    """Full throttle, no thrusters, a 10 m+ gap: the car must end up in the pit."""
    env = trainlab.make("roadster_gap_jump", level=4)
    env.reset(seed=0)
    action = np.zeros(7)
    action[0] = 1.0
    for _ in range(600):
        _, r, term, trunc, info = env.step(action)
        if term or trunc:
            break
    assert term and not info["is_success"]
    assert info["reason"] in {"fell", "body_hit", "flipped", "hard_landing"}
    env.close()


def test_lift_thrusters_produce_lift():
    env = trainlab.make("roadster_gap_jump")
    env.reset(seed=0)
    u = env.unwrapped
    z0 = u.car_pos[2]
    action = np.zeros(7)
    action[3] = 1.0  # lift
    for _ in range(15):
        env.step(action)
    assert u.car_pos[2] > z0 + 0.5
    assert u.fuel_frac() < 1.0
    env.close()
