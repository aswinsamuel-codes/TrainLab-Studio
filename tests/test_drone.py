import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import trainlab
from trainlab.core.registry import get_scenario

DRONE = ["drone_waypoints", "drone_landing"]


def goto(d, target, target_vel=np.zeros(3)):
    """Tiny hand-written autopilot (proves the tasks are winnable)."""
    R = d.rot
    rel = R.T @ (target - d.pos)
    v = R.T @ (d.vel - target_vel)
    ex = np.clip(0.6 * rel - 0.9 * v, -1, 1)
    return np.array([np.clip(1.2 * rel[2] - 1.0 * v[2], -0.6, 0.8), np.clip(-ex[1], -1, 1), np.clip(ex[0], -1, 1), 0.0])


def test_registered():
    names = trainlab.list_scenarios()
    for n in DRONE:
        assert n in names


@pytest.mark.parametrize("name", DRONE)
def test_check_env(name):
    env = get_scenario(name)()
    check_env(env, skip_render_check=True)
    env.close()


@pytest.mark.parametrize("name", DRONE)
def test_every_level_runs(name):
    env = trainlab.make(name)
    u = env.unwrapped
    for level in range(len(u.CURRICULUM)):
        u.set_level(level)
        env.reset(seed=level)
        for _ in range(25):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)) and np.isfinite(r)
            if term or trunc:
                env.reset()
    env.close()


def test_zero_action_is_hover_thrust():
    env = trainlab.make("drone_landing")
    env.reset(seed=0)
    d = env.unwrapped.body_obj
    env.step(np.zeros(4))
    assert abs(d.thrusts.sum() - d.mass * 9.81) < 0.2


def test_pitch_flies_forward():
    env = trainlab.make("drone_landing")
    env.reset(seed=0)
    d = env.unwrapped.body_obj
    p0, fwd = d.pos, d.rot[:, 0]
    for _ in range(20):
        env.step(np.array([0.0, 0.0, 1.0, 0.0]))
    assert (d.pos - p0) @ fwd > 1.0


def test_waypoints_winnable():
    env = trainlab.make("drone_waypoints", level=2)
    env.reset(seed=0)
    d, t = env.unwrapped.body_obj, env.unwrapped.task
    for _ in range(500):
        _, _, term, trunc, info = env.step(goto(d, t.target))
        if term or trunc:
            break
    assert info["is_success"], info["reason"]


def test_lands_on_moving_ship_deck():
    env = trainlab.make("drone_landing", level=3)
    env.reset(seed=1)
    d, t = env.unwrapped.body_obj, env.unwrapped.task
    for _ in range(400):
        rel, rvel, _ = t.relative()
        close = np.linalg.norm(rel[:2]) < 0.25
        a = goto(d, d.pos + rel + np.array([0, 0, 0.6 if not close else -0.05]), d.vel - rvel)
        a[0] = max(a[0], -0.35)
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert info["is_success"], info["reason"]


def test_crash_into_ground_fails():
    env = trainlab.make("drone_landing", level=1)
    env.reset(seed=0)
    for _ in range(200):
        _, r, term, trunc, info = env.step(np.array([-1.0, 1.0, 0.0, 0.0]))  # dive and bank hard
        if term or trunc:
            break
    assert term and not info["is_success"] and r == -100.0


def test_render_with_effects():
    env = trainlab.make("drone_waypoints", render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.array([0.4, 0, 0, 0]))
    frame = env.render()
    assert frame.std() > 1
    env.close()
