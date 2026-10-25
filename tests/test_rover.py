import numpy as np
from gymnasium.utils.env_checker import check_env

import trainlab
from trainlab.core.registry import get_scenario


def test_registered_and_checked():
    assert "rover_waypoints" in trainlab.list_scenarios()
    env = get_scenario("rover_waypoints")()
    check_env(env, skip_render_check=True)
    env.close()


def test_drives_and_turns():
    env = trainlab.make("rover_waypoints")
    env.reset(seed=0)
    r = env.unwrapped.body_obj
    p0, fwd = r.pos, r.rot[:, 0]
    for _ in range(30):
        env.step(np.array([1.0, 0.0]))
    assert (r.pos - p0) @ fwd > 3.0  # ~2.5 m/s top speed
    h0 = r.heading
    for _ in range(10):
        env.step(np.array([0.0, 1.0]))
    assert (r.heading - h0 + np.pi) % (2 * np.pi) - np.pi > 0.5  # turned left on the spot


def test_ground_waypoints_are_on_the_ground():
    env = trainlab.make("rover_waypoints", level=1)
    env.reset(seed=2)
    t = env.unwrapped.task
    assert all(abs(p[2] - env.unwrapped.body_obj.spawn_z) < 1e-6 for p in t.points)


def test_rocks_appear_on_hard_levels_and_lasers_see_them():
    env = trainlab.make("rover_waypoints", level=4)  # top level: up to 12 rocks
    seen = False
    for seed in range(5):
        env.reset(seed=seed)
        u = env.unwrapped
        b = u.body_obj
        rocks_up = sum(u.data.mocap_pos[u.model.body_mocapid[u.body(n)]][2] > 0 for n in u.task.rock_bodies)
        assert rocks_up >= 5
        for _ in range(60):  # drive straight; something should cross the laser fan
            obs, *_ , term, trunc, _ = env.step(np.array([0.6, 0.0]))
            if b.eyes.last.min() < 1.0:
                seen = True
            if term or trunc:
                break
    assert seen


def test_autopilot_wins_flat_levels():
    for level in (0, 1):
        env = trainlab.make("rover_waypoints", level=level)
        env.reset(seed=0)
        b, t = env.unwrapped.body_obj, env.unwrapped.task
        for _ in range(500):
            rel = b.rot.T @ (t.target - b.pos)
            err = np.arctan2(rel[1], rel[0])
            _, _, term, trunc, info = env.step(np.array([np.clip(0.8 * np.cos(err), -0.3, 1), np.clip(2 * err, -1, 1)]))
            if term or trunc:
                break
        assert info["is_success"], (level, info["reason"])


def test_driving_away_counts_as_lost():
    """Fleeing the route must fail (it used to be the cheapest way to avoid the rocks)."""
    env = trainlab.make("rover_waypoints", level=2)
    env.reset(seed=0)
    b, t = env.unwrapped.body_obj, env.unwrapped.task
    for _ in range(500):
        rel = b.rot.T @ (t.target - b.pos)
        away = np.arctan2(-rel[1], -rel[0])  # steer to face directly away from the target
        _, r, term, trunc, info = env.step(np.array([1.0, np.clip(2 * away, -1, 1)]))
        if term or trunc:
            break
    assert term and info["reason"] == "lost" and r == -100.0


def test_drone_waypoints_still_fly():
    env = trainlab.make("drone_waypoints", level=2)
    env.reset(seed=0)
    pts = env.unwrapped.task.points
    assert any(p[2] > 0.9 for p in pts)  # flying bodies keep airborne targets
