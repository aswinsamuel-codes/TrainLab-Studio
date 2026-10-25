"""Landing: touch down gently on a pad. At the top level the pad is a ship deck that bobs,
drifts and pitches in the waves, with gusty wind (the classic "land on a moving ship").

Body needs: legs_touching(world_geoms), crash_reason(), effort() (see DroneBody).
"""

from __future__ import annotations

import numpy as np

from trainlab.core.rewards import PotentialShaper
from trainlab.tasks.base import Task

HOLD_STEPS = 6  # stay settled on the pad this many control steps to count as landed


class LandingTask(Task):
    name = "landing"
    episode_seconds = 20.0
    start = (0.0, 0.0, 3.0)
    CURRICULUM = [
        {"dist": [1.5, 3.0], "height": [2.0, 3.0], "pad": 0.7, "bob": 0.0, "drift": 0.0, "pitch": 0.0, "wind": 0.0},
        {"dist": [3.0, 6.0], "height": [2.5, 5.0], "pad": 0.6, "bob": 0.0, "drift": 0.0, "pitch": 0.0, "wind": 1.0},
        {"dist": [4.0, 8.0], "height": [3.0, 6.0], "pad": 0.5, "bob": 0.15, "drift": 0.2, "pitch": 0.04,
         "wind": 2.0},
        {"dist": [5.0, 10.0], "height": [3.0, 7.0], "pad": 0.45, "bob": 0.3, "drift": 0.5, "pitch": 0.1,
         "wind": 4.0},
    ]

    # The deck is a real (heavy, gravity-compensated) body moved by stiff position motors, not
    # a teleported mocap: friction then carries whatever stands on it, like a real ship deck.
    def world_xml(self) -> str:
        return super().world_xml() + """
<body name="pad" pos="0 0 1.0" gravcomp="1">
  <joint name="pad_x" type="slide" axis="1 0 0" damping="3000"/>
  <joint name="pad_y" type="slide" axis="0 1 0" damping="3000"/>
  <joint name="pad_z" type="slide" axis="0 0 1" damping="3000"/>
  <joint name="pad_pitch" type="hinge" axis="0 1 0" damping="800"/>
  <geom name="pad" class="world" type="box" size="0.7 0.7 0.05" mass="200" material="mat_pad"/>
  <geom class="deco" type="box" size="0.04 0.25 0.002" pos="-0.17 0 0.051" material="mat_mark"/>
  <geom class="deco" type="box" size="0.04 0.25 0.002" pos="0.17 0 0.051" material="mat_mark"/>
  <geom class="deco" type="box" size="0.17 0.04 0.002" pos="0 0 0.051" material="mat_mark"/>
  <geom name="deck" class="deco" type="box" size="1.6 0.9 0.5" pos="0 0 -0.56" material="mat_dark"/>
</body>"""

    def actuators_xml(self) -> str:
        return ('<position name="pad_x" joint="pad_x" kp="400000"/>'
                '<position name="pad_y" joint="pad_y" kp="400000"/>'
                '<position name="pad_z" joint="pad_z" kp="400000"/>'
                '<position name="pad_pitch" joint="pad_pitch" kp="300000"/>')

    def bind(self, env) -> None:
        super().bind(env)
        m = env.model
        self.pad_body = env.body("pad")
        self.pad_geom = env.geom("pad")
        self.pad_geoms = {self.pad_geom}
        self.pad_acts = [env.actuator(f"pad_{k}") for k in ("x", "y", "z", "pitch")]
        self.pad_qpos = [m.jnt_qposadr[env.joint(f"pad_{k}")] for k in ("x", "y", "z", "pitch")]
        self.shaper = PotentialShaper()

    # ------------------------------------------------------------- the deck
    def deck_target(self, t: float) -> np.ndarray:
        """Where the deck motors drive to at time t: (x, y, z offset, pitch)."""
        p = self.p
        return np.array([p["drift"] * t * np.cos(self.drift_dir), p["drift"] * t * np.sin(self.drift_dir),
                         p["bob"] * np.sin(0.9 * t + self.phase), p["pitch"] * np.sin(0.7 * t + 2 * self.phase)])

    def pad_state(self):
        """Actual pad top-centre, velocity and pitch, read from the physics."""
        d = self.env.data
        R = d.xmat[self.pad_body].reshape(3, 3)
        top = d.xpos[self.pad_body] + R[:, 2] * 0.05
        pitch = float(np.arctan2(R[0, 2], R[2, 2]))
        return top, self.env.body_velocity(self.pad_body), pitch

    def substep(self, t: float) -> None:
        self.env.data.ctrl[self.pad_acts] = self.deck_target(t)

    def randomize(self, p: dict) -> None:
        rng = self.env.rng
        m, d = self.env.model, self.env.data
        self.p = p
        self.phase = float(rng.uniform(0, 2 * np.pi))
        self.drift_dir = float(rng.uniform(0, 2 * np.pi))
        half = np.array([p["pad"], p["pad"], 0.05])  # never larger than compiled, so collision bounds stay valid
        m.geom_size[self.pad_geom] = half
        m.geom_rbound[self.pad_geom] = float(np.linalg.norm(half))
        m.geom_aabb[self.pad_geom] = np.concatenate([np.zeros(3), half])
        d.qpos[self.pad_qpos] = self.deck_target(0.0)
        d.ctrl[self.pad_acts] = self.deck_target(0.0)
        ang, dist = rng.uniform(0, 2 * np.pi), p["dist"]  # ranges arrive already sampled
        start = np.array([dist * np.cos(ang), dist * np.sin(ang), 1.0 + p["height"]])
        self.body.place(start, yaw=float(rng.uniform(-np.pi, np.pi)), vel=rng.normal(scale=0.3, size=3))
        import mujoco
        mujoco.mj_forward(m, d)
        top, _, _ = self.pad_state()
        self.d0 = max(float(np.linalg.norm((start - top)[:2])), 1.0)
        self.h0 = max(float(start[2] - top[2]), 1.0)
        self.settled = 0
        self.prev_rel_vz = 0.0
        self.was_touching = False
        self.shaper.prev = None

    # --------------------------------------------------------------- obs
    def relative(self):
        top, pvel, pitch = self.pad_state()
        rel = top - self.body.pos
        rvel = self.body.vel - pvel
        return rel, rvel, pitch

    def obs(self) -> list[float]:
        R = self.body.rot
        rel, rvel, pitch = self.relative()
        pad_up = R.T @ np.array([np.sin(pitch), 0.0, np.cos(pitch)])
        on_pad = self.body.legs_touching(self.pad_geoms) / 4.0
        return [*(R.T @ rel / 5.0), *(R.T @ rvel / 3.0), *pad_up, on_pad, float(self.body.pos[2]) / 5.0]

    # ---------------------------------------------------------- termination
    def check_done(self):
        rel, rvel, _ = self.relative()
        on_pad = self.body.legs_touching(self.pad_geoms)
        on_ground = self.body.legs_touching() - on_pad
        touching = on_pad > 0
        if touching and not self.was_touching and self.prev_rel_vz < -2.0:
            return True, False, "hard_landing"
        self.was_touching = touching
        self.prev_rel_vz = float(rvel[2])
        crash = self.body.crash_reason()
        if crash:
            return True, False, crash
        if on_ground > 0:
            return True, False, "missed_pad"
        if np.linalg.norm(rel[:2]) > 3 * self.d0 + 10 or self.body.pos[2] > 25:
            return True, False, "flew_away"
        if on_pad >= 3 and np.linalg.norm(rvel) < 0.4 and self.body.up_z > 0.9:
            self.settled += 1
            if self.settled >= HOLD_STEPS:
                return True, True, "landed"
        else:
            self.settled = 0
        return False, False, ""

    def potential(self) -> float:
        rel, rvel, _ = self.relative()
        d_xy = float(np.linalg.norm(rel[:2]))
        h = max(float(-rel[2]), 0.0)
        speed_ok = 0.5 + 0.8 * float(np.linalg.norm(rel))  # slow down as you get close
        excess = max(0.0, float(np.linalg.norm(rvel)) - speed_ok)
        return (-30.0 * min(d_xy / self.d0, 1.5) - 15.0 * min(h / self.h0, 1.5) - 4.0 * excess
                - 5.0 * (1.0 - self.body.up_z))

    def compute_reward(self, action, terminated, success):
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        terms = {"shaping": self.shaper(self.potential()), "time": -0.02, "energy": -0.01 * self.body.effort()}
        if success:
            terms["landed"] = 100.0
        return float(sum(terms.values())), terms
