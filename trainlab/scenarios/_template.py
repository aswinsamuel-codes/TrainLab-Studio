"""__TITLE__: a ball-pusher starter scenario. Replace it with your own idea.

A box robot slides on the floor (two force actuators) and must push a ball into
the green goal. Edit model.xml for the world and this file for obs/reward/done.
"""

from __future__ import annotations

import numpy as np

from trainlab.core import MujocoScenarioEnv, PotentialShaper, register_scenario


@register_scenario("__NAME__")
class __CLASS__(MujocoScenarioEnv):
    xml_file = "model.xml"
    episode_seconds = 20.0
    wind_body = None

    CURRICULUM = [
        {"ball_dist": [1.5, 2.5], "goal_dist": [2.0, 3.0]},
        {"ball_dist": [2.0, 4.0], "goal_dist": [3.0, 5.0]},
        {"ball_dist": [3.0, 6.0], "goal_dist": [4.0, 8.0]},
    ]

    def setup(self):
        self.robot = self.body("robot")
        self.ball = self.body("ball")
        self.goal_geom = self.geom("goal")
        self.robot_q = self.model.jnt_qposadr[self.joint("robot_x")]
        self.ball_q = self.model.jnt_qposadr[self.joint("ball_free")]
        self.shaper = PotentialShaper()

    def randomize(self, p):
        rng = self.rng
        a = rng.uniform(0, 2 * np.pi)
        ball = np.array([np.cos(a), np.sin(a)]) * p["ball_dist"]
        b = a + rng.uniform(-0.8, 0.8)
        self.goal = ball + np.array([np.cos(b), np.sin(b)]) * p["goal_dist"]
        self.data.qpos[self.ball_q:self.ball_q + 3] = [ball[0], ball[1], 0.25]
        self.model.geom_pos[self.goal_geom][:2] = self.goal
        self.shaper.prev = None

    def positions(self):
        return self.data.xpos[self.robot][:2].copy(), self.data.xpos[self.ball][:2].copy()

    def get_obs(self):
        r, b = self.positions()
        rv = self.body_velocity(self.robot)[:2]
        bv = self.body_velocity(self.ball)[:2]
        return np.concatenate([(b - r) / 5, (self.goal - b) / 5, rv / 3, bv / 3]).astype(np.float32)

    def check_done(self):
        r, b = self.positions()
        if np.linalg.norm(b - self.goal) < 0.5:
            return True, True, "goal"
        if np.abs(b).max() > 15 or np.abs(r).max() > 15:
            return True, False, "out_of_bounds"
        return False, False, ""

    def compute_reward(self, action, terminated, success):
        r, b = self.positions()
        phi = -50 * np.linalg.norm(b - self.goal) / 5 - 10 * np.linalg.norm(b - r) / 5
        terms = {"shaping": self.shaper(float(phi)), "effort": -0.01 * float(np.square(action).sum()),
                 "time": -0.02}
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        if success:
            terms["goal"] = 100.0
        return float(sum(terms.values())), terms
