"""LaunchSprint: launch hard, cover the distance fast, stop exactly on the line.

Rewards speed (time cost) but punishes overshooting, so the agent has to learn
when to lift off the throttle and brake. Road grip is randomized (wet/dry).
"""

from __future__ import annotations

import numpy as np

from trainlab.core import MujocoScenarioEnv, PotentialShaper, register_scenario
from trainlab.core.rewards import braking_envelope
from trainlab.scenarios._roadster import RoadsterMixin, roadster_xml

WORLD = """
<geom name="floor" class="world" type="plane" size="500 500 0.1" pos="0 0 -0.02" material="mat_grid"/>
<geom name="road" class="world" type="box" size="160 6 0.5" pos="140 0 -0.5" material="mat_road"/>
<geom name="start_line" class="deco" type="box" size="0.3 6 0.01" pos="0 0 0.005" rgba="1 1 1 1"/>
<geom name="finish_line" class="deco" type="box" size="0.5 6 0.012" pos="200 0 0.006" rgba="0.1 0.9 0.3 1"/>
"""

FINISH_TOL = 1.5  # metres either side of the line


@register_scenario("roadster_launch_sprint")
class RoadsterLaunchSprint(RoadsterMixin, MujocoScenarioEnv):
    CONTROLS = ("throttle", "steer", "brake")
    episode_seconds = 25.0

    CURRICULUM = [
        {"distance": [60.0, 100.0], "grip": 1.1, "wind": 0.0},
        {"distance": [100.0, 160.0], "grip": [0.9, 1.2], "wind": 2.0},
        {"distance": [150.0, 200.0], "grip": [0.6, 1.2], "wind": 6.0},
    ]

    def build_xml(self) -> str:
        return roadster_xml(WORLD, name="roadster_launch_sprint", start=(0, 0, 0.7))

    def setup(self) -> None:
        super().setup()
        self.road_geom = self.geom("road")
        self.finish_geom = self.geom("finish_line")
        self.shaper = PotentialShaper()

    def randomize(self, p: dict) -> None:
        self.target = p["distance"]
        self.model.geom_pos[self.finish_geom] = [self.target, 0, 0.006]
        self.model.geom_friction[self.road_geom][0] = p["grip"]
        self.place_car([0.0, self.rng.uniform(-0.5, 0.5), 0.6], yaw=self.rng.uniform(-0.03, 0.03))
        self.shaper.prev = None

    def get_obs(self) -> np.ndarray:
        p = self.car_pos
        return np.array([(self.target - p[0]) / 100.0, p[1] / 6.0, *self.car_obs()[:14],
                         self.wheel_slip() / 5.0, self.params["grip"]], dtype=np.float32)

    def check_done(self):
        p = self.car_pos
        if abs(p[1]) > 6.0 or self.up_z < 0.5:
            return True, False, "off_road"
        if p[0] > self.target + 15.0:
            return True, False, "overshot"
        stopped = np.linalg.norm(self.car_vel) < 0.3 and abs(p[0] - self.target) < FINISH_TOL
        return (True, True, "stopped_on_line") if stopped else (False, False, "")

    def potential(self) -> float:
        p = self.car_pos
        remaining = self.target - p[0]
        excess = max(0.0, abs(self.car_vel[0]) - braking_envelope(remaining, decel=8.0 * self.params["grip"]))
        return -100.0 * abs(remaining) / self.target - 3.0 * excess - 5.0 * abs(p[1]) / 6.0

    def compute_reward(self, action, terminated, success):
        terms = {"shaping": self.shaper(self.potential()), "time": -0.1,
                 "slip": -0.02 * self.wheel_slip()}
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        if success:
            err = abs(self.car_pos[0] - self.target)
            terms["finish"] = 100.0 + 50.0 * (1.0 - err / FINISH_TOL) + 3.0 * max(0.0, 20.0 - self.elapsed)
        return float(sum(terms.values())), terms
