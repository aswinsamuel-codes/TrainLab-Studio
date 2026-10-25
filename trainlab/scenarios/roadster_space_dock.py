"""SpaceDock (Starman mode): zero gravity, thrusters only. Fly the Roadster into
the station's docking port: slow, lined up, and without tumbling.

The closest 3D cousin of Lunar Lander: 6-DoF thrust, limited fuel, soft-contact bonus.
"""

from __future__ import annotations

import numpy as np

from trainlab.core import MujocoScenarioEnv, PotentialShaper, register_scenario
from trainlab.scenarios._roadster import RoadsterMixin, roadster_xml

WORLD = """
<geom name="station" class="world" type="box" size="3 6 6" pos="0 0 0" rgba="0.75 0.77 0.8 1"/>
<geom name="panel_l" class="deco" type="box" size="2 14 0.05" pos="0 20 0" rgba="0.1 0.2 0.5 1"/>
<geom name="panel_r" class="deco" type="box" size="2 14 0.05" pos="0 -20 0" rgba="0.1 0.2 0.5 1"/>
<geom name="port_ring" class="deco" type="cylinder" size="1.6 0.08" pos="-3.02 0 0" zaxis="1 0 0" rgba="1 0.75 0.1 1"/>
<geom name="port_core" class="deco" type="cylinder" size="1.2 0.09" pos="-3.03 0 0" zaxis="1 0 0" rgba="0.05 0.05 0.05 1"/>
<geom name="earth" class="deco" type="sphere" size="120" pos="150 -260 -220" rgba="0.15 0.35 0.8 1"/>
"""

DOCK_POINT = np.array([-3.0 - 2.4, 0.0, 0.0])  # car centre when its nose touches the port
DOCK_AXIS = np.array([1.0, 0.0, 0.0])  # car must point this way


def random_quat(rng: np.random.Generator, max_angle: float) -> np.ndarray:
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(0, max_angle)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


@register_scenario("roadster_space_dock")
class RoadsterSpaceDock(RoadsterMixin, MujocoScenarioEnv):
    CONTROLS = ("thrust_x", "thrust_y", "thrust_z", "pitch", "roll", "yaw")
    FUEL_SECONDS = 6.0
    thrust_scale = 0.15  # gentle RCS thrusters, ~1.2 m/s^2
    wind_body = None  # no air in space
    episode_seconds = 45.0
    camera = "chase"

    CURRICULUM = [
        {"distance": [8.0, 15.0], "misalign": 0.2, "drift": 0.1, "tumble": 0.0},
        {"distance": [15.0, 30.0], "misalign": 0.6, "drift": 0.3, "tumble": 0.1},
        {"distance": [25.0, 45.0], "misalign": 1.2, "drift": 0.6, "tumble": 0.3},
        {"distance": [35.0, 60.0], "misalign": 3.14, "drift": 1.0, "tumble": 0.6},
    ]

    def build_xml(self) -> str:
        # Pin the scene extent: the huge Earth would otherwise push the near clip plane past the car.
        return roadster_xml(WORLD, name="roadster_space_dock", gravity="0 0 0", start=(-20, 0, 0),
                            extra='<statistic extent="20" center="0 0 0"/>')

    def setup(self) -> None:
        super().setup()
        self.station_geom = self.geom("station")
        self.shaper = PotentialShaper()

    def randomize(self, p: dict) -> None:
        rng = self.rng
        direction = rng.normal(size=3)
        direction[0] = -abs(direction[0]) - 1.0  # approach from the docking side
        direction /= np.linalg.norm(direction)
        pos = DOCK_POINT + p["distance"] * direction
        quat = random_quat(rng, p["misalign"])
        self.place_car(pos, quat=quat, vel=rng.normal(size=3) * p["drift"])
        dof = self.root_qvel
        self.data.qvel[dof + 3:dof + 6] = rng.normal(size=3) * p["tumble"]
        self.d0 = float(np.linalg.norm(pos - DOCK_POINT))
        self.shaper.prev = None

    def alignment(self) -> float:
        return float(self.car_rot[:, 0] @ DOCK_AXIS)

    def get_obs(self) -> np.ndarray:
        R = self.car_rot
        rel = R.T @ (DOCK_POINT - self.car_pos)
        return np.array([
            *(rel / 20.0), *(self.car_vel_body / 5.0), *(self.car_angvel_body / 2.0),
            *(R.T @ DOCK_AXIS), *(R.T @ np.array([0.0, 0.0, 1.0])),
            np.linalg.norm(rel) / 50.0, self.fuel_frac(),
        ], dtype=np.float32)

    def check_done(self):
        d = float(np.linalg.norm(self.car_pos - DOCK_POINT))
        speed = float(np.linalg.norm(self.car_vel))
        spin = float(np.linalg.norm(self.car_angvel_body))
        touching = self.shell_hit()
        if touching and speed > 0.8:
            return True, False, "collision"
        if d > 2.0 * self.d0 + 20.0:
            return True, False, "lost_in_space"
        docked = d < 0.7 and speed < 0.4 and self.alignment() > 0.97 and spin < 0.2
        return (True, True, "docked") if docked else (False, False, "")

    def potential(self) -> float:
        d = float(np.linalg.norm(self.car_pos - DOCK_POINT))
        speed = float(np.linalg.norm(self.car_vel))
        excess = max(0.0, speed - (0.3 + 0.25 * d))
        return (-100.0 * min(d / self.d0, 1.5) - 15.0 * (1.0 - self.alignment())
                - 10.0 * excess - 2.0 * float(np.linalg.norm(self.car_angvel_body)))

    def compute_reward(self, action, terminated, success):
        terms = {"shaping": self.shaper(self.potential()), "fuel": -0.05 * self.thrust_use, "time": -0.02}
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        if success:
            terms["docked"] = 100.0 + 50.0 * self.fuel_frac()
        return float(sum(terms.values())), terms
