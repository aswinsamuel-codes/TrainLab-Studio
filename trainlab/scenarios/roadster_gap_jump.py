"""GapJump: speed up, fire the thrusters to clear a pit, land on a raised ledge,
then brake to a stop inside the green parking box.

The Lunar Lander idea in 3D: a dense potential-based shaping reward guides the car
towards the box, a big bonus for parking, a big penalty for crashing.
"""

from __future__ import annotations

import numpy as np

from trainlab.core import MujocoScenarioEnv, PotentialShaper, register_scenario
from trainlab.core.rewards import braking_envelope
from trainlab.scenarios._roadster import RoadsterMixin, roadster_xml

WORLD = """
<geom name="floor" class="world" type="plane" size="400 400 0.1" pos="0 0 -8" material="mat_grid"/>
<geom name="start_road" class="world" type="box" size="40 5 1" pos="-40 0 -1" material="mat_road"/>
<body name="ledge" mocap="true" pos="54 0 -4">
  <geom class="world" type="box" size="50 5 4" material="mat_ledge"/>
</body>
<geom name="edge_line" class="deco" type="box" size="0.3 5 0.01" pos="-0.3 0 0.005" rgba="1 0.8 0 1"/>
<geom name="park" class="deco" type="box" size="4 2.5 0.02" pos="25 0 0.02" rgba="0.1 0.9 0.3 0.55"/>
"""

START_X = -50.0
PARK_HALF = np.array([4.0, 2.5])
LEDGE_HALF_LEN = 50.0


@register_scenario("roadster_gap_jump")
class RoadsterGapJump(RoadsterMixin, MujocoScenarioEnv):
    CONTROLS = ("throttle", "steer", "brake", "lift", "pitch", "roll", "yaw")
    FUEL_SECONDS = 4.0
    episode_seconds = 30.0

    # Level 0 is a crack you can drive over: first learn to park, then learn to jump.
    # (With a real gap at level 0, PPO learns to stop safely at the edge and never leaves.)
    CURRICULUM = [
        {"gap": [0.3, 1.0], "ledge": 0.0, "park": [15.0, 25.0], "wind": 0.0},
        {"gap": [1.5, 3.0], "ledge": 0.0, "park": [20.0, 30.0], "wind": 0.0},
        {"gap": [4.0, 7.0], "ledge": [0.0, 1.0], "park": [25.0, 35.0], "wind": 0.0},
        {"gap": [6.0, 10.0], "ledge": [0.5, 2.5], "park": [25.0, 40.0], "wind": 2.0},
        {"gap": [8.0, 13.0], "ledge": [1.5, 4.0], "park": [30.0, 40.0], "wind": 4.0},
        {"gap": [10.0, 16.0], "ledge": [2.5, 5.0], "park": [30.0, 45.0], "wind": 8.0},
    ]

    def build_xml(self) -> str:
        return roadster_xml(WORLD, name="roadster_gap_jump", start=(START_X, 0, 0.7))

    def setup(self) -> None:
        super().setup()
        self.park_geom = self.geom("park")
        self.shaper = PotentialShaper()

    def randomize(self, p: dict) -> None:
        gap, h = p["gap"], p["ledge"]
        self.gap_end, self.ledge_h = gap, h
        self.park_center = np.array([gap + p["park"], 0.0, h + 0.6])
        self.set_box("ledge", pos=[gap + LEDGE_HALF_LEN, 0, h - 4.0])
        self.model.geom_pos[self.park_geom] = [self.park_center[0], 0, h + 0.02]
        self.place_car([START_X, self.rng.uniform(-1, 1), 0.7], yaw=self.rng.uniform(-0.05, 0.05))
        self.airborne = False
        self.prev_vz = 0.0
        self.d0 = float(np.linalg.norm(self.park_center - np.array([START_X, 0, 0.6])))
        self.shaper.prev = None

    # ------------------------------------------------------------ observation
    def get_obs(self) -> np.ndarray:
        p = self.car_pos
        return np.array([
            -p[0] / 50.0,                       # distance to the pit edge
            (self.gap_end - p[0]) / 50.0,       # distance to the ledge
            (self.park_center[0] - p[0]) / 50.0,  # distance to the parking box
            p[1] / 5.0, p[2] / 5.0, (p[2] - self.ledge_h) / 5.0,
            *self.car_obs(),
            self.gap_end / 16.0, self.ledge_h / 5.0,
        ], dtype=np.float32)

    # ---------------------------------------------------------- termination
    def in_park_box(self) -> bool:
        p = self.car_pos
        return bool(np.all(np.abs(p[:2] - self.park_center[:2]) < PARK_HALF))

    def check_done(self):
        p, v = self.car_pos, self.car_vel
        wheels = self.wheel_contacts().sum()
        crash = ""
        if self.shell_hit():
            crash = "body_hit"
        elif self.up_z < 0.5:
            crash = "flipped"
        elif p[2] < self.ledge_h - 2.5 and p[2] < -2.5:
            crash = "fell"
        elif self.airborne and wheels > 0 and self.prev_vz < -8.0:
            crash = "hard_landing"
        elif p[0] > self.gap_end + 2 * LEDGE_HALF_LEN:
            crash = "overshot"
        self.airborne = wheels == 0
        self.prev_vz = float(v[2])
        if crash:
            return True, False, crash

        parked = (wheels >= 3 and p[0] > self.gap_end and self.in_park_box()
                  and np.linalg.norm(v) < 0.5 and self.up_z > 0.96)
        return (True, True, "parked") if parked else (False, False, "")

    # --------------------------------------------------------------- reward
    def potential(self) -> float:
        p, speed = self.car_pos, float(np.linalg.norm(self.car_vel))
        d = float(np.linalg.norm(p - self.park_center))
        excess = max(0.0, speed - braking_envelope(self.park_center[0] - p[0], decel=7.0))
        return -100.0 * min(d / self.d0, 1.5) - 20.0 * (1.0 - self.up_z) - 2.0 * excess

    def compute_reward(self, action, terminated, success):
        c = self.controls
        terms = {
            "shaping": self.shaper(self.potential()),
            "lift_fuel": -0.3 * c["lift"],
            "attitude_fuel": -0.03 * (abs(c["pitch"]) + abs(c["roll"]) + abs(c["yaw"])),
            "time": -0.03,
        }
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        if success:
            terms["parked"] = 100.0 + 30.0 * self.fuel_frac()
        return float(sum(terms.values())), terms
