"""Waypoints: reach a chain of targets in order.

Works for flying bodies (targets float at different heights) and ground bodies (targets sit
on the ground, and the harder levels scatter rocks along the route to steer around).

Body needs: the Body basics (pos, rot, vel, place), crash_reason(), effort(), and the
traits flies / spawn_z / reach.
"""

from __future__ import annotations

import numpy as np

from trainlab.core.rewards import PotentialShaper
from trainlab.tasks.base import Task

N_MAX = 5
N_ROCKS = 12
HIDDEN = (0.0, 0.0, -5.0)  # parked under the floor


class WaypointsTask(Task):
    name = "waypoints"
    episode_seconds = 25.0
    start = (0.0, 0.0, 0.3)
    CURRICULUM = [  # flying bodies
        {"count": 1, "dist": [1.5, 3.0], "height": [1.0, 2.0], "rocks": 0, "wind": 0.0},
        {"count": 2, "dist": [2.0, 4.5], "height": [1.0, 3.0], "rocks": 0, "wind": 0.0},
        {"count": 3, "dist": [3.0, 7.0], "height": [1.0, 4.0], "rocks": 0, "wind": 1.5},
        {"count": 5, "dist": [4.0, 10.0], "height": [1.0, 6.0], "rocks": 0, "wind": 3.0},
    ]
    GROUND_CURRICULUM = [  # ground bodies: flat targets, then rocks in the way, gently
        {"count": 1, "dist": [2.0, 4.0], "rocks": 0, "wind": 0.0},
        {"count": 2, "dist": [3.0, 6.0], "rocks": 0, "wind": 0.0},
        {"count": 2, "dist": [3.0, 5.0], "rocks": 2, "wind": 0.0},  # first rocks: just a couple
        {"count": 3, "dist": [4.0, 7.0], "rocks": 5, "wind": 0.0},
        {"count": 5, "dist": [3.0, 6.5], "rocks": 12, "wind": 0.0},  # ~25 m of rocky route in 25 s
    ]

    @classmethod
    def curriculum_for(cls, body_cls) -> list[dict]:
        return cls.CURRICULUM if body_cls.flies else cls.GROUND_CURRICULUM

    @classmethod
    def start_for(cls, body_cls) -> tuple:
        return (0.0, 0.0, body_cls.spawn_z)

    def world_xml(self) -> str:
        targets = "\n".join(f'<geom name="wp{i}" class="deco" type="sphere" size="0.45" pos="0 0 -3" '
                            f'material="mat_target"/>' for i in range(N_MAX))
        rocks = "\n".join(f'<body name="rock{i}" mocap="true" pos="0 0 -5"><geom class="world" type="ellipsoid" '
                          f'size="0.35 0.35 0.3" material="mat_rock"/></body>' for i in range(N_ROCKS))
        return (super().world_xml()
                + '\n<geom name="helipad" class="deco" type="cylinder" size="0.6 0.003" pos="0 0 0.003" material="mat_pad"/>'
                + '\n<geom class="deco" type="box" size="0.25 0.04 0.004" pos="0 0 0.005" material="mat_mark"/>'
                + "\n" + targets + "\n" + rocks)

    def bind(self, env) -> None:
        super().bind(env)
        self.wp_geoms = [env.geom(f"wp{i}") for i in range(N_MAX)]
        self.rock_bodies = [f"rock{i}" for i in range(N_ROCKS)]
        self.flies = self.body.flies
        self.reach = self.body.reach
        self.shaper = PotentialShaper()

    # ------------------------------------------------------------- episode
    def randomize(self, p: dict) -> None:
        rng = self.env.rng
        z0 = self.body.spawn_z
        self.body.place([0.0, 0.0, z0], yaw=float(rng.uniform(-np.pi, np.pi)))
        self.points = []
        here = np.array([0.0, 0.0, 0.0])
        for _ in range(int(p["count"])):
            ang = rng.uniform(0, 2 * np.pi)
            d = rng.uniform(*self.span("dist"))
            pt = here + np.array([d * np.cos(ang), d * np.sin(ang), 0.0])
            pt[2] = rng.uniform(*self.span("height")) if self.flies else z0
            self.points.append(pt)
            here = pt
        self._place_rocks(int(p.get("rocks", 0)), rng)
        self.idx = 0
        self.seg_start = np.array([0.0, 0.0, z0])
        self._paint()
        self.shaper.prev = None
        self.just_reached = False

    def _place_rocks(self, n: int, rng) -> None:
        """Scatter rocks near the route, never on the start or a target."""
        route = [np.zeros(3)] + self.points
        placed = 0
        for name in self.rock_bodies:
            pos = None
            for _ in range(30 if placed < n else 0):
                a, b = route[rng.integers(len(route) - 1)], route[rng.integers(1, len(route))]
                c = a + (b - a) * rng.uniform(0.2, 0.8)
                c[:2] += rng.normal(scale=1.0, size=2)
                if all(np.linalg.norm((c - q)[:2]) > 1.3 for q in route):
                    pos = c
                    break
            if pos is None:
                self.env.set_box(name, pos=HIDDEN)
                continue
            half = np.array([rng.uniform(0.18, 0.35), rng.uniform(0.18, 0.35), rng.uniform(0.15, 0.3)])
            yaw = rng.uniform(0, np.pi)
            self.env.set_box(name, pos=[pos[0], pos[1], half[2] * 0.8], half_size=half,
                             quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            placed += 1

    def _paint(self) -> None:
        m = self.env.model
        for i, g in enumerate(self.wp_geoms):
            if i < len(self.points) and i >= self.idx:
                m.geom_pos[g] = self.points[i]
                m.geom_rgba[g] = (0.15, 0.9, 0.4, 0.55) if i == self.idx else (0.9, 0.9, 0.95, 0.18)
            else:
                m.geom_pos[g] = (0, 0, -3)
                m.geom_rgba[g][3] = 0.0

    @property
    def target(self) -> np.ndarray:
        return self.points[min(self.idx, len(self.points) - 1)]

    def _dist(self, a, b) -> float:
        diff = a - b
        return float(np.linalg.norm(diff if self.flies else diff[:2]))

    # ---------------------------------------------------------------- obs
    def obs(self) -> list[float]:
        R, pos = self.body.rot, self.body.pos
        cur = R.T @ (self.target - pos)
        nxt = R.T @ (self.points[self.idx + 1] - pos) if self.idx + 1 < len(self.points) else np.zeros(3)
        return [*(cur / 5.0), *(nxt / 5.0), self._dist(self.target, pos) / 10.0, float(pos[2]) / 5.0,
                self.idx / N_MAX]

    def check_done(self):
        pos = self.body.pos
        self.just_reached = False
        if self._dist(pos, self.target) < self.reach:
            self.just_reached = True
            self.seg_start = pos.copy()
            self.idx += 1
            if self.idx >= len(self.points):
                return True, True, "all_waypoints"
            self._paint()
        crash = self.body.crash_reason()
        if crash:
            return True, False, crash
        # Wandering far off counts as a failure too. Without this, fleeing a hard area (e.g. the
        # rocks) was cheaper than trying: the rover learned to drive away into empty space.
        span = max(self._dist(self.target, self.seg_start), 1.0)
        if self._dist(pos, self.target) > 2.0 * span + 5.0 or abs(pos[0]) > 40 or abs(pos[1]) > 40 or pos[2] > 25:
            return True, False, "lost"
        return False, False, ""

    def potential(self) -> float:
        d = self._dist(self.body.pos, self.target)
        span = max(self._dist(self.target, self.seg_start), 1.0)
        tilt = 1.0 - self.body.up_z
        return -15.0 * min(d / span, 3.0) - 5.0 * tilt  # keeps growing as you stray (was capped at 1.5x)

    def compute_reward(self, action, terminated, success):
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        terms = {"time": -0.03, "energy": -0.02 * self.body.effort()}
        if self.just_reached:
            terms["waypoint"] = 25.0
            self.shaper.prev = None  # new target: restart the potential, don't punish the jump
        terms["shaping"] = self.shaper(self.potential())
        if success:
            terms["finished"] = 100.0
        return float(sum(terms.values())), terms

    def update_visuals(self, dt: float) -> None:
        if self.idx < len(self.points):  # gently pulse the active target
            g = self.wp_geoms[self.idx]
            self.env.model.geom_rgba[g][3] = 0.45 + 0.15 * np.sin(6 * self.env.data.time)
