"""TrackRace: lap a floating 3D track with curves, hills and banked corners.

A new random track every episode (domain randomization), so the agent learns to
drive from what it sees ahead rather than memorizing one layout.
"""

from __future__ import annotations

import mujoco
import numpy as np

from trainlab.core import MujocoScenarioEnv, register_scenario
from trainlab.scenarios._roadster import RoadsterMixin, roadster_xml

N_SEG = 60
DENSE = 720
SLAB = 0.3  # half thickness of track slabs
LOOKAHEAD = (8.0, 18.0, 30.0, 45.0, 65.0)

WORLD = '<geom name="floor" class="world" type="plane" size="500 500 0.1" pos="0 0 -12" material="mat_grid"/>\n' + "\n".join(
    f'<body name="seg{i}" mocap="true" pos="0 0 {-20 - i}">'
    f'<geom class="world" type="box" size="4 6 {SLAB}" material="mat_road"/></body>'
    for i in range(N_SEG))


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


@register_scenario("roadster_track_race")
class RoadsterTrackRace(RoadsterMixin, MujocoScenarioEnv):
    CONTROLS = ("throttle", "steer", "brake")
    episode_seconds = 60.0

    CURRICULUM = [
        {"width": 16.0, "wiggle": 0.08, "hills": 0.0, "bank": 0.0, "wind": 0.0},
        {"width": 13.0, "wiggle": [0.1, 0.2], "hills": [0.0, 2.0], "bank": 0.15, "wind": 0.0},
        {"width": [10.0, 12.0], "wiggle": [0.15, 0.3], "hills": [1.0, 4.0], "bank": 0.25, "wind": 3.0},
        {"width": [9.0, 11.0], "wiggle": [0.2, 0.35], "hills": [2.0, 6.0], "bank": 0.3, "wind": 6.0},
    ]

    def build_xml(self) -> str:
        return roadster_xml(WORLD, name="roadster_track_race", start=(60, 0, 0.7))

    def setup(self) -> None:
        super().setup()
        self.seg_names = [f"seg{i}" for i in range(N_SEG)]

    # ------------------------------------------------------------ track
    def centerline(self, theta: np.ndarray, p: dict) -> np.ndarray:
        c = self.track_coef
        r = 60.0 * (1 + p["wiggle"] * np.sin(2 * theta + c[0]) + 0.6 * p["wiggle"] * np.sin(3 * theta + c[1]))
        z = p["hills"] * np.sin(theta + c[2]) + 0.5 * p["hills"] * np.sin(2 * theta + c[3])
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=-1)

    def randomize(self, p: dict) -> None:
        self.track_coef = self.rng.uniform(0, 2 * np.pi, size=4)
        th = np.linspace(0, 2 * np.pi, N_SEG + 1)
        pts = self.centerline(th, p)
        W = p["width"]
        yaws = []
        for i in range(N_SEG):
            t = pts[i + 1] - pts[i]
            yaws.append(np.arctan2(t[1], t[0]))
        yaws = np.unwrap(yaws)
        for i, g in enumerate(self.seg_names):
            a, b = pts[i], pts[i + 1]
            t = b - a
            L = float(np.linalg.norm(t))
            yaw = yaws[i]
            pitch = -np.arcsin(t[2] / L)
            turn = (yaws[(i + 1) % N_SEG] - yaws[i - 1] + np.pi) % (2 * np.pi) - np.pi
            roll = -np.clip(turn * 2.0, -p["bank"], p["bank"])  # raise the outside of the corner
            R = _rz(yaw) @ _ry(pitch) @ _rx(roll)
            center = (a + b) / 2 - SLAB * R[:, 2]
            self.set_box(g, pos=center, quat=rot_to_quat(R), half_size=np.array([L / 2 + 0.8, W / 2, SLAB]))

        dense_th = np.linspace(0, 2 * np.pi, DENSE, endpoint=False)
        self.dense = self.centerline(dense_th, p)
        seglen = np.linalg.norm(np.roll(self.dense, -1, axis=0) - self.dense, axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seglen)[:-1]])
        self.track_len = float(seglen.sum())
        self.width = W

        start = self.dense[0]
        yaw0 = float(np.arctan2(*(self.dense[1] - self.dense[0])[[1, 0]]))
        self.place_car(start + np.array([0, 0, 0.75]), yaw=yaw0)
        self.idx = 0
        self.progress = 0.0
        self.prev_progress = 0.0
        self._lat, self._vert, self._herr = 0.0, 0.75, 0.0

    def nearest(self, pos: np.ndarray) -> int:
        window = (self.idx + np.arange(-20, 40)) % DENSE
        d = np.linalg.norm(self.dense[window, :2] - pos[:2], axis=1)
        return int(window[np.argmin(d)])

    def point_at(self, s: float) -> np.ndarray:
        s = s % self.track_len
        i = int(np.searchsorted(self.arc, s, side="right") - 1)
        return self.dense[i]

    def track_state(self):
        pos = self.car_pos
        j = self.nearest(pos)
        delta = self.arc[j] - self.arc[self.idx]
        if delta < -self.track_len / 2:
            delta += self.track_len
        elif delta > self.track_len / 2:
            delta -= self.track_len
        self.progress += delta
        self.idx = j
        tangent = self.dense[(j + 1) % DENSE] - self.dense[j]
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0], 0.0])
        lateral = float((pos - self.dense[j]) @ normal)
        vertical = float(pos[2] - self.dense[j][2])
        heading_err = self.heading - np.arctan2(tangent[1], tangent[0])
        return lateral, vertical, (heading_err + np.pi) % (2 * np.pi) - np.pi

    # ----------------------------------------------------------- obs/done
    def get_obs(self) -> np.ndarray:
        R, pos = self.car_rot, self.car_pos
        s = self.arc[self.idx]
        ahead = [R.T @ (self.point_at(s + d) - pos) / 50.0 for d in LOOKAHEAD]
        return np.array([self._lat / (self.width / 2), self._vert / 3.0, np.sin(self._herr), np.cos(self._herr),
                         *np.concatenate(ahead), *self.car_obs()[:14], self.width / 16.0], dtype=np.float32)

    def check_done(self):
        # Runs first after each physics step: update progress once, cache the rest.
        self._lat, self._vert, self._herr = self.track_state()
        if abs(self._lat) > self.width / 2 + 0.5 or self._vert < -2.0:
            return True, False, "off_track"
        if self.up_z < 0.5:
            return True, False, "flipped"
        if self.progress >= self.track_len:
            return True, True, "lap_complete"
        return False, False, ""

    def compute_reward(self, action, terminated, success):
        lateral = self._lat
        gained = self.progress - self.prev_progress
        self.prev_progress = self.progress
        terms = {"progress": 0.5 * gained, "time": -0.05,
                 "centre": -0.05 * (lateral / (self.width / 2)) ** 2}
        if terminated and not success:
            return -100.0, {"crash": -100.0}
        if success:
            terms["lap"] = 100.0 + 2.0 * max(0.0, 50.0 - self.elapsed)
        return float(sum(terms.values())), terms


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
