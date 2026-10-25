"""Base class every TrainLab scenario builds on.

A scenario subclasses ``MujocoScenarioEnv`` and fills in these hooks:

    build_xml()                  -> MJCF string for the world (or set ``xml_file``)
    randomize(params)            -> move things around at reset (params = current level)
    get_obs()                    -> 1-D float observation
    check_done()                 -> (terminated, success, reason)
    compute_reward(action, terminated, success) -> (reward, {term: value})

Optional: ``apply_action`` (map the normalized action to ``data.ctrl``),
``action_dim`` / ``CURRICULUM`` / ``wind_body`` class attributes.
Everything else (stepping, seeding, rendering, wind, curriculum levels) is shared.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from trainlab.core.wind import Wind


class MujocoScenarioEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    scenario_name = "unnamed"
    xml_file: str | None = None  # MJCF path relative to the scenario file
    frame_skip = 10
    episode_seconds = 30.0
    camera = "chase"
    wind_body: str | None = None  # body that feels wind drag (None = no wind)
    drag_coef = 1.5  # F = c * |w - v| * (w - v)

    # One dict per difficulty level. Values are fixed numbers or [low, high]
    # ranges sampled uniformly at every reset (domain randomization).
    CURRICULUM: list[dict] = [{}]

    def __init__(self, render_mode: str | None = None, level: int = 0, wind: float | None = None,
                 width: int = 640, height: int = 480):
        self.render_mode = render_mode
        self.level = int(np.clip(level, 0, len(self.CURRICULUM) - 1))
        self.wind_override = wind
        self.render_width, self.render_height = width, height

        self.model = mujoco.MjModel.from_xml_string(self.build_xml())
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * self.frame_skip
        self.metadata = dict(self.metadata, render_fps=int(round(1.0 / self.dt)))

        self._renderer = None
        self._viewer = None
        self._last_frame_time = 0.0
        self.wind = Wind()
        self.params: dict = {}

        self.setup()
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.action_dim(),), dtype=np.float32)
        self._reset_sim(np.random.default_rng(0))
        obs = self.get_obs()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    # ------------------------------------------------------------------ hooks
    def build_xml(self) -> str:
        if self.xml_file is None:
            raise NotImplementedError("Set xml_file or override build_xml()")
        return (Path(inspect.getfile(type(self))).parent / self.xml_file).read_text()

    def setup(self) -> None:
        """Called once after the model is compiled (cache ids here)."""

    def action_dim(self) -> int:
        return self.model.nu

    def apply_action(self, action: np.ndarray) -> None:
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = lo + (action + 1.0) * 0.5 * (hi - lo)

    def substep_control(self) -> None:
        """Optional low-level controller run every physics step (e.g. traction control)."""

    def randomize(self, params: dict) -> None:
        """Called at reset after mj_resetData, before mj_forward."""

    def get_obs(self) -> np.ndarray:
        raise NotImplementedError

    def check_done(self) -> tuple[bool, bool, str]:
        return False, False, ""

    def compute_reward(self, action, terminated: bool, success: bool) -> tuple[float, dict]:
        raise NotImplementedError

    # --------------------------------------------------------------- gym api
    @classmethod
    def max_episode_steps(cls) -> int:
        # Scenarios use a 0.005 s physics timestep (set in their MJCF <option>).
        return int(round(cls.episode_seconds / (0.005 * cls.frame_skip)))

    def set_level(self, level: int) -> None:
        self.level = int(np.clip(level, 0, len(self.CURRICULUM) - 1))

    def get_level(self) -> int:
        return self.level

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "level" in options:
            self.set_level(options["level"])
        self._reset_sim(self.np_random)
        return self.get_obs(), {"level": self.level, **self.params}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.apply_action(action)
        for _ in range(self.frame_skip):
            self.substep_control()
            self._apply_wind()
            mujoco.mj_step(self.model, self.data)
        self.elapsed += self.dt

        terminated, success, reason = self.check_done()
        reward, terms = self.compute_reward(action, terminated, success)
        obs = self.get_obs()
        if not np.all(np.isfinite(obs)):  # physics blew up: end the episode safely
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            terminated, reason, reward = True, "unstable", -100.0
        info = {"is_success": bool(success), "reason": reason, "reward_terms": terms, "level": self.level}
        if self.render_mode == "human":
            self.render()
        return obs.astype(np.float32), float(reward), bool(terminated), False, info

    def _reset_sim(self, rng: np.random.Generator) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.elapsed = 0.0
        self.params = self.sample_params(rng)
        strength = self.wind_override if self.wind_override is not None else self.params.get("wind", 0.0)
        self.wind.reset(rng, strength)
        self.rng = rng
        self.randomize(self.params)
        mujoco.mj_forward(self.model, self.data)

    def sample_params(self, rng: np.random.Generator) -> dict:
        out = {}
        for key, val in self.CURRICULUM[self.level].items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                out[key] = float(rng.uniform(val[0], val[1]))
            else:
                out[key] = val
        return out

    def _apply_wind(self) -> None:
        # Air drag is always on for wind_body; with no wind it is plain aero drag.
        if self.wind_body is None:
            return
        bid = self.body(self.wind_body)
        w = self.wind.step(self.model.opt.timestep, self.rng)
        rel = w - self.body_velocity(bid)
        force = self.drag_coef * np.linalg.norm(rel) * rel
        self.data.xfrc_applied[bid, :3] = force

    # -------------------------------------------------------------- helpers
    def body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def geom(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)

    def actuator(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def joint(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def body_velocity(self, bid: int) -> np.ndarray:
        """World-frame linear velocity of a body's origin."""
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        return vel[3:].copy()

    def body_angvel(self, bid: int) -> np.ndarray:
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
        return vel[:3].copy()

    def contacts(self) -> list[tuple[int, int]]:
        c = self.data.contact
        return [(int(c.geom1[i]), int(c.geom2[i])) for i in range(self.data.ncon)]

    def set_box(self, body_name: str, pos=None, quat=None, half_size=None) -> None:
        """Move/resize an obstacle at reset time.

        The obstacle must be a ``mocap="true"`` body holding one box geom at its
        origin. (Moving geoms of the static world body breaks MuJoCo's
        precomputed collision tree, so solid moving things live in mocap bodies.)
        """
        m, d = self.model, self.data
        b = self.body(body_name)
        mocap = m.body_mocapid[b]
        if mocap < 0:
            raise ValueError(f"body '{body_name}' must be declared mocap=\"true\"")
        if pos is not None:
            d.mocap_pos[mocap] = pos
        if quat is not None:
            d.mocap_quat[mocap] = quat
        if half_size is not None:
            g = m.body_geomadr[b]
            half_size = np.asarray(half_size, dtype=float)
            m.geom_size[g] = half_size
            m.geom_rbound[g] = float(np.linalg.norm(half_size))
            m.geom_aabb[g] = np.concatenate([np.zeros(3), half_size])
            if m.body_bvhnum[b] == 1:
                m.bvh_aabb[m.body_bvhadr[b]] = m.geom_aabb[g]

    # ------------------------------------------------------------- rendering
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, self.render_width)
                self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, self.render_height)
                self._renderer = mujoco.Renderer(self.model, self.render_height, self.render_width)
            cam = self.camera if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera) >= 0 else -1
            self._renderer.update_scene(self.data, camera=cam)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer

                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera)
                if cid >= 0:
                    self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self._viewer.cam.fixedcamid = cid
            if self._viewer.is_running():
                self._viewer.sync()
            wait = self.dt - (time.perf_counter() - self._last_frame_time)
            if wait > 0:
                time.sleep(wait)
            self._last_frame_time = time.perf_counter()
        return None

    def viewer_running(self) -> bool:
        return self._viewer is None or self._viewer.is_running()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
