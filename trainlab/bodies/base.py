"""A Body is the thing being trained: a drone, rover, bike, robot... It owns its model,
its controls, how actions become motor commands, and its own sensor readings.

A Task (trainlab/tasks/) owns the world, the goal and the reward. Any task can use any
body that offers what the task needs, so bodies and tasks mix and match:
``trainlab.make("drone_landing")``, later ``rover_waypoints``, ...

Subclass ``Body`` and fill in:
    xml()            -> dict(assets=, defaults=, body=, actuators=, world_extra=)
    bind(env)        cache ids once the model is compiled
    place(pos, yaw)  put the body somewhere at reset
    apply_action(a)  map the agent's [-1, 1] action to controls
    obs()            body-only observations (list of floats, roughly unit scale)
Optional: substep_control() (runs every physics step), update_visuals(dt) (render only),
crash_reason() (body-level failures such as flipping over).
"""

from __future__ import annotations

import mujoco
import numpy as np


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class Body:
    name = "body"
    root = "body"  # MuJoCo body name of the root (free joint)
    CONTROLS: tuple[str, ...] = ()
    drag_coef = 0.0  # air drag / wind force coefficient for this body
    extent = 25.0  # scene scale for rendering (visual only)
    # Traits tasks use to adapt (e.g. waypoints in the air vs. on the ground):
    flies = False
    spawn_z = 0.5  # root height when standing/parked on flat ground
    reach = 0.5  # how close (m) counts as reaching a target

    def xml(self, start=(0.0, 0.0, 1.0)) -> dict:
        raise NotImplementedError

    # ------------------------------------------------------------- binding
    def bind(self, env) -> None:
        self.env = env
        m = env.model
        self.root_id = env.body(self.root)
        jnt = m.body_jntadr[self.root_id]
        self.qpos_adr = m.jnt_qposadr[jnt]
        self.qvel_adr = m.jnt_dofadr[jnt]
        tree = {b for b in range(m.nbody) if self._in_tree(b)}
        self.geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in tree}
        self.mass = float(m.body_subtreemass[self.root_id])

    def _in_tree(self, b: int) -> bool:
        m = self.env.model
        while b > 0:
            if b == self.root_id:
                return True
            b = m.body_parentid[b]
        return False

    # --------------------------------------------------------------- state
    def place(self, pos, yaw: float = 0.0, vel=(0.0, 0.0, 0.0)) -> None:
        d, q, v = self.env.data, self.qpos_adr, self.qvel_adr
        d.qpos[q:q + 3] = pos
        d.qpos[q + 3:q + 7] = yaw_quat(yaw)
        d.qvel[v:v + 3] = vel
        d.qvel[v + 3:v + 6] = 0.0

    @property
    def pos(self) -> np.ndarray:
        return self.env.data.xpos[self.root_id].copy()

    @property
    def rot(self) -> np.ndarray:
        return self.env.data.xmat[self.root_id].reshape(3, 3).copy()

    @property
    def vel(self) -> np.ndarray:
        return self.env.data.qvel[self.qvel_adr:self.qvel_adr + 3].copy()

    @property
    def vel_body(self) -> np.ndarray:
        return self.rot.T @ self.vel

    @property
    def angvel_body(self) -> np.ndarray:
        # free-joint angular velocity is already in the body frame
        return self.env.data.qvel[self.qvel_adr + 3:self.qvel_adr + 6].copy()

    @property
    def up_z(self) -> float:
        return float(self.rot[2, 2])

    @property
    def heading(self) -> float:
        f = self.rot[:, 0]
        return float(np.arctan2(f[1], f[0]))

    def gravity_body(self) -> np.ndarray:
        return self.rot.T @ np.array([0.0, 0.0, -1.0])

    def world_contacts(self) -> list[tuple[int, int]]:
        """(body geom, world geom) pairs currently touching."""
        out = []
        for g1, g2 in self.env.contacts():
            if g1 in self.geoms and g2 not in self.geoms:
                out.append((g1, g2))
            elif g2 in self.geoms and g1 not in self.geoms:
                out.append((g2, g1))
        return out

    # --------------------------------------------------------------- hooks
    def reset_state(self) -> None:
        """Called at every reset before the task places the body."""

    def apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def substep_control(self) -> None:
        pass

    def obs(self) -> list[float]:
        raise NotImplementedError

    def crash_reason(self) -> str:
        return ""

    def effort(self) -> float:
        """0..1 how hard the body is working this step (for energy costs)."""
        return 0.0

    def update_visuals(self, dt: float) -> None:
        pass


GROUND_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # ray casts that only see world geoms


class LaserEyes:
    """A fan of rangefinder beams ("laser eyes"/simple lidar) any body can carry.

    ``read()`` returns distances normalised to [0, 1] (1 = nothing within range). Beams only
    see world geoms (render group 0), never the body itself. With ``beams=True`` the env
    also draws them (render only) as faint red lines.
    """

    def __init__(self, n: int = 7, spread_deg: float = 120.0, max_dist: float = 4.0,
                 origin=(0.0, 0.0, 0.0), pitch_deg: float = 0.0, name: str = "laser"):
        self.n, self.max_dist, self.origin, self.name = n, max_dist, np.asarray(origin, float), name
        ang = np.radians(np.linspace(-spread_deg / 2, spread_deg / 2, n))
        pitch = np.radians(pitch_deg)
        self.dirs = np.stack([np.cos(ang) * np.cos(pitch), np.sin(ang) * np.cos(pitch),
                              np.full(n, -np.sin(pitch))], axis=1)  # in the body frame
        self.last = np.ones(n)

    def world_xml(self) -> str:
        """Beam geoms for rendering. Each sits in its own mocap body: MuJoCo re-reads the position
        of geoms on the static world body but not their rotation, so a plain geom can't aim."""
        return "\n".join(f'<body name="{self.name}{i}" mocap="true" pos="0 0 -5"><geom class="deco" type="capsule" '
                         f'size="0.006 0.01" rgba="1 0.15 0.1 0"/></body>' for i in range(self.n))

    def bind(self, env) -> None:
        self.env = env
        bodies = [env.body(f"{self.name}{i}") for i in range(self.n)]
        self.beam_mocap = [env.model.body_mocapid[b] for b in bodies]
        self.beam_geoms = [env.model.body_geomadr[b] for b in bodies]

    def read(self, pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        hit = np.zeros(1, dtype=np.int32)
        start = pos + rot @ self.origin
        out = np.empty(self.n)
        for i, d in enumerate(self.dirs):
            dist = mujoco.mj_ray(self.env.model, self.env.data, start, rot @ d, GROUND_ONLY, 1, -1, hit)
            out[i] = min(dist, self.max_dist) / self.max_dist if dist >= 0 else 1.0
        self.last, self.start = out, start
        self.rot = rot
        return out

    def draw(self) -> None:
        m, data = self.env.model, self.env.data
        q = np.zeros(4)
        for i, g in enumerate(self.beam_geoms):
            length = max(self.last[i] * self.max_dist, 0.02)
            d = self.rot @ self.dirs[i]
            mujoco.mju_quatZ2Vec(q, d)
            data.mocap_pos[self.beam_mocap[i]] = self.start + d * length / 2
            data.mocap_quat[self.beam_mocap[i]] = q
            m.geom_size[g][:2] = (0.006, length / 2)
            m.geom_rgba[g][3] = 0.55 if self.last[i] < 1.0 else 0.18  # brighter when it hits something


def ray_down(env, start: np.ndarray, max_dist: float = 5.0) -> float:
    """Distance from ``start`` straight down to the world, or -1 if nothing within max_dist."""
    hit = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(env.model, env.data, start, np.array([0.0, 0.0, -1.0]), GROUND_ONLY, 1, -1, hit)
    return dist if 0 <= dist <= max_dist else -1.0
