"""Shared Roadster car: MJCF assembly, action mapping, fuel and state helpers.

Scenarios mix ``RoadsterMixin`` in *before* ``MujocoScenarioEnv``:

    class GapJump(RoadsterMixin, MujocoScenarioEnv): ...

and choose which controls the agent gets with ``CONTROLS``. Every control is in
[-1, 1]; one-sided ones (brake, lift) use only the positive half, so 0 = "off".
"""

from __future__ import annotations

from pathlib import Path
from string import Template

import mujoco
import numpy as np

ASSET = Path(__file__).resolve().parent.parent / "assets" / "roadster.xml"

ALL_CONTROLS = ("throttle", "steer", "brake", "lift", "pitch", "roll", "yaw",
                "thrust_x", "thrust_y", "thrust_z")
THRUSTER_CONTROLS = {"lift", "pitch", "roll", "yaw", "thrust_x", "thrust_y", "thrust_z"}
WHEELS = ("fl", "fr", "rl", "rr")

DRIVE_TORQUE = 1500.0  # N·m per wheel at full throttle (AWD)
BRAKE_TORQUE = 2500.0  # N·m per wheel at full brake
STEER_MAX = 0.5  # rad
WHEEL_RADIUS = 0.35


N_PUFFS = 48  # smoke particle pool (visual only)


def roadster_xml(world: str, name: str = "roadster", gravity: str = "0 0 -9.81",
                 start=(0.0, 0.0, 0.7), quat=(1.0, 0.0, 0.0, 0.0), extra: str = "") -> str:
    # Unused puffs sit (fully transparent) at the car's start. Parking them far away would blow up
    # MuJoCo's scene extent, which pushes the camera's near clip plane out and hides the car.
    start_str = " ".join(f"{v:g}" for v in start)
    puffs = "\n".join(f'<body name="puff{i}" mocap="true" pos="{start_str}">'
                      f'<geom class="deco" type="sphere" size="0.2" material="mat_smoke"/></body>'
                      for i in range(N_PUFFS))
    # MuJoCo scales the camera's near clip plane by the scene extent. A long road (e.g. the 320 m
    # sprint) would push it ~17 m out and hide the car, so pin a car-sized extent (visual only).
    if "<statistic" not in extra:
        extra += '\n<statistic extent="25"/>'
    return Template(ASSET.read_text()).substitute(
        name=name, gravity=gravity, world=world + "\n" + puffs, extra=extra,
        start=" ".join(f"{v:g}" for v in start), quat=" ".join(f"{v:g}" for v in quat))


# nozzle -> (exit point, exhaust direction), both in the car's frame
NOZZLES = {
    "fl": ((1.6, 0.6, -0.31), (0, 0, -1)), "fr": ((1.6, -0.6, -0.31), (0, 0, -1)),
    "rl": ((-1.6, 0.6, -0.31), (0, 0, -1)), "rr": ((-1.6, -0.6, -0.31), (0, 0, -1)),
    "rear": ((-2.23, 0, 0), (-1, 0, 0)), "front": ((2.2, 0, 0.02), (1, 0, 0)),
    "left": ((1.6, 1.02, 0.02), (0, 1, 0)), "right": ((1.6, -1.02, 0.02), (0, -1, 0)),
}
GROUND_ONLY = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)  # ray casts see world geoms (group 0) only


class ThrusterFX:
    """Cold-gas plumes + smoke puffs. Pure visuals: only runs when the env renders."""

    def __init__(self, env):
        self.env = env
        m = env.model
        self.plumes = {k: env.geom(f"plume_{k}") for k in NOZZLES}
        self.puff_mocap = [m.body_mocapid[env.body(f"puff{i}")] for i in range(N_PUFFS)]
        self.puff_geom = [m.body_geomadr[env.body(f"puff{i}")] for i in range(N_PUFFS)]
        self.rng = np.random.default_rng(1)
        self.space = abs(m.opt.gravity[2]) < 1e-6  # vacuum: no lingering smoke
        self.life = 0.6 if self.space else 1.6
        self.reset()

    def reset(self) -> None:
        self.pos = np.zeros((N_PUFFS, 3))
        self.vel = np.zeros((N_PUFFS, 3))
        self.age = np.full(N_PUFFS, np.inf)
        self.strength = np.zeros(N_PUFFS)
        self.next = 0
        for k, g in self.plumes.items():
            self.env.model.geom_rgba[g][3] = 0.0
        for i in range(N_PUFFS):
            self._hide(i)

    def _hide(self, i: int) -> None:
        self.env.model.geom_rgba[self.puff_geom[i]][3] = 0.0  # invisible; position doesn't matter

    def levels(self) -> dict[str, float]:
        """How hard each nozzle fires (0..1), from the thrust the car actually applied."""
        e, c = self.env, self.env.controls
        if e.fuel <= 0.0:
            return dict.fromkeys(NOZZLES, 0.0)
        up = max(c["thrust_z"] if "thrust_z" in e.CONTROLS else c["lift"], 0.0)
        p, r, y = c["pitch"], c["roll"], c["yaw"]
        lv = {  # +pitch (nose down) fires the rear jets, +roll lifts the left side, +yaw pushes the nose left
            "fl": up + 0.6 * max(-p, 0) + 0.6 * max(r, 0), "fr": up + 0.6 * max(-p, 0) + 0.6 * max(-r, 0),
            "rl": up + 0.6 * max(p, 0) + 0.6 * max(r, 0), "rr": up + 0.6 * max(p, 0) + 0.6 * max(-r, 0),
            "rear": max(c["thrust_x"], 0), "front": max(-c["thrust_x"], 0),
            "right": max(c["thrust_y"], 0) + 0.6 * max(y, 0), "left": max(-c["thrust_y"], 0) + 0.6 * max(-y, 0),
        }
        return {k: float(np.clip(v, 0.0, 1.0)) for k, v in lv.items()}

    def update(self, dt: float) -> None:
        e, m, d = self.env, self.env.model, self.env.data
        R, origin, v_car = e.car_rot, e.car_pos, e.car_vel
        for k, level in self.levels().items():
            g = self.plumes[k]
            exit_pt, direction = (np.array(a, dtype=float) for a in NOZZLES[k])
            if level < 0.02:
                m.geom_rgba[g][3] = 0.0
                continue
            i = level * (1.0 + 0.12 * self.rng.standard_normal())  # flicker
            half, radius = 0.05 + 0.45 * i, 0.08 + 0.12 * i  # short, fat, see-through jet
            m.geom_size[g][:2] = (radius, half)
            m.geom_pos[g] = exit_pt + direction * (half + radius * 0.3)
            m.geom_rgba[g][3] = 0.1 + 0.25 * min(i, 1.0)
            if level > 0.12 and self.rng.random() < 0.35 + 0.6 * level:
                self._emit(origin + R @ (exit_pt + direction * 2 * half), R @ direction, v_car, level)
        self._advance(dt)

    def _emit(self, start: np.ndarray, world_dir: np.ndarray, v_car: np.ndarray, level: float) -> None:
        e = self.env
        i = self.next
        self.next = (self.next + 1) % N_PUFFS
        speed = 5.0 * level
        vel = world_dir * speed + 0.5 * v_car + self.rng.normal(scale=0.4, size=3)
        pos = start.copy()
        if not self.space and world_dir[2] < -0.3:  # jet hits the ground: smoke spreads sideways
            hit = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(e.model, e.data, start, world_dir, GROUND_ONLY, 1, -1, hit)
            if 0 <= dist < 2.5:
                pos = start + world_dir * dist + np.array([0, 0, 0.15])
                ang = self.rng.uniform(0, 2 * np.pi)
                vel = np.array([np.cos(ang), np.sin(ang), 0.15]) * speed * 0.8 + 0.4 * v_car
        self.pos[i], self.vel[i], self.age[i], self.strength[i] = pos, vel, 0.0, level

    def _advance(self, dt: float) -> None:
        m, d = self.env.model, self.env.data
        live = self.age < self.life
        self.age[live] += dt
        drag = np.exp(-(0.8 if self.space else 2.5) * dt)
        self.vel[live] *= drag
        if not self.space:
            self.vel[live, 2] += 0.6 * dt  # warm gas drifts up a little
        self.pos[live] += self.vel[live] * dt
        for i in np.flatnonzero(self.age < np.inf):
            t = self.age[i] / self.life
            if t >= 1.0:
                self.age[i] = np.inf
                self._hide(i)
                continue
            g = self.puff_geom[i]
            m.geom_size[g][0] = 0.2 + (0.5 if self.space else 1.3) * t
            m.geom_rgba[g][3] = 0.32 * self.strength[i] * (1.0 - t) ** 1.5
            d.mocap_pos[self.puff_mocap[i]] = self.pos[i]


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class RoadsterMixin:
    CONTROLS: tuple[str, ...] = ("throttle", "steer", "brake")
    FUEL_SECONDS = 6.0  # seconds of full-power thruster use
    thrust_scale = 1.0
    wind_body = "chassis"

    # ----------------------------------------------------------------- setup
    def setup(self) -> None:
        m = self.model
        self.chassis_id = self.body("chassis")
        self.root_qpos = m.jnt_qposadr[self.joint("root")]
        self.root_qvel = m.jnt_dofadr[self.joint("root")]
        self.wheel_geoms = [self.geom(f"wheel_{w}") for w in WHEELS]
        self.spin_dofs = [m.jnt_dofadr[self.joint(f"spin_{w}")] for w in WHEELS]
        self.drive_ids = [self.actuator(f"drive_{w}") for w in WHEELS]
        self.steer_ids = [self.actuator("steer_fl"), self.actuator("steer_fr")]
        self.thruster_ids = {k: self.actuator(k) for k in
                             ("thrust_x", "thrust_y", "thrust_z", "torque_roll", "torque_pitch", "torque_yaw")}

        # Every geom attached to the car's body tree, and the solid body shell.
        car_bodies = {b for b in range(m.nbody) if self._in_car(b)}
        self.car_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in car_bodies}
        self.shell_geoms = {g for g in range(m.ngeom)
                            if m.geom_bodyid[g] == self.chassis_id and m.geom_contype[g] != 0}
        self.fuel = self.FUEL_SECONDS
        self.thrust_use = 0.0
        self.controls = dict.fromkeys(ALL_CONTROLS, 0.0)
        # Exhaust effects cost nothing while training: training envs never render.
        self.fx = ThrusterFX(self) if self.render_mode is not None else None

    def update_visuals(self) -> None:
        if self.fx is not None:
            # Poses refresh on the next physics step (a 50 ms lag nobody can see). Calling
            # mj_kinematics here would also refresh the pose the controllers read, making a
            # rendered car behave slightly differently from the training cars.
            self.fx.update(self.dt)

    def _in_car(self, b: int) -> bool:
        while b > 0:
            if b == self.chassis_id:
                return True
            b = self.model.body_parentid[b]
        return False

    def action_dim(self) -> int:
        return len(self.CONTROLS)

    # ---------------------------------------------------------------- reset
    def place_car(self, pos, yaw: float = 0.0, vel=(0.0, 0.0, 0.0), quat=None) -> None:
        q = self.root_qpos
        self.data.qpos[q:q + 3] = pos
        self.data.qpos[q + 3:q + 7] = yaw_quat(yaw) if quat is None else quat
        self.data.qvel[self.root_qvel:self.root_qvel + 3] = vel
        self.fuel = self.FUEL_SECONDS
        self.thrust_use = 0.0
        self.controls = dict.fromkeys(ALL_CONTROLS, 0.0)
        if getattr(self, "fx", None) is not None:
            self.fx.reset()

    # --------------------------------------------------------------- action
    def apply_action(self, action: np.ndarray) -> None:
        c = dict.fromkeys(ALL_CONTROLS, 0.0)
        c.update(zip(self.CONTROLS, (float(a) for a in action)))
        for k in ("brake", "lift"):
            c[k] = max(0.0, c[k])
        self.controls = c
        ctrl = self.data.ctrl

        self.substep_control()
        for act in self.steer_ids:
            ctrl[act] = c["steer"] * STEER_MAX

        tz = c["thrust_z"] if "thrust_z" in self.CONTROLS else c["lift"]
        thrust = {"thrust_x": c["thrust_x"], "thrust_y": c["thrust_y"], "thrust_z": tz,
                  "torque_roll": c["roll"], "torque_pitch": c["pitch"], "torque_yaw": c["yaw"]}
        use = abs(thrust["thrust_x"]) + abs(thrust["thrust_y"]) + abs(tz) \
            + 0.3 * (abs(c["roll"]) + abs(c["pitch"]) + abs(c["yaw"]))
        if self.fuel <= 0.0:
            thrust = dict.fromkeys(thrust, 0.0)
            use = 0.0
        self.thrust_use = use
        self.fuel = max(0.0, self.fuel - use * self.thrust_scale * self.dt)
        for k, v in thrust.items():
            ctrl[self.thruster_ids[k]] = v * self.thrust_scale

    def substep_control(self) -> None:
        """Wheel torques at physics rate: traction control + anti-lock brakes."""
        c, ctrl = self.controls, self.data.ctrl
        v_fwd = self.car_vel_body[0]
        for act, dof in zip(self.drive_ids, self.spin_dofs):
            omega = self.data.qvel[dof]
            slip = (omega * WHEEL_RADIUS - v_fwd) * np.sign(c["throttle"])
            tc = float(np.clip(1.0 - (slip - 0.5) / 1.0, 0.0, 1.0))
            ctrl[act] = c["throttle"] * DRIVE_TORQUE * tc - c["brake"] * BRAKE_TORQUE * np.tanh(2.0 * omega)

    # ---------------------------------------------------------------- state
    @property
    def car_pos(self) -> np.ndarray:
        return self.data.xpos[self.chassis_id].copy()

    @property
    def car_rot(self) -> np.ndarray:
        return self.data.xmat[self.chassis_id].reshape(3, 3).copy()

    @property
    def car_vel(self) -> np.ndarray:
        return self.data.qvel[self.root_qvel:self.root_qvel + 3].copy()

    @property
    def car_vel_body(self) -> np.ndarray:
        return self.car_rot.T @ self.car_vel

    @property
    def car_angvel_body(self) -> np.ndarray:
        # Free-joint angular velocity is already expressed in the body frame.
        return self.data.qvel[self.root_qvel + 3:self.root_qvel + 6].copy()

    @property
    def up_z(self) -> float:
        """cos(tilt): 1 when level, 0 on its side, -1 upside down."""
        return float(self.car_rot[2, 2])

    @property
    def heading(self) -> float:
        f = self.car_rot[:, 0]
        return float(np.arctan2(f[1], f[0]))

    def gravity_body(self) -> np.ndarray:
        return self.car_rot.T @ np.array([0.0, 0.0, -1.0])

    def fuel_frac(self) -> float:
        return self.fuel / self.FUEL_SECONDS

    def wheel_contacts(self) -> np.ndarray:
        touching = np.zeros(4, dtype=np.float32)
        for g1, g2 in self.contacts():
            for i, wg in enumerate(self.wheel_geoms):
                if (g1 == wg and g2 not in self.car_geoms) or (g2 == wg and g1 not in self.car_geoms):
                    touching[i] = 1.0
        return touching

    def shell_hit(self, ignore: set[int] | None = None) -> bool:
        """True if the car body (not the wheels) is touching the world."""
        ignore = ignore or set()
        for g1, g2 in self.contacts():
            if g1 in self.shell_geoms and g2 not in self.car_geoms and g2 not in ignore:
                return True
            if g2 in self.shell_geoms and g1 not in self.car_geoms and g1 not in ignore:
                return True
        return False

    def wheel_slip(self) -> float:
        """Mean |wheel surface speed - forward speed| (m/s)."""
        v = self.car_vel_body[0]
        return float(np.mean([abs(self.data.qvel[d] * WHEEL_RADIUS - v) for d in self.spin_dofs]))

    def car_obs(self) -> list[float]:
        """Generic car state (16 values) most scenarios append."""
        h = self.heading
        return [*(self.car_vel_body / 20.0), *(self.car_angvel_body / 5.0), *self.gravity_body(),
                np.sin(h), np.cos(h), *self.wheel_contacts(), self.fuel_frac()]
