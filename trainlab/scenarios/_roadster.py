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


def roadster_xml(world: str, name: str = "roadster", gravity: str = "0 0 -9.81",
                 start=(0.0, 0.0, 0.7), quat=(1.0, 0.0, 0.0, 0.0), extra: str = "") -> str:
    return Template(ASSET.read_text()).substitute(
        name=name, gravity=gravity, world=world, extra=extra,
        start=" ".join(f"{v:g}" for v in start), quat=" ".join(f"{v:g}" for v in quat))


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
