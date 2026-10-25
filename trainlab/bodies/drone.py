"""Quadcopter drone (about 1.2 kg, X frame, 4 motors), the second body after the Roadster.

Controls (all in [-1, 1]), "angle mode" like a real flight controller:
    thrust  climb (+) / sink (-); 0 is roughly hover
    roll    bank left/right (target tilt, up to ~26 degrees)
    pitch   nose down = fly forward (target tilt)
    yaw     turn rate
A built-in stabilizer (runs every physics step, 200 Hz) turns these into 4 motor thrusts,
so the AI learns *where* to fly, not how to keep 4 propellers balanced. That's what makes
it trainable in minutes on a laptop CPU.
"""

from __future__ import annotations

import numpy as np

from trainlab.bodies.base import Body, ray_down

ARM = 0.13  # motor distance along x and y (m)
F_MAX = 7.0  # max thrust per motor (N): thrust-to-weight about 2.4
K_YAW = 0.016  # propeller drag torque per newton of thrust
TILT_MAX = 0.45  # rad, about 26 degrees
YAW_RATE_MAX = 2.5  # rad/s
# motor -> (x, y, spin): +1 = counter-clockwise seen from above
MOTORS = {"fl": (ARM, ARM, 1), "fr": (ARM, -ARM, -1), "rl": (-ARM, ARM, -1), "rr": (-ARM, -ARM, 1)}
N_DUST = 24


class DroneBody(Body):
    name = "drone"
    root = "drone"
    CONTROLS = ("thrust", "roll", "pitch", "yaw")
    drag_coef = 0.03
    extent = 12.0
    flies = True
    spawn_z = 0.15
    reach = 0.45

    def xml(self, start=(0.0, 0.0, 1.0)) -> dict:
        s = " ".join(f"{v:g}" for v in start)
        arms, motors, props, legs, sites, acts = [], [], [], [], [], []
        for k, (x, y, spin) in MOTORS.items():
            arms.append(f'<geom name="drone_arm_{k}" type="capsule" fromto="0 0 0 {x} {y} 0" size="0.008" '
                        f'mass="0.03" material="mat_dark"/>')
            motors.append(f'<geom name="drone_motor_{k}" type="cylinder" pos="{x} {y} 0.008" size="0.017 0.014" '
                          f'mass="0.05" material="mat_metal"/>')
            props.append(f'<geom name="prop_disc_{k}" class="deco" type="cylinder" pos="{x} {y} 0.03" '
                         f'size="0.066 0.0015" material="mat_propdisc"/>')
            props.append(f'<geom name="prop_blade_{k}" class="deco" type="box" pos="{x} {y} 0.032" '
                         f'size="0.066 0.008 0.0025" material="mat_blade"/>')
            lx, ly = 0.45 * x, 0.45 * y
            legs.append(f'<geom name="drone_leg_{k}" type="capsule" fromto="{lx} {ly} -0.025 {1.5 * lx} {1.5 * ly} -0.1" '
                        f'size="0.006" mass="0.01" material="mat_dark"/>')
            sites.append(f'<site name="rotor_{k}" pos="{x} {y} 0.03" size="0.005"/>')
            acts.append(f'<general name="motor_{k}" site="rotor_{k}" gear="0 0 1 0 0 {-spin * K_YAW:g}" '
                        f'ctrlrange="0 {F_MAX:g}"/>')
        body = f"""
<body name="drone" pos="{s}" childclass="drone">
  <freejoint name="drone_root"/>
  <camera name="chase" mode="trackcom" pos="-2.2 -1.2 1.0" xyaxes="0.478 -0.878 0 0.325 0.177 0.928"/>
  <camera name="side" mode="trackcom" pos="0 -3 0.4" xyaxes="1 0 0 0 0.13 0.99"/>
  <camera name="fpv" mode="fixed" pos="0.09 0 -0.01" xyaxes="0 -1 0 0.15 0 0.99"/>
  <geom name="drone_hull" type="box" size="0.07 0.05 0.022" mass="0.55" material="mat_drone"/>
  <geom name="drone_canopy" class="deco" type="ellipsoid" size="0.075 0.052 0.03" pos="0 0 0.012" material="mat_drone"/>
  <geom name="drone_battery" type="box" size="0.05 0.035 0.016" pos="0 0 -0.038" mass="0.25" material="mat_dark"/>
  <geom class="deco" type="sphere" size="0.014" pos="0.075 0 -0.012" material="mat_dark"/>
  <geom class="deco" type="sphere" size="0.007" pos="0.087 0 -0.012" material="mat_lens"/>
  <geom class="deco" type="sphere" size="0.007" pos="0.06 0.045 0.0" material="mat_led_front"/>
  <geom class="deco" type="sphere" size="0.007" pos="0.06 -0.045 0.0" material="mat_led_front"/>
  <geom class="deco" type="sphere" size="0.007" pos="-0.06 0.045 0.0" material="mat_led_rear"/>
  <geom class="deco" type="sphere" size="0.007" pos="-0.06 -0.045 0.0" material="mat_led_rear"/>
  {''.join(arms)}
  {''.join(motors)}
  {''.join(legs)}
  {''.join(props)}
  {''.join(sites)}
</body>"""
        dust = "\n".join(f'<body name="dust{i}" mocap="true" pos="{s}"><geom class="deco" type="sphere" size="0.1" '
                         f'material="mat_dust"/></body>' for i in range(N_DUST))
        return {
            "assets": """
    <material name="mat_drone" rgba="0.92 0.93 0.95 1" specular="0.6" shininess="0.6"/>
    <material name="mat_propdisc" rgba="0.15 0.15 0.17 0" specular="0"/>
    <material name="mat_blade" rgba="0.12 0.12 0.13 1"/>
    <material name="mat_lens" rgba="0.1 0.3 0.6 1" specular="1" shininess="1"/>
    <material name="mat_led_front" rgba="0.2 1 0.3 1" emission="1"/>
    <material name="mat_led_rear" rgba="1 0.15 0.1 1" emission="1"/>""",
            "defaults": """
    <default class="drone">
      <geom contype="2" conaffinity="1" friction="0.9 0.005 0.0001" group="1"/>
      <site group="3"/>
    </default>""",
            "body": body, "actuators": "\n    ".join(acts), "world_extra": dust,
        }

    # ------------------------------------------------------------- binding
    def bind(self, env) -> None:
        super().bind(env)
        m = env.model
        self.motor_ids = [env.actuator(f"motor_{k}") for k in MOTORS]
        self.leg_geoms = {env.geom(f"drone_leg_{k}") for k in MOTORS}
        self.frame_geoms = self.geoms - self.leg_geoms
        # Mixer: [total thrust, roll torque, pitch torque, yaw torque] = M @ motor thrusts
        M = np.array([[1.0] * 4,
                      [y for x, y, s in MOTORS.values()],
                      [-x for x, y, s in MOTORS.values()],
                      [-s * K_YAW for x, y, s in MOTORS.values()]])
        self.unmix = np.linalg.inv(M)
        self.inertia = m.body_inertia[self.root_id].copy()
        self.cmd = np.zeros(4)
        self.thrusts = np.zeros(4)
        self.fx = DroneFX(self) if env.render_mode is not None else None

    def reset_state(self) -> None:
        self.cmd = np.zeros(4)
        self.thrusts = np.zeros(4)
        if self.fx is not None:
            self.fx.reset()

    # ------------------------------------------------------------- control
    def apply_action(self, action: np.ndarray) -> None:
        self.cmd = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.substep_control()

    def attitude(self) -> tuple[float, float]:
        R = self.rot
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
        return roll, pitch

    def substep_control(self) -> None:
        """Angle-mode stabilizer + motor mixer (like a real flight controller)."""
        thrust, roll_c, pitch_c, yaw_c = self.cmd
        roll, pitch = self.attitude()
        p, q, r = self.angvel_body
        wn, zeta = 14.0, 0.8  # attitude loop: fast and well damped
        kp, kd = wn * wn, 2 * zeta * wn
        Ix, Iy, Iz = self.inertia
        tau_x = Ix * (kp * (TILT_MAX * roll_c - roll) - kd * p)
        tau_y = Iy * (kp * (TILT_MAX * pitch_c - pitch) - kd * q)
        tau_z = Iz * 8.0 * (YAW_RATE_MAX * yaw_c - r)
        tilt_comp = 1.0 / max(np.cos(roll) * np.cos(pitch), 0.6)  # keep height while tilting
        total = self.mass * 9.81 * (1.0 + thrust) * tilt_comp
        f = self.unmix @ np.array([total, tau_x, tau_y, tau_z])
        self.thrusts = np.clip(f, 0.0, F_MAX)
        self.env.data.ctrl[self.motor_ids] = self.thrusts

    # ---------------------------------------------------------------- state
    def obs(self) -> list[float]:
        h = self.heading
        return [*(self.vel_body / 5.0), *(self.angvel_body / 5.0), *self.gravity_body(),
                float(np.sin(h)), float(np.cos(h))]

    def legs_touching(self, world_geoms: set[int] | None = None) -> int:
        """How many landing legs touch the world (optionally only the given world geoms)."""
        legs = set()
        for mine, other in self.world_contacts():
            if mine in self.leg_geoms and (world_geoms is None or other in world_geoms):
                legs.add(mine)
        return len(legs)

    def frame_hit(self) -> bool:
        """True if anything but the legs (props, arms, hull) touches the world."""
        return any(mine in self.frame_geoms for mine, _ in self.world_contacts())

    def crash_reason(self) -> str:
        if self.up_z < 0.3:
            return "flipped"
        if self.frame_hit():
            return "crashed"
        return ""

    def effort(self) -> float:
        return float(self.thrusts.sum() / (4 * F_MAX))

    def update_visuals(self, dt: float) -> None:
        if self.fx is not None:
            self.fx.update(dt)


class DroneFX:
    """Spinning props + downwash dust near the ground. Render-only: costs nothing in training."""

    def __init__(self, drone: DroneBody):
        env = drone.env
        self.d = drone
        self.blades = [env.geom(f"prop_blade_{k}") for k in MOTORS]
        self.discs = [env.geom(f"prop_disc_{k}") for k in MOTORS]
        self.spins = [s for _, _, s in MOTORS.values()]
        self.dust_mocap = [env.model.body_mocapid[env.body(f"dust{i}")] for i in range(N_DUST)]
        self.dust_geom = [env.model.body_geomadr[env.body(f"dust{i}")] for i in range(N_DUST)]
        self.rng = np.random.default_rng(2)
        self.reset()

    def reset(self) -> None:
        self.angle = np.zeros(4)
        self.pos = np.zeros((N_DUST, 3))
        self.vel = np.zeros((N_DUST, 3))
        self.age = np.full(N_DUST, np.inf)
        self.next = 0
        for g in self.dust_geom:
            self.d.env.model.geom_rgba[g][3] = 0.0

    def update(self, dt: float) -> None:
        m, d = self.d.env.model, self.d.env.data
        level = self.d.thrusts / F_MAX
        for i, (blade, disc) in enumerate(zip(self.blades, self.discs)):
            self.angle[i] += self.spins[i] * (8 + 120 * level[i]) * dt * 7.3  # visibly spinning, not aliased
            a = self.angle[i]
            m.geom_quat[blade] = (np.cos(a / 2), 0.0, 0.0, np.sin(a / 2))
            m.geom_rgba[disc][3] = 0.15 + 0.35 * level[i]  # motion-blur disc
            m.geom_rgba[blade][3] = 1.0 - 0.7 * level[i]
        # downwash dust when close to the ground
        h = ray_down(self.d.env, self.d.pos, 2.5)
        push = float(level.mean())
        if 0 <= h < 2.0 and push > 0.2 and self.rng.random() < 0.9:
            ground = self.d.pos - np.array([0.0, 0.0, h - 0.05])
            for _ in range(2):
                i = self.next
                self.next = (self.next + 1) % N_DUST
                ang = self.rng.uniform(0, 2 * np.pi)
                out = np.array([np.cos(ang), np.sin(ang), 0.0])
                self.pos[i] = ground + out * 0.15
                self.vel[i] = out * (1.6 * push) * (1.6 - h / 2.0) + np.array([0, 0, 0.15])
                self.age[i] = 0.0
        live = self.age < 1.2
        self.age[live] += dt
        self.vel[live] *= np.exp(-2.0 * dt)
        self.pos[live] += self.vel[live] * dt
        for i in range(N_DUST):
            g = self.dust_geom[i]
            if self.age[i] >= 1.2:
                m.geom_rgba[g][3] = 0.0
                continue
            t = self.age[i] / 1.2
            m.geom_size[g][0] = 0.04 + 0.16 * t  # drone-sized puffs (it's a 30 cm aircraft)
            m.geom_rgba[g][3] = 0.25 * (1 - t) ** 1.5
            d.mocap_pos[self.dust_mocap[i]] = self.pos[i]
