"""Six-wheel rover (Mars-rover style, about 60 kg), the third body after the Roadster and the drone.

Controls (in [-1, 1]):
    drive  forward (+) / reverse (-)
    turn   left (+) / right (-)  (skid steering: left and right wheels at different speeds)
Each wheel has spring suspension and a speed-controlled motor (like real rover motor drivers),
so the AI chooses *where* to go, not raw wheel torques.

Sensors: 7 "laser eyes" (rangefinder beams fanned across the front, 120 degrees, 4 m) so it
can see rocks and steer around them.
"""

from __future__ import annotations

import numpy as np

from trainlab.bodies.base import Body, LaserEyes

WHEEL_R = 0.14
V_MAX = 2.5  # m/s top speed
TORQUE_MAX = 40.0  # N·m per wheel
K_SPEED = 6.0  # wheel speed controller gain (N·m per rad/s)
# wheel -> (x, y): three per side
WHEELS = {"fl": (0.38, 0.40), "ml": (0.0, 0.42), "rl": (-0.38, 0.40),
          "fr": (0.38, -0.40), "mr": (0.0, -0.42), "rr": (-0.38, -0.40)}


class RoverBody(Body):
    name = "rover"
    root = "rover"
    CONTROLS = ("drive", "turn")
    drag_coef = 0.2
    extent = 15.0
    flies = False
    spawn_z = 0.34
    reach = 0.6

    def __init__(self):
        self.eyes = LaserEyes(n=7, spread_deg=120, max_dist=4.0, origin=(0.47, 0.0, 0.02), name="rover_laser")

    def xml(self, start=(0.0, 0.0, 0.34)) -> dict:
        s = " ".join(f"{v:g}" for v in start)
        wheels, acts = [], []
        for k, (x, y) in WHEELS.items():
            wheels.append(f"""
  <body name="rmount_{k}" pos="{x} {y} -0.17">
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <joint name="rsusp_{k}" type="slide" axis="0 0 1" range="-0.06 0.06" stiffness="9000" damping="450"/>
    <body name="rwheel_{k}">
      <joint name="rspin_{k}" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="rwheel_{k}" type="cylinder" size="{WHEEL_R} 0.06" zaxis="0 1 0" mass="2.5" margin="0.01" gap="0.01" material="mat_rtyre"/>
      <geom class="deco" type="cylinder" size="0.09 0.062" zaxis="0 1 0" material="mat_metal"/>
      <geom class="deco" type="box" size="0.135 0.064 0.012" material="mat_rtyre"/>
      <geom class="deco" type="box" size="0.135 0.064 0.012" euler="0 1.047 0" material="mat_rtyre"/>
      <geom class="deco" type="box" size="0.135 0.064 0.012" euler="0 2.094 0" material="mat_rtyre"/>
    </body>
  </body>""")
            acts.append(f'<motor name="rdrive_{k}" joint="rspin_{k}" ctrlrange="-{TORQUE_MAX:g} {TORQUE_MAX:g}"/>')
        body = f"""
<body name="rover" pos="{s}" childclass="rover">
  <freejoint name="rover_root"/>
  <camera name="chase" mode="trackcom" pos="-3.2 -1.7 1.6" xyaxes="0.469 -0.883 0 0.357 0.189 0.914"/>
  <camera name="side" mode="trackcom" pos="0 -4 0.8" xyaxes="1 0 0 0 0.2 0.98"/>
  <camera name="driver" mode="fixed" pos="0.32 0 0.45" xyaxes="0 -1 0 0.1 0 0.99"/>
  <geom name="rover_chassis" type="box" size="0.45 0.3 0.08" mass="40" material="mat_rbody"/>
  <geom class="deco" type="box" size="0.5 0.36 0.008" pos="-0.05 0 0.13" material="mat_solar"/>
  <geom class="deco" type="box" size="0.02 0.02 0.05" pos="-0.05 0 0.09" material="mat_metal"/>
  <geom class="deco" type="cylinder" size="0.02 0.16" pos="0.3 0 0.24" material="mat_metal"/>
  <geom class="deco" type="box" size="0.05 0.09 0.045" pos="0.32 0 0.42" material="mat_rbody"/>
  <geom class="deco" type="sphere" size="0.022" pos="0.37 0.04 0.42" material="mat_lens"/>
  <geom class="deco" type="sphere" size="0.022" pos="0.37 -0.04 0.42" material="mat_lens"/>
  <geom class="deco" type="cylinder" size="0.005 0.12" pos="-0.35 0.2 0.25" material="mat_metal"/>
  <geom class="deco" type="sphere" size="0.012" pos="-0.35 0.2 0.37" material="mat_led_rear"/>
  <geom class="deco" type="box" size="0.03 0.2 0.03" pos="0.46 0 -0.02" material="mat_metal"/>
  <geom class="deco" type="sphere" size="0.015" pos="0.48 0 0.02" material="mat_laser"/>
  {''.join(wheels)}
</body>"""
        return {
            "assets": """
    <material name="mat_rbody" rgba="0.85 0.82 0.74 1" specular="0.4"/>
    <material name="mat_solar" rgba="0.12 0.2 0.42 1" specular="0.9" shininess="0.9"/>
    <material name="mat_rtyre" rgba="0.18 0.18 0.19 1"/>
    <material name="mat_lens" rgba="0.1 0.3 0.6 1" specular="1" shininess="1"/>
    <material name="mat_led_rear" rgba="1 0.15 0.1 1" emission="1"/>
    <material name="mat_laser" rgba="1 0.2 0.1 1" emission="1"/>""",
            "defaults": """
    <default class="rover">
      <geom contype="2" conaffinity="1" friction="1.1 0.01 0.001" group="1"/>
    </default>""",
            "body": body, "actuators": "\n    ".join(acts), "world_extra": self.eyes.world_xml(),
        }

    # ------------------------------------------------------------- binding
    def bind(self, env) -> None:
        super().bind(env)
        m = env.model
        self.wheel_geoms = {env.geom(f"rwheel_{k}") for k in WHEELS}
        self.shell_geoms = self.geoms - self.wheel_geoms
        self.drive_ids = [env.actuator(f"rdrive_{k}") for k in WHEELS]
        self.spin_dofs = [m.jnt_dofadr[env.joint(f"rspin_{k}")] for k in WHEELS]
        self.side = np.array([1.0 if y > 0 else -1.0 for _, y in WHEELS.values()])  # +1 = left
        self.eyes.bind(env)
        self.cmd = np.zeros(2)
        self.torques = np.zeros(6)

    def reset_state(self) -> None:
        self.cmd = np.zeros(2)
        self.torques = np.zeros(6)

    # ------------------------------------------------------------- control
    def apply_action(self, action: np.ndarray) -> None:
        self.cmd = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.substep_control()

    def substep_control(self) -> None:
        """Wheel-speed controller with skid steering (left = drive - turn, right = drive + turn)."""
        drive, turn = self.cmd
        target = np.clip(drive - self.side * turn, -1.0, 1.0) * V_MAX / WHEEL_R
        omega = self.env.data.qvel[self.spin_dofs]
        self.torques = np.clip(K_SPEED * (target - omega), -TORQUE_MAX, TORQUE_MAX)
        self.env.data.ctrl[self.drive_ids] = self.torques

    # ---------------------------------------------------------------- state
    def obs(self) -> list[float]:
        h = self.heading
        lasers = self.eyes.read(self.pos, self.rot)
        wheels_down = len({mine for mine, _ in self.world_contacts() if mine in self.wheel_geoms}) / 6.0
        return [*(self.vel_body / 3.0), *(self.angvel_body / 3.0), *self.gravity_body(),
                float(np.sin(h)), float(np.cos(h)), wheels_down, *lasers]

    def crash_reason(self) -> str:
        if self.up_z < 0.5:
            return "flipped"
        if any(mine in self.shell_geoms for mine, _ in self.world_contacts()):
            return "bumped"
        return ""

    def effort(self) -> float:
        return float(np.abs(self.torques).mean() / TORQUE_MAX)

    def update_visuals(self, dt: float) -> None:
        if self.env.render_mode is not None:
            self.eyes.draw()
