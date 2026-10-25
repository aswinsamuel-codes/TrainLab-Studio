"""Task = world + goal + reward. ``BodyTaskEnv`` combines any Body with any Task into a
normal TrainLab scenario (all CLI commands, Studio, live view, resume... just work).

Subclass ``Task`` and fill in:
    world_xml()             -> MJCF fragment placed in <worldbody>
    bind(env)               cache ids
    randomize(params)       build this episode (use self.env.rng) and place the body
    obs()                   task observations (list of floats)
    check_done()            -> (terminated, success, reason)
    compute_reward(action, terminated, success) -> (reward, {term: value})
Optional: substep(t) (moving platforms...), update_visuals(dt).
"""

from __future__ import annotations

from pathlib import Path
from string import Template

import numpy as np

from trainlab.core.base_env import MujocoScenarioEnv

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "world_template.xml"


class Task:
    name = "task"
    episode_seconds = 20.0
    gravity = "0 0 -9.81"
    CURRICULUM: list[dict] = [{}]
    start = (0.0, 0.0, 1.0)  # where the body sits in the compiled model (moved at reset)

    def world_xml(self) -> str:
        return '<geom name="floor" class="world" type="plane" size="200 200 0.1" material="mat_grid"/>'

    def actuators_xml(self) -> str:
        """Actuators the task itself drives (moving platforms...), appended after the body's."""
        return ""

    def bind(self, env) -> None:
        self.env = env
        self.body = env.body_obj

    def span(self, key: str) -> tuple[float, float]:
        """The raw (low, high) range of a curriculum value at the current level.

        ``randomize`` receives each range already sampled once per episode; use this
        when you need several independent draws (e.g. one per waypoint)."""
        v = self.env.CURRICULUM[self.env.level][key]  # the env's table (may be body-specific)
        return (float(v[0]), float(v[1])) if isinstance(v, (list, tuple)) else (float(v), float(v))

    def randomize(self, p: dict) -> None:
        raise NotImplementedError

    def obs(self) -> list[float]:
        return []

    def check_done(self) -> tuple[bool, bool, str]:
        return False, False, ""

    def compute_reward(self, action, terminated: bool, success: bool) -> tuple[float, dict]:
        raise NotImplementedError

    def substep(self, t: float) -> None:
        pass

    def update_visuals(self, dt: float) -> None:
        pass

    # Tasks can adapt to the body (flying vs. ground...). Defaults: one table, fixed start.
    @classmethod
    def curriculum_for(cls, body_cls) -> list[dict]:
        return cls.CURRICULUM

    @classmethod
    def start_for(cls, body_cls) -> tuple:
        return cls.start


def compose_xml(body, task, name: str) -> str:
    parts = body.xml(start=task.start_for(type(body)))
    return Template(TEMPLATE.read_text()).substitute(
        name=name, gravity=task.gravity, extent=f"{body.extent:g}",
        assets=parts.get("assets", ""), defaults=parts.get("defaults", ""),
        world=task.world_xml() + "\n" + parts.get("world_extra", ""),
        body=parts["body"], actuators=parts.get("actuators", "") + "\n" + task.actuators_xml(),
        extra=parts.get("extra", ""))


class BodyTaskEnv(MujocoScenarioEnv):
    BODY: type = None
    TASK: type = None

    def __init__(self, render_mode=None, level: int = 0, wind=None, width: int = 640, height: int = 480):
        self.body_obj = self.BODY()
        self.task = self.TASK()
        super().__init__(render_mode=render_mode, level=level, wind=wind, width=width, height=height)

    def build_xml(self) -> str:
        return compose_xml(self.body_obj, self.task, self.scenario_name)

    def setup(self) -> None:
        self.body_obj.bind(self)
        self.task.bind(self)

    def action_dim(self) -> int:
        return len(self.BODY.CONTROLS)

    def apply_action(self, action) -> None:
        self.body_obj.apply_action(action)

    def substep_control(self) -> None:
        self.task.substep(float(self.data.time))
        self.body_obj.substep_control()

    def randomize(self, p: dict) -> None:
        self.body_obj.reset_state()
        self.task.randomize(p)

    def get_obs(self) -> np.ndarray:
        return np.array(self.body_obj.obs() + self.task.obs(), dtype=np.float32)

    def check_done(self):
        return self.task.check_done()

    def compute_reward(self, action, terminated, success):
        return self.task.compute_reward(action, terminated, success)

    def update_visuals(self) -> None:
        self.body_obj.update_visuals(self.dt)
        self.task.update_visuals(self.dt)

    @property
    def car_vel(self) -> np.ndarray:  # speed readout used by HUDs, live view and play
        return self.body_obj.vel


def body_task_env(body_cls, task_cls) -> type:
    """Build the scenario class for one body + task pair (register it with @register_scenario)."""
    return type(f"{body_cls.__name__}{task_cls.__name__}", (BodyTaskEnv,), {
        "BODY": body_cls, "TASK": task_cls,
        "CONTROLS": body_cls.CONTROLS, "CURRICULUM": task_cls.curriculum_for(body_cls),
        "episode_seconds": task_cls.episode_seconds,
        "wind_body": body_cls.root, "drag_coef": body_cls.drag_coef,
        "__doc__": f"{body_cls.name} body + {task_cls.name} task.",
    })
