"""Scenario registry.

Every scenario class is decorated with ``@register_scenario("snake_name")``. That
stores it here and registers it with Gymnasium as ``TrainLab/<CamelName>-v0`` so
``gymnasium.make`` works too.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
import warnings
from pathlib import Path

import gymnasium as gym

_SCENARIOS: dict[str, type] = {}
_LOADED = False

GENERATED_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "generated"


def camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def env_id(name: str) -> str:
    return f"TrainLab/{camel(name)}-v0"


def register_scenario(name: str):
    """Class decorator that makes a MujocoScenarioEnv subclass available everywhere."""

    def deco(cls):
        cls.scenario_name = name
        _SCENARIOS[name] = cls
        gid = env_id(name)
        if gid in gym.registry:
            del gym.registry[gid]
        gym.register(id=gid, entry_point=cls, max_episode_steps=cls.max_episode_steps())
        return cls

    return deco


def load_all(reload_generated: bool = False) -> None:
    """Import built-in scenarios and every generated scenario folder."""
    global _LOADED
    if _LOADED and not reload_generated:
        return
    import trainlab.scenarios as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if not mod.name.startswith("_") and mod.name != "generated":
            importlib.import_module(f"trainlab.scenarios.{mod.name}")

    if GENERATED_DIR.exists():
        for folder in sorted(GENERATED_DIR.iterdir()):
            script = folder / "scenario.py"
            if script.exists() and not folder.name.startswith("_"):
                try:
                    load_generated(folder)
                except Exception as e:  # one broken generated scenario must not break the rest
                    warnings.warn(f"Skipping generated scenario '{folder.name}': {e}")
    _LOADED = True


def load_generated(folder: Path) -> None:
    modname = f"trainlab_generated_{folder.name}"
    spec = importlib.util.spec_from_file_location(modname, folder / "scenario.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)


def get_scenario(name: str) -> type:
    load_all()
    if name not in _SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. Available: {', '.join(sorted(_SCENARIOS))}")
    return _SCENARIOS[name]


def list_scenarios() -> list[str]:
    load_all()
    return sorted(_SCENARIOS)


def make(name: str, **kwargs) -> gym.Env:
    get_scenario(name)
    return gym.make(env_id(name), **kwargs)
