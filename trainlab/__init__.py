"""TrainLab: train RL agents on any 3D MuJoCo simulation."""

from trainlab.core.registry import list_scenarios, make, register_scenario

__all__ = ["make", "list_scenarios", "register_scenario"]
__version__ = "0.1.0"
