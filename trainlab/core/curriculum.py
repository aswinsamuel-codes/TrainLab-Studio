"""Training callbacks: difficulty curriculum, VecNormalize snapshots, stop-file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """Run after each EvalCallback evaluation; raise the level when the agent is good enough."""

    def __init__(self, train_env, eval_env, start_level: int, max_level: int,
                 threshold: float = 0.8, run_dir: Path | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.train_env, self.eval_env = train_env, eval_env
        self.level, self.max_level, self.threshold = start_level, max_level, threshold
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        buf = getattr(self.parent, "_is_success_buffer", [])
        rate = float(np.mean(buf)) if len(buf) else 0.0
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/success_rate", rate)
        if rate >= self.threshold and self.level < self.max_level:
            if self.run_dir is not None:
                self.model.save(self.run_dir / f"level{self.level}_model")
                venv = self.model.get_vec_normalize_env()
                if venv is not None:  # the snapshot needs its own input-scaling stats to replay correctly
                    venv.save(str(self.run_dir / f"level{self.level}_vecnormalize.pkl"))
            self.level += 1
            for env in (self.train_env, self.eval_env):
                env.env_method("set_level", self.level)
            # A new level is a new game: let the best-model tracker start over.
            self.parent.best_mean_reward = -np.inf
            if self.verbose:
                print(f"[curriculum] success {rate:.0%} -> level {self.level}/{self.max_level}")
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Save normalization stats alongside best_model.zip so they always match."""

    def __init__(self, path: Path):
        super().__init__(0)
        self.path = path

    def _on_step(self) -> bool:
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.path))
        return True


class AutosaveCallback(BaseCallback):
    """Save model + normalization + progress every ``every`` steps into <run>/autosave/.

    Shutting the laptop down mid-run then loses at most ``every`` steps:
    ``trainlab resume`` picks up from here.
    """

    def __init__(self, run_dir: Path, every: int = 20_000):
        super().__init__(0)
        self.dir = Path(run_dir) / "autosave"
        self.every = every
        self.next_save = every

    def _on_training_start(self) -> None:
        self.next_save = self.num_timesteps + self.every  # also right after a resume

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.save_now()
            self.next_save = self.num_timesteps + self.every
        return True

    def save_now(self) -> None:
        import json
        import os
        import time

        self.dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.dir / "model_tmp"))
        os.replace(self.dir / "model_tmp.zip", self.dir / "model.zip")  # atomic: never a half-written model
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            venv.save(str(self.dir / "vecnormalize_tmp.pkl"))
            os.replace(self.dir / "vecnormalize_tmp.pkl", self.dir / "vecnormalize.pkl")
        try:
            level = int(self.training_env.env_method("get_level", indices=[0])[0])
        except Exception:
            level = 0
        state = {"timesteps": int(self.num_timesteps), "level": level, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.dir / "state_tmp.json").write_text(json.dumps(state))
        os.replace(self.dir / "state_tmp.json", self.dir / "state.json")


class StopFileCallback(BaseCallback):
    """Stop training cleanly when a file named STOP appears in the run folder (used by Studio)."""

    def __init__(self, run_dir: Path):
        super().__init__(0)
        self.flag = run_dir / "STOP"

    def _on_step(self) -> bool:
        return not self.flag.exists()
