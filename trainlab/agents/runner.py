"""Load a trained run and drive an environment with it (used by watch, record, Studio)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from trainlab.agents.train import ALGOS, EnvFactory


class TrainedPolicy:
    def __init__(self, path: str | Path, prefer: str = "best"):
        path = Path(path)
        self.run_dir = path if path.is_dir() else path.parent
        self.meta = json.loads((self.run_dir / "run.json").read_text())
        if path.is_file():
            model_file = path
        else:
            best, final = self.run_dir / "best_model.zip", self.run_dir / "final_model.zip"
            model_file = best if (prefer == "best" and best.exists()) else final
        self.model = ALGOS[self.meta["algo"]].load(str(model_file), device="cpu")
        # each saved model has its own input-scaling stats; mismatched stats make a good policy look bad
        if model_file.stem == "best_model":
            stats = self.run_dir / "vecnormalize_best.pkl"
        elif model_file.stem.endswith("_model") and model_file.stem.startswith("level"):
            stats = self.run_dir / f"{model_file.stem[:-len('_model')]}_vecnormalize.pkl"  # levelN snapshot
        else:
            stats = self.run_dir / "vecnormalize.pkl"
        if not stats.exists():
            stats = self.run_dir / "vecnormalize.pkl"
        self.vecnorm = None
        if stats.exists():
            dummy = DummyVecEnv([EnvFactory(self.meta["scenario"], 0, discrete=self.meta["discrete"])])
            self.vecnorm = VecNormalize.load(str(stats), dummy)
            self.vecnorm.training = False
        self.model_file = model_file

    @property
    def scenario(self) -> str:
        return self.meta["scenario"]

    @property
    def level(self) -> int:
        return int(self.meta.get("final_level", self.meta.get("level", 0)))

    def make_env(self, render_mode=None, level=None, seed=0):
        return EnvFactory(self.scenario, seed, self.level if level is None else level,
                          self.meta.get("wind"), self.meta["discrete"], render_mode)()

    def act(self, obs: np.ndarray):
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs[None])[0]
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def run_episodes(policy: TrainedPolicy, env, episodes: int = 3, on_step=None, seed: int = 0) -> list[dict]:
    """Play episodes; ``on_step(env, info)`` is called after every step. Returns per-episode stats."""
    results = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, done, info, steps = 0.0, False, {}, 0
        while not done:
            obs, r, term, trunc, info = env.step(policy.act(obs))
            total += r
            steps += 1
            done = term or trunc
            if on_step is not None and on_step(env, info) is False:
                return results
        results.append({"episode": ep, "return": round(total, 2), "steps": steps,
                        "success": bool(info.get("is_success")), "reason": info.get("reason", "")})
    return results
