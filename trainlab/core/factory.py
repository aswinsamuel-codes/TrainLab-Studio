"""Picklable environment constructor.

Lives in a lightweight module on purpose: SubprocVecEnv workers (spawned fresh on
Windows) import this file, so it must not pull in torch / stable-baselines3.
"""

from __future__ import annotations


class EnvFactory:
    def __init__(self, scenario: str, seed: int, level: int = 0, wind: float | None = None,
                 discrete: bool = False, render_mode: str | None = None):
        self.scenario, self.seed, self.level, self.wind = scenario, seed, level, wind
        self.discrete, self.render_mode = discrete, render_mode

    def __call__(self):
        import trainlab
        from trainlab.core.wrappers import DiscretizeAction

        env = trainlab.make(self.scenario, level=self.level, wind=self.wind, render_mode=self.render_mode)
        if self.discrete:
            env = DiscretizeAction(env)
        env.reset(seed=self.seed)
        env.action_space.seed(self.seed)
        return env
