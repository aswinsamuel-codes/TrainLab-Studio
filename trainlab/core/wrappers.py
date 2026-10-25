"""Wrappers shared by training scripts."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DiscretizeAction(gym.ActionWrapper):
    """Turn a continuous Box(-1, 1, n) into Discrete(1 + 2n) so DQN can play.

    Action 0 does nothing; action 2i+1 pushes channel i to +1 and 2i+2 to -1.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = env.action_space.shape[0]
        table = [np.zeros(n, dtype=np.float32)]
        for i in range(n):
            for sign in (1.0, -1.0):
                a = np.zeros(n, dtype=np.float32)
                a[i] = sign
                table.append(a)
        self.table = np.stack(table)
        self.action_space = gym.spaces.Discrete(len(table))

    def action(self, act):
        return self.table[int(act)]
