"""Gusty wind: a steady breeze plus Ornstein-Uhlenbeck gusts, horizontal only."""

from __future__ import annotations

import numpy as np


class Wind:
    def __init__(self, gust_time: float = 1.5):
        self.gust_time = gust_time
        self.strength = 0.0
        self.mean = np.zeros(3)
        self.gust = np.zeros(3)

    def reset(self, rng: np.random.Generator, strength: float) -> None:
        self.strength = float(strength)
        angle = rng.uniform(0, 2 * np.pi)
        self.mean = 0.5 * self.strength * np.array([np.cos(angle), np.sin(angle), 0.0])
        self.gust = np.zeros(3)

    def step(self, dt: float, rng: np.random.Generator) -> np.ndarray:
        """Advance the gust process and return the wind velocity (m/s)."""
        if self.strength <= 0:
            return np.zeros(3)
        theta = 1.0 / self.gust_time
        sigma = 0.5 * self.strength * np.sqrt(2 * theta)
        noise = rng.normal(size=3) * np.array([1.0, 1.0, 0.0])
        self.gust += -theta * self.gust * dt + sigma * np.sqrt(dt) * noise
        return self.mean + self.gust
