"""Reward-shaping helpers.

Potential-based shaping (Ng et al., 1999) is the trick Lunar Lander uses: define a
"how good is this state" score Φ(s) and reward the *change* in it each step,
r = Φ(s') - Φ(s). The agent gets dense feedback for moving the right way, but the
best policy is unchanged, so it can't farm the shaping term by wiggling.
"""

from __future__ import annotations


class PotentialShaper:
    def __init__(self):
        self.prev: float | None = None

    def reset(self, phi: float) -> None:
        self.prev = phi

    def __call__(self, phi: float) -> float:
        delta = 0.0 if self.prev is None else phi - self.prev
        self.prev = phi
        return delta


def braking_envelope(distance: float, decel: float, margin: float = 1.0) -> float:
    """Fastest speed from which you can still stop within ``distance`` metres."""
    return (2.0 * decel * max(distance, 0.0)) ** 0.5 + margin
