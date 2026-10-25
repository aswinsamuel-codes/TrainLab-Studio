"""Keep training laptop-friendly.

* ``profile()`` picks parallel envs / CPU threads from the machine's RAM and cores.
* ``be_nice()`` lowers this process's priority and caps PyTorch threads, so the laptop
  stays usable (browser, music, typing) while an agent trains in the background.
* ``MemoryGuardCallback`` pauses training when free RAM runs low instead of letting
  Windows freeze, and stops cleanly (with an autosave) if it doesn't recover.
"""

from __future__ import annotations

import os
import time

import psutil
from stable_baselines3.common.callbacks import BaseCallback

GB = 1024 ** 3


def total_ram_gb() -> float:
    return psutil.virtual_memory().total / GB


def free_ram_gb() -> float:
    return psutil.virtual_memory().available / GB


def profile() -> dict:
    """Training defaults for this machine. Override any value in configs/*.yaml or with CLI flags."""
    ram, cores = total_ram_gb(), psutil.cpu_count(logical=False) or 2
    if ram <= 9:  # 8 GB laptops (Windows itself + a browser already use 5-6 GB)
        # ~0.4-0.8 GB "available" is normal on a busy 8 GB Windows laptop; below 0.2 GB it starts to freeze.
        return {"name": "low (8 GB)", "n_envs": 2, "torch_threads": 2, "min_free_gb": 0.2}
    if ram <= 17:
        return {"name": "medium (16 GB)", "n_envs": 4, "torch_threads": min(4, cores), "min_free_gb": 0.8}
    return {"name": "high", "n_envs": min(8, cores), "torch_threads": max(2, cores // 2), "min_free_gb": 1.5}


def be_nice(torch_threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, torch_threads))
    try:
        proc = psutil.Process()
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            proc.nice(10)
    except (psutil.Error, OSError):
        pass


class MemoryGuardCallback(BaseCallback):
    """Pause when free RAM drops below ``min_free_gb``; stop safely if it stays low."""

    def __init__(self, autosave, min_free_gb: float = 0.5, check_every: int = 1_000,
                 max_wait_s: float = 600.0, verbose: int = 1):
        super().__init__(verbose)
        self.autosave, self.min_free, self.check_every, self.max_wait = autosave, min_free_gb, check_every, max_wait_s
        self.next_check = 0
        self.stopped_for_memory = False

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_check:
            return True
        self.next_check = self.num_timesteps + self.check_every
        if free_ram_gb() >= self.min_free:
            return True
        self.autosave.save_now()
        if self.verbose:
            print(f"[memory] only {free_ram_gb():.2f} GB RAM free - pausing training (progress saved). "
                  f"Close some apps to continue.", flush=True)
        start = time.time()
        while time.time() - start < self.max_wait:
            time.sleep(5)
            if free_ram_gb() >= self.min_free + 0.15:
                if self.verbose:
                    print("[memory] RAM recovered - continuing.", flush=True)
                return True
        self.stopped_for_memory = True
        print("[memory] RAM stayed low for 10 minutes - stopped to protect the laptop. "
              "Continue later with:  trainlab resume", flush=True)
        return False
