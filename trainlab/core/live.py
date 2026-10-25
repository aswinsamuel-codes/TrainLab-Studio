"""Live trial-and-error view while training.

Every ``every_steps`` the current (still learning) policy plays one episode in a
rendered copy of the environment, and the clip is saved with its outcome, so you can
watch the agent go from crashing to winning. Every finished training episode is also
tallied by outcome (parked, fell, flipped, time up, ...).

``SpectatorCallback`` is the real-time version: a spare copy of the environment where the
current policy plays non-stop, a few frames per second, written to live/frame.jpg (Studio
shows it) and/or shown in MuJoCo's interactive 3D window.

Files in <run>/live/:
    attempt_0003.gif / .png    the clip and its final frame
    attempts.json              one entry per recorded attempt
    stats.json                 win/fail counts by reason + success-rate timeline
    frame.jpg, spectator.json  the latest real-time frame and its status
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from trainlab.core.factory import EnvFactory

WIN, FAIL = (40, 170, 90), (215, 60, 60)


def _hud(frame: np.ndarray, top: str, banner: str | None = None, ok: bool = False) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 20], fill=(0, 0, 0, 160))
    draw.text((8, 4), top, fill=(255, 255, 255, 255))
    if banner:
        h = 36
        y = img.height // 2 - h // 2
        draw.rectangle([0, y, img.width, y + h], fill=(*(WIN if ok else FAIL), 210))
        draw.text((img.width // 2 - 4 * len(banner), y + 12), banner, fill=(255, 255, 255, 255))
    return np.asarray(img)


def save_small_gif(frames: list[np.ndarray], path: Path, frame_ms: float, max_frames: int = 120,
                   colors: int = 128) -> None:
    """Browser-friendly GIF: at most ``max_frames`` frames (sped up to keep real time) and one shared
    palette, so a 30 s attempt stays around a megabyte instead of ~10 MB."""
    from PIL import Image

    step = max(1, int(np.ceil(len(frames) / max_frames)))
    keep = frames[::step]
    if len(frames) > 1 and keep[-1] is not frames[-1]:
        keep.append(frames[-1])
    imgs = [Image.fromarray(f) for f in keep]
    palette = imgs[len(imgs) // 2].quantize(colors=colors, method=Image.Quantize.FASTOCTREE)
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    durations = [int(frame_ms * step)] * len(pal)
    durations[-1] = 1500  # hold the verdict
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)


def _replace(tmp: Path, target: Path) -> None:
    import os

    try:
        os.replace(tmp, target)
    except PermissionError:  # Studio is reading the old file right now; next frame will do
        tmp.unlink(missing_ok=True)


def _write_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    _replace(tmp, path)  # atomic, so Studio never reads half a file


class SpectatorCallback(BaseCallback):
    """Watch the agent learn in real time.

    A spare env (in the training process, never used for learning) is driven by the
    current policy with exploration on, so it behaves like the training episodes. It's
    paced by wall-clock time, not training steps, so the cost stays fixed: about
    ``fps`` renders per second (around 5% of training speed on a laptop).

    frames=True  -> live/frame.jpg + live/spectator.json for Studio
    window=True  -> MuJoCo's interactive 3D viewer (rotate/zoom with the mouse)
    """

    def __init__(self, run_dir: Path, scenario: str, discrete: bool = False, wind: float | None = None,
                 frames: bool = True, window: bool = False, fps: float = 3.0, size=(480, 272)):
        super().__init__(0)
        self.dir = Path(run_dir) / "live"
        self.scenario, self.discrete, self.wind = scenario, discrete, wind
        self.frames, self.window, self.size = frames, window, size
        # The 3D window wants smooth real-time motion; the Studio image only refreshes ~2x/second,
        # so 3 renders/s (no shadows) is enough there. Measured on the 8 GB laptop: a render is
        # ~16 ms at 480x272 without shadows vs ~37 ms at 640x360 with them.
        self.fps = 20.0 if window else fps
        self.steps_per_frame = 1 if window else 5
        self.env = self.viewer = None
        self.next_t = 0.0
        self.hold = 0
        self.episode, self.wins, self.fails = 0, 0, 0
        self.ep_return, self.last = 0.0, {}

    def _on_training_start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def _new_episode(self) -> None:
        if self.env is None:
            self.env = EnvFactory(self.scenario, 7_000, 0, self.wind, self.discrete, "rgb_array")()
            u = self.env.unwrapped
            u.render_width, u.render_height = self.size
            u.fast_render = True
            if self.window:
                from mujoco import viewer as mj_viewer

                self.viewer = mj_viewer.launch_passive(u.model, u.data)
        try:
            level = int(self.training_env.env_method("get_level", indices=[0])[0])
        except Exception:
            level = 0
        self.env.unwrapped.set_level(level)
        self.obs, _ = self.env.reset(seed=7_000 + self.episode)
        self.episode += 1
        self.ep_return = 0.0

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now < self.next_t:
            return True
        self.next_t = now + 1.0 / self.fps
        if self.viewer is not None and not self.viewer.is_running():
            self.window = False  # user closed the 3D window: keep training, stop drawing there
            self.viewer = None
        if self.hold > 0:  # showing the WON/FAILED banner for a moment
            self.hold -= 1
            if self.hold == 0:
                self._new_episode()
            return True
        if self.env is None:
            self._new_episode()

        vecnorm = self.model.get_vec_normalize_env()
        done, info = False, {}
        for _ in range(self.steps_per_frame):
            o = vecnorm.normalize_obs(self.obs[None])[0] if vecnorm is not None else self.obs
            action, _ = self.model.predict(o, deterministic=False)
            self.obs, r, term, trunc, info = self.env.step(action)
            self.ep_return += float(r)
            done = term or trunc
            if done:
                break
        if self.viewer is not None:
            self.viewer.sync()

        u = self.env.unwrapped
        banner, ok = None, False
        if done:
            ok = bool(info.get("is_success"))
            reason = (info.get("reason") or "time_up").replace("_", " ").upper()
            banner = ("SUCCESS: " if ok else "FAILED: ") + reason
            self.wins += ok
            self.fails += not ok
            self.hold = int(1.5 * self.fps)
        self.last = {"episode": self.episode, "steps": int(self.num_timesteps), "level": u.level,
                     "score": round(self.ep_return, 1), "wins": self.wins, "fails": self.fails,
                     "result": banner, "time": time.time()}
        if self.frames:
            hud = (f"LIVE · {self.num_timesteps / 1000:.0f}k steps learned · level {u.level} · "
                   f"try #{self.episode} · score {self.ep_return:.0f}")
            if hasattr(u, "car_vel"):
                hud += f" · {np.linalg.norm(u.car_vel) * 3.6:.0f} km/h"
            from PIL import Image

            img = Image.fromarray(_hud(self.env.render(), hud, banner, ok))
            tmp = self.dir / "frame_tmp.jpg"
            img.save(tmp, quality=80)
            _replace(tmp, self.dir / "frame.jpg")
            _write_json(self.dir / "spectator.json", self.last)
        return True

    def _on_training_end(self) -> None:
        if self.frames and self.last:
            _write_json(self.dir / "spectator.json", {**self.last, "finished": True})
        if self.viewer is not None:
            self.viewer.close()
        if self.env is not None:
            self.env.close()


class LiveViewCallback(BaseCallback):
    def __init__(self, run_dir: Path, scenario: str, discrete: bool = False, wind: float | None = None,
                 every_steps: int = 25_000, keep_clips: int = 16, size=(480, 272), frame_stride: int = 3,
                 stats_every: int = 2_000):
        super().__init__(0)
        self.dir = Path(run_dir) / "live"
        self.scenario, self.discrete, self.wind = scenario, discrete, wind
        self.every, self.keep, self.size, self.stride = every_steps, keep_clips, size, frame_stride
        self.stats_every = stats_every
        self.env = None
        self.attempts: list[dict] = []
        self.counts: Counter = Counter()
        self.win_reasons: set[str] = set()
        self.wins = self.fails = 0
        self.window_wins = self.window_eps = 0
        self.timeline: list[dict] = []

    # ------------------------------------------------------------ callback
    def _on_training_start(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # Resumed run: carry on the existing attempt list and tallies.
        try:
            self.attempts = json.loads((self.dir / "attempts.json").read_text())
            stats = json.loads((self.dir / "stats.json").read_text())
            self.counts = Counter(stats.get("by_reason", {}))
            self.win_reasons = set(stats.get("win_reasons", []))
            self.wins, self.fails = stats.get("wins", 0), stats.get("fails", 0)
            self.timeline = stats.get("timeline", [])
        except (OSError, ValueError):
            pass
        self.next_clip = self.num_timesteps  # record the untrained agent first: the "before" clip
        self.next_stats = self.num_timesteps + self.stats_every

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            reason = info.get("reason") or ("time_up" if info.get("TimeLimit.truncated") else "ended")
            self.counts[reason] += 1
            if info.get("is_success"):
                self.win_reasons.add(reason)
                self.wins += 1
                self.window_wins += 1
            else:
                self.fails += 1
            self.window_eps += 1
        if self.num_timesteps >= self.next_stats:
            self._save_stats()
            self.next_stats += self.stats_every
        if self.num_timesteps >= self.next_clip:
            self._record_attempt()
            self.next_clip += self.every
        return True

    def _on_training_end(self) -> None:
        last = self.attempts[-1]["steps"] if self.attempts else -1
        if self.num_timesteps - last > self.every // 4:  # the "after" clip, unless one was just taken
            self._record_attempt()
        self._save_stats()
        if self.env is not None:
            self.env.close()

    # -------------------------------------------------------------- helpers
    def _level(self) -> int:
        try:
            return int(self.training_env.env_method("get_level", indices=[0])[0])
        except Exception:
            return 0

    def _save_stats(self) -> None:
        if self.window_eps:
            self.timeline.append({"steps": int(self.num_timesteps), "episodes": self.window_eps,
                                  "success_rate": self.window_wins / self.window_eps})
            self.window_wins = self.window_eps = 0
        _write_json(self.dir / "stats.json", {
            "steps": int(self.num_timesteps), "wins": self.wins, "fails": self.fails,
            "by_reason": dict(self.counts.most_common()), "win_reasons": sorted(self.win_reasons),
            "level": self._level(),
            "timeline": self.timeline[-400:], "updated": time.time()})

    def _record_attempt(self) -> None:
        level = self._level()
        n = len(self.attempts)
        if self.env is None:
            self.env = EnvFactory(self.scenario, 9_000, level, self.wind, self.discrete, "rgb_array")()
            u = self.env.unwrapped
            u.render_width, u.render_height = self.size
        u = self.env.unwrapped
        u.set_level(level)
        obs, _ = self.env.reset(seed=9_000 + n)
        vecnorm = self.model.get_vec_normalize_env()
        frames, total, steps, info, done = [], 0.0, 0, {}, False
        while not done:
            o = vecnorm.normalize_obs(obs[None])[0] if vecnorm is not None else obs
            action, _ = self.model.predict(o, deterministic=False)  # explore, like in training
            obs, r, term, trunc, info = self.env.step(action)
            total += r
            steps += 1
            done = term or trunc
            if steps % self.stride == 0 or done:
                hud = f"attempt #{n + 1} · {self.num_timesteps / 1000:.0f}k steps learned · level {level}"
                if hasattr(u, "car_vel"):
                    hud += f" · {np.linalg.norm(u.car_vel) * 3.6:.0f} km/h"
                frames.append(_hud(self.env.render(), hud))
        ok = bool(info.get("is_success"))
        reason = info.get("reason") or "time_up"
        if ok:
            self.win_reasons.add(reason)
        banner = ("SUCCESS: " if ok else "FAILED: ") + reason.replace("_", " ").upper()
        last = _hud(self.env.render(), f"attempt #{n + 1} · {self.num_timesteps / 1000:.0f}k steps learned",
                    banner, ok)
        frames.append(last)  # shown for 1.5 s at the end

        from PIL import Image

        stem = f"attempt_{n + 1:04d}"
        gif, png = self.dir / f"{stem}.gif", self.dir / f"{stem}.png"
        save_small_gif(frames, gif, frame_ms=1000 * u.dt * self.stride)
        Image.fromarray(last).save(png)
        self.attempts.append({"attempt": n + 1, "steps": int(self.num_timesteps), "level": level,
                              "return": round(float(total), 1), "success": ok, "reason": reason,
                              "gif": gif.name, "png": png.name, "time": time.time()})
        # keep disk use bounded: drop old clips (the list entry stays, with its thumbnail)
        for old in self.attempts[:-self.keep]:
            if old.get("gif"):
                (self.dir / old["gif"]).unlink(missing_ok=True)
                old["gif"] = None
        _write_json(self.dir / "attempts.json", self.attempts)
