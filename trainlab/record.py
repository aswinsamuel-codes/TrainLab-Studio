"""Turn a trained agent into a GIF or MP4 clip with a small heads-up display."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from trainlab.agents.runner import TrainedPolicy, run_episodes


def hud_text(env, info: dict, title: str) -> str:
    u = env.unwrapped
    parts = [title]
    if hasattr(u, "car_vel"):
        parts.append(f"{np.linalg.norm(u.car_vel) * 3.6:5.0f} km/h")
    if hasattr(u, "fuel_frac"):
        parts.append(f"fuel {u.fuel_frac():4.0%}")
    if info.get("reason"):
        parts.append(info["reason"].upper())
    return "  |  ".join(parts)


def draw_hud(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 26], fill=(0, 0, 0, 150))
    draw.text((10, 7), text, fill=(255, 255, 255, 255))
    return np.asarray(img)


def record(model_path: str | Path, out: str | Path, episodes: int = 2, level: int | None = None,
           width: int = 960, height: int = 544, camera: str | None = None, seed: int = 0) -> Path:
    policy = TrainedPolicy(model_path)
    env = policy.make_env(render_mode="rgb_array", level=level, seed=seed)
    u = env.unwrapped
    u.render_width, u.render_height = width, height
    if camera:
        u.camera = camera
    title = f"TrainLab · {policy.scenario} · {policy.meta['algo'].upper()}"
    frames: list[np.ndarray] = []
    last_info: dict = {}

    def grab(e, info):
        last_info.clear()
        last_info.update(info)
        frames.append(draw_hud(e.render(), hud_text(e, info, title)))

    stats = run_episodes(policy, env, episodes, on_step=grab, seed=seed)
    # hold the final frame for a second so the result is readable
    if frames:
        frames.extend([frames[-1]] * int(u.metadata["render_fps"]))
    env.close()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = int(u.metadata["render_fps"])
    if out.suffix.lower() == ".gif":
        small = [np.asarray(Image.fromarray(f).resize((width // 2, height // 2))) for f in frames[::2]]
        imageio.mimsave(out, small, duration=2000 / fps, loop=0)
    else:
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)
    print(f"saved {out}  ({len(frames)} frames)  episodes: {stats}")
    return out
