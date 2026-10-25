"""Drive any scenario yourself with the keyboard (pygame window).

Your scores are logged to runs/<scenario>/human_scores.csv, so you can compare
yourself against the trained agent.
"""

from __future__ import annotations

import csv
import time

import numpy as np

from trainlab.agents.train import RUNS_DIR

# control name -> (positive key, negative key) as pygame key names
KEYMAP = {
    "throttle": ("w", "s"), "steer": ("a", "d"), "brake": ("left shift", None),
    "lift": ("space", None), "pitch": ("k", "i"), "roll": ("l", "j"), "yaw": ("q", "e"),
    "thrust_x": ("up", "down"), "thrust_y": ("left", "right"), "thrust_z": ("space", "left ctrl"),
}
# for scenarios without named controls: action dim i -> key pair
GENERIC_KEYS = [("w", "s"), ("a", "d"), ("i", "k"), ("j", "l"), ("up", "down"),
                ("left", "right"), ("q", "e"), ("space", "left ctrl"), ("u", "o"), ("n", "m")]
CAMERAS = ["chase", "side", "driver"]


def play(scenario: str, level: int = 0, wind: float | None = None, width: int = 960, height: int = 544):
    import pygame

    import trainlab

    env = trainlab.make(scenario, level=level, wind=wind, render_mode="rgb_array")
    u = env.unwrapped
    u.render_width, u.render_height = width, height
    names = list(getattr(u, "CONTROLS", [])) or [f"a{i}" for i in range(env.action_space.shape[0])]
    keys = [KEYMAP.get(n, GENERIC_KEYS[i % len(GENERIC_KEYS)]) for i, n in enumerate(names)]

    pygame.init()
    screen = pygame.display.set_mode((width, height + 60))
    pygame.display.set_caption(f"TrainLab · {scenario} · level {level}")
    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()
    help_line = "  ".join(f"{n}:{p.upper()}{'/' + q.upper() if q else ''}" for n, (p, q) in zip(names, keys))
    help_line += "  |  R reset  TAB camera  1-5 level  ESC quit"

    log_path = RUNS_DIR / scenario / "human_scores.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["time", "level", "return", "success", "reason", "steps"])

    obs, _ = env.reset()
    total, steps, msg, cam = 0.0, 0, "", 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    obs, _ = env.reset(); total, steps, msg = 0.0, 0, ""
                elif event.key == pygame.K_TAB:
                    cam = (cam + 1) % len(CAMERAS); u.camera = CAMERAS[cam]
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    u.set_level(event.key - pygame.K_1)
                    obs, _ = env.reset(); total, steps, msg = 0.0, 0, ""
                    pygame.display.set_caption(f"TrainLab · {scenario} · level {u.level}")

        pressed = pygame.key.get_pressed()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        for i, (pos, neg) in enumerate(keys):
            if pressed[pygame.key.key_code(pos)]:
                action[i] += 1.0
            if neg and pressed[pygame.key.key_code(neg)]:
                action[i] -= 1.0

        if not msg:
            obs, r, term, trunc, info = env.step(action)
            total += r
            steps += 1
            if term or trunc:
                ok = info.get("is_success")
                msg = f"{'SUCCESS' if ok else 'FAILED'} ({info.get('reason') or 'time up'})  score {total:.1f}  — press R"
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), u.level, round(total, 2), ok,
                                 info.get("reason", ""), steps])
                log.flush()

        frame = env.render()
        screen.blit(pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1)), (0, 0))
        pygame.draw.rect(screen, (15, 15, 20), (0, height, width, 60))
        speed = f"{np.linalg.norm(u.car_vel) * 3.6:.0f} km/h  " if hasattr(u, "car_vel") else ""
        fuel = f"fuel {u.fuel_frac():.0%}  " if hasattr(u, "fuel_frac") else ""
        status = f"score {total:7.1f}  {speed}{fuel}level {u.level}  {msg}"
        screen.blit(font.render(status, True, (255, 255, 255)), (10, height + 8))
        screen.blit(font.render(help_line[:140], True, (160, 170, 190)), (10, height + 34))
        pygame.display.flip()
        clock.tick(u.metadata["render_fps"])

    log.close()
    env.close()
    pygame.quit()
