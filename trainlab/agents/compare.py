"""Train several algorithms x seeds on one scenario and compare them.

Writes runs/<scenario>/compare_<timestamp>/ with:
    compare.png   evaluation reward and success rate vs. timesteps (mean, min-max band)
    summary.csv   one row per run
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from trainlab.agents.train import RUNS_DIR, train

# Fixed hue per algorithm, so PPO is the same color in every chart.
ALGO_COLORS = {"ppo": "#2a78d6", "sac": "#eb6834", "dqn": "#1baf7a", "td3": "#eda100", "a2c": "#e87ba4"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def load_evals(run_dir: Path):
    path = run_dir / "evaluations.npz"
    if not path.exists():
        return None
    d = np.load(path)
    t = d["timesteps"]
    reward = d["results"].mean(axis=1)
    success = d["successes"].mean(axis=1) if "successes" in d and d["successes"].size else np.zeros_like(reward)
    return t, reward, success


def plot(groups: dict[str, list[Path]], out: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=SURFACE)
    for ax, (idx, label) in zip(axes, [(1, "Mean evaluation reward"), (2, "Success rate")]):
        ax.set_facecolor(SURFACE)
        for algo, runs in groups.items():
            curves = [c for c in (load_evals(r) for r in runs) if c is not None and len(c[0])]
            if not curves:
                continue
            grid = np.linspace(0, min(c[0][-1] for c in curves), 100)
            ys = np.stack([np.interp(grid, c[0], c[idx]) for c in curves])
            color = ALGO_COLORS.get(algo, "#6250d6")
            ax.fill_between(grid, ys.min(0), ys.max(0), color=color, alpha=0.15, linewidth=0)
            ax.plot(grid, ys.mean(0), color=color, linewidth=2, label=f"{algo.upper()} ({len(curves)} seed{'s' if len(curves) > 1 else ''})")
            ax.annotate(algo.upper(), (grid[-1], ys.mean(0)[-1]), xytext=(6, 0), textcoords="offset points",
                        color=INK_2, fontsize=9, va="center")
        ax.set_title(label, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("environment steps", color=INK_2)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=INK_2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        if idx == 2:
            ax.set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=False, labelcolor=INK_2)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def compare(scenario: str, algos=("ppo", "sac", "dqn"), seeds: int = 3, steps: int = 1_000_000,
            level: int = 0, curriculum: bool = False, n_envs: int | None = None) -> Path:
    out_dir = RUNS_DIR / scenario / f"compare_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {a: [] for a in algos}
    rows = []
    for algo in algos:
        for seed in range(seeds):
            print(f"\n=== {algo.upper()} seed {seed} ===")
            t0 = time.time()
            rd = train(scenario, algo, steps, n_envs=n_envs, seed=seed, level=level, curriculum=curriculum,
                       run_dir=out_dir / f"{algo}_s{seed}", verbose=0, live=False)
            groups[algo].append(rd)
            ev = load_evals(rd)
            rows.append({
                "algo": algo, "seed": seed, "run_dir": str(rd),
                "final_eval_reward": round(float(ev[1][-1]), 2) if ev else "",
                "best_eval_reward": round(float(ev[1].max()), 2) if ev else "",
                "final_success_rate": round(float(ev[2][-1]), 3) if ev else "",
                "wall_minutes": round((time.time() - t0) / 60, 1),
            })
            plot(groups, out_dir / "compare.png", f"{scenario}: {' vs '.join(a.upper() for a in algos)}")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison saved to {out_dir}")
    return out_dir
