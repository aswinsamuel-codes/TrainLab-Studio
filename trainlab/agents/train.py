"""Train any registered scenario with Stable-Baselines3.

Output goes to runs/<scenario>/<algo>_s<seed>_<timestamp>/:
    best_model.zip + vecnormalize_best.pkl   (best evaluation so far)
    final_model.zip + vecnormalize.pkl       (end of training)
    progress.csv, evaluations.npz, tb/       (curves; open with TensorBoard)
    run.json                                 (what was trained, for watch/record)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import yaml
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from trainlab.core.curriculum import (AutosaveCallback, CurriculumCallback, SaveVecNormalizeCallback,
                                     StopFileCallback)
from trainlab.core.factory import EnvFactory
from trainlab.core.live import LiveViewCallback, SpectatorCallback
from trainlab.core.resources import MemoryGuardCallback, be_nice, profile
from trainlab.core.registry import GENERATED_DIR, get_scenario

ALGOS = {"ppo": PPO, "a2c": A2C, "sac": SAC, "td3": TD3, "dqn": DQN}
DISCRETE_ONLY = {"dqn"}
ON_POLICY = {"ppo", "a2c"}

PACKAGE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("TRAINLAB_RUNS", PACKAGE_DIR.parent / "runs"))


def make_vec(scenario, n_envs, seed, level=0, wind=None, discrete=False, subproc=True):
    fns = [EnvFactory(scenario, seed + i, level, wind, discrete) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if (subproc and n_envs > 1) else DummyVecEnv(fns)
    return VecMonitor(venv, info_keywords=("is_success",))


def _deep_merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(scenario: str) -> dict:
    cfg = yaml.safe_load((PACKAGE_DIR / "configs" / "default.yaml").read_text())
    for path in (PACKAGE_DIR / "configs" / f"{scenario}.yaml", GENERATED_DIR / scenario / "config.yaml"):
        if path.exists():
            cfg = _deep_merge(cfg, yaml.safe_load(path.read_text()) or {})
    return cfg


def build_model(algo: str, venv, hp: dict, seed: int, tb_dir: Path):
    hp = dict(hp)
    net_arch = hp.pop("net_arch", [256, 256])
    return ALGOS[algo]("MlpPolicy", venv, seed=seed, verbose=0, tensorboard_log=str(tb_dir),
                       policy_kwargs={"net_arch": net_arch}, device="cpu", **hp)


def latest_save(run_dir: Path) -> tuple[Path, Path, int, int] | None:
    """Newest saved (model, vecnormalize, timesteps, level) in a run: autosave or end-of-run save."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    options = []
    state = run_dir / "autosave" / "state.json"
    if state.exists() and (run_dir / "autosave" / "model.zip").exists():
        s = json.loads(state.read_text())
        options.append((run_dir / "autosave" / "model.zip", run_dir / "autosave" / "vecnormalize.pkl",
                        int(s["timesteps"]), int(s.get("level", meta.get("level", 0)))))
    if (run_dir / "final_model.zip").exists() and "timesteps" in meta:
        options.append((run_dir / "final_model.zip", run_dir / "vecnormalize.pkl", int(meta["timesteps"]),
                        int(meta.get("final_level", meta.get("level", 0)))))
    return max(options, key=lambda o: o[2]) if options else None


def is_unfinished(run_dir: Path) -> bool:
    meta_path = Path(run_dir) / "run.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    save = latest_save(run_dir)
    done = save[2] if save else 0
    return done < int(meta.get("steps", 0)) and save is not None


def find_unfinished() -> Path | None:
    """Most recent run that was cut short (laptop shut down, Stop, low memory)."""
    runs = sorted((p.parent for p in RUNS_DIR.glob("*/*/run.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    return next((r for r in runs if is_unfinished(r)), None)


def resume_run(run_dir: Path | None = None, verbose: int = 1, live: bool = True, spectate: bool = False,
               view: bool = False, steps: int | None = None) -> Path:
    """Continue a run from its last save. ``steps`` raises the run's total (to train a finished run longer)."""
    run_dir = Path(run_dir) if run_dir else find_unfinished()
    if run_dir is None:
        raise SystemExit("Nothing to resume: every run in runs/ has finished. "
                         "To train one longer: trainlab resume <run folder> --steps <new total>")
    meta = json.loads((run_dir / "run.json").read_text())
    if steps:
        meta["steps"] = int(steps)
    return train(meta["scenario"], meta["algo"], meta["steps"], meta.get("n_envs"), meta["seed"],
                 curriculum=meta.get("curriculum", True), wind=meta.get("wind"), run_dir=run_dir,
                 resume=run_dir, verbose=verbose, live=live, spectate=spectate, view=view)


def train(scenario: str, algo: str = "ppo", steps: int = 1_000_000, n_envs: int | None = None,
          seed: int = 0, level: int = 0, curriculum: bool = True, wind: float | None = None,
          run_dir: Path | None = None, resume: Path | None = None, verbose: int = 1, live: bool = True,
          spectate: bool = False, view: bool = False) -> Path:
    algo = algo.lower()
    if algo not in ALGOS:
        raise ValueError(f"Unknown algo '{algo}'. Choose from {', '.join(ALGOS)}")
    cls = get_scenario(scenario)
    cfg = load_config(scenario)
    hp = dict(cfg["algos"].get(algo, {}))
    discrete = algo in DISCRETE_ONLY
    machine = profile()
    be_nice(int(cfg.get("torch_threads", machine["torch_threads"])))
    n_envs = n_envs or min(int(cfg.get("n_envs", machine["n_envs"])), machine["n_envs"], os.cpu_count() or 1)
    if algo not in ON_POLICY:
        n_envs = min(n_envs, 4)

    save = None
    done_steps = 0
    if resume:
        run_dir = Path(resume)
        save = latest_save(run_dir)
        if save is None:
            raise SystemExit(f"No saved model in {run_dir} to resume from.")
        done_steps, level = save[2], save[3]
    run_dir = Path(run_dir or RUNS_DIR / scenario / f"{algo}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)
    remaining = int(steps) - done_steps
    if remaining <= 0:
        print(f"{run_dir} already reached {steps:,} steps.")
        return run_dir

    venv = make_vec(scenario, n_envs, seed + done_steps, level, wind, discrete)
    eval_env = make_vec(scenario, 1, seed + 10_000, level, wind, discrete, subproc=False)
    gamma = hp.get("gamma", 0.99)
    if save is not None:
        model_file, norm_file = save[0], save[1]
        venv = VecNormalize.load(str(norm_file), venv) if norm_file.exists() else \
            VecNormalize(venv, norm_obs=True, norm_reward=algo in ON_POLICY, clip_obs=10.0, gamma=gamma)
        model = ALGOS[algo].load(str(model_file), env=venv, device="cpu")
        print(f"Resuming {run_dir.name} from {done_steps:,} / {int(steps):,} steps (level {level}).")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=algo in ON_POLICY, clip_obs=10.0, gamma=gamma)
        model = build_model(algo, venv, hp, seed, run_dir / "tb")
    eval_env = VecNormalize(eval_env, training=False, norm_reward=False, clip_obs=10.0)

    model.set_logger(configure(str(run_dir), ["stdout", "csv", "tensorboard"] if verbose else ["csv", "tensorboard"]))

    old = json.loads((run_dir / "run.json").read_text()) if (resume and (run_dir / "run.json").exists()) else {}
    meta = {"scenario": scenario, "algo": algo, "discrete": discrete, "level": old.get("level", level), "seed": seed,
            "steps": int(steps), "n_envs": n_envs, "wind": wind, "curriculum": curriculum,
            "max_level": len(cls.CURRICULUM) - 1, "hyperparams": hp, "machine": machine["name"],
            "started": old.get("started", time.strftime("%Y-%m-%d %H:%M:%S"))}
    if resume:
        meta["resumed"] = old.get("resumed", []) + [{"at_steps": done_steps, "time": time.strftime("%Y-%m-%d %H:%M:%S")}]
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    if verbose:
        print(f"Machine profile: {machine['name']} -> {n_envs} parallel envs, "
              f"{cfg.get('torch_threads', machine['torch_threads'])} CPU threads, low priority. "
              f"Autosave every {int(cfg.get('autosave_every', 20_000)):,} steps.")

    curr_cb = None
    if curriculum and len(cls.CURRICULUM) > 1:
        curr_cb = CurriculumCallback(venv, eval_env, level, len(cls.CURRICULUM) - 1,
                                     cfg["curriculum"]["threshold"], run_dir, verbose)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path=str(run_dir), log_path=str(run_dir),
        eval_freq=max(int(cfg["eval_every"]) // n_envs, 1), n_eval_episodes=int(cfg["eval_episodes"]),
        deterministic=True, callback_after_eval=curr_cb,
        callback_on_new_best=SaveVecNormalizeCallback(run_dir / "vecnormalize_best.pkl"), verbose=verbose)
    ckpt_cb = CheckpointCallback(save_freq=max(200_000 // n_envs, 1), save_path=str(run_dir / "checkpoints"),
                                 name_prefix=algo, save_vecnormalize=True)

    autosave = AutosaveCallback(run_dir, every=int(cfg.get("autosave_every", 20_000)))
    guard = MemoryGuardCallback(autosave, min_free_gb=float(cfg.get("min_free_ram_gb", machine["min_free_gb"])),
                                verbose=verbose, run_dir=run_dir)
    callbacks = [eval_cb, ckpt_cb, autosave, guard, StopFileCallback(run_dir)]
    if live:
        callbacks.append(LiveViewCallback(run_dir, scenario, discrete, wind,
                                          every_steps=int(cfg.get("live_every", 25_000))))
    if spectate or view:
        callbacks.append(SpectatorCallback(run_dir, scenario, discrete, wind, frames=spectate, window=view,
                                           fps=float(cfg.get("spectator_fps", 3))))

    t0 = time.time()
    try:
        model.learn(remaining, callback=CallbackList(callbacks),
                    reset_num_timesteps=not resume, tb_log_name="run")
    finally:
        model.save(str(run_dir / "final_model"))
        venv.save(str(run_dir / "vecnormalize.pkl"))
        autosave.save_now()
        reached = int(model.num_timesteps) >= int(steps)
        status = "finished" if reached else ("paused_low_memory" if guard.stopped_for_memory else "stopped")
        meta.update(final_level=curr_cb.level if curr_cb else level, timesteps=int(model.num_timesteps),
                    status=status, wall_seconds=round(time.time() - t0, 1) + old.get("wall_seconds", 0),
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        if not reached:
            print(f"Run paused at {model.num_timesteps:,} / {int(steps):,} steps. Continue with:  trainlab resume")
        (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
        venv.close()
        eval_env.close()
    return run_dir
