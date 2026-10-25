"""TrainLab command line.

    trainlab list
    trainlab play     <scenario> [--level N]
    trainlab train    <scenario> [--algo ppo] [--steps 1e6] [--level N] [--no-curriculum]
    trainlab watch    <run dir or model.zip> [--episodes 5]
    trainlab record   <run dir or model.zip> [--out clip.gif]
    trainlab compare  <scenario> [--algos ppo sac dqn] [--seeds 3] [--steps 1e6]
    trainlab new      <name>
    trainlab generate "<describe a simulation>"
    trainlab studio
"""

from __future__ import annotations

import argparse
import sys


SAMPLE_SCENARIO = "roadster_gap_jump"  # the built-in example every command falls back to


def _steps(text: str) -> int:
    return int(float(text))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="trainlab", description="Train RL agents on any 3D simulation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available scenarios")

    sp = sub.add_parser("play", help="drive a scenario yourself with the keyboard")
    sp.add_argument("scenario", nargs="?", default=SAMPLE_SCENARIO)
    sp.add_argument("--level", type=int, default=0)
    sp.add_argument("--wind", type=float, default=None)

    sp = sub.add_parser("train", help="train an agent (default: the sample Roadster GapJump)")
    sp.add_argument("scenario", nargs="?", default=SAMPLE_SCENARIO)
    sp.add_argument("--algo", default="ppo", choices=["ppo", "a2c", "sac", "td3", "dqn"])
    sp.add_argument("--steps", type=_steps, default=1_000_000)
    sp.add_argument("--envs", type=int, default=None, help="parallel environments")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--level", type=int, default=0, help="starting difficulty level")
    sp.add_argument("--wind", type=float, default=None, help="override wind strength (m/s)")
    sp.add_argument("--no-curriculum", action="store_true", help="stay on --level the whole time")
    sp.add_argument("--resume", default=None, help="continue a previous run folder")
    sp.add_argument("--run-dir", default=None, help="where to save (default runs/<scenario>/...)")
    sp.add_argument("--no-live", action="store_true", help="don't record live trial-and-error clips")

    sp = sub.add_parser("resume", help="continue a run that was cut short (shutdown, Stop, low memory)")
    sp.add_argument("run", nargs="?", default=None, help="run folder (default: the most recent unfinished run)")
    sp.add_argument("--no-live", action="store_true")

    sub.add_parser("status", help="show this machine's training profile and unfinished runs")

    for name, helptext in (("watch", "watch a trained agent in the 3D viewer"),
                           ("record", "save a GIF/MP4 of a trained agent")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("model", help="run folder or .zip model")
        sp.add_argument("--episodes", type=int, default=3 if name == "watch" else 2)
        sp.add_argument("--level", type=int, default=None)
        sp.add_argument("--camera", default=None, help="chase | side | driver")
        if name == "record":
            sp.add_argument("--out", default=None, help="output .gif or .mp4")

    sp = sub.add_parser("compare", help="compare algorithms across seeds")
    sp.add_argument("scenario", nargs="?", default=SAMPLE_SCENARIO)
    sp.add_argument("--algos", nargs="+", default=["ppo", "sac", "dqn"])
    sp.add_argument("--seeds", type=int, default=3)
    sp.add_argument("--steps", type=_steps, default=1_000_000)
    sp.add_argument("--level", type=int, default=0)
    sp.add_argument("--curriculum", action="store_true")

    sp = sub.add_parser("new", help="create a new scenario from the template")
    sp.add_argument("name")

    sp = sub.add_parser("generate", help="describe a simulation in words; an LLM builds it")
    sp.add_argument("prompt")
    sp.add_argument("--provider", default=None, help="anthropic | openai | gemini | groq | openrouter | "
                                                      "deepseek | mistral | ollama | custom (default: from .env)")
    sp.add_argument("--model", default=None, help="model id (default: the provider's default)")
    sp.add_argument("--base-url", default=None, help="API base URL for 'custom' or a non-default Ollama")
    sp.add_argument("--name", default=None, help="scenario name (snake_case)")

    sp = sub.add_parser("studio", help="open the TrainLab Studio web app")
    sp.add_argument("--port", type=int, default=7860)
    sp.add_argument("--share", action="store_true", help="create a public Gradio link")
    sp.add_argument("--no-browser", action="store_true", help="don't open a browser tab")

    a = p.parse_args(argv)

    if a.cmd == "list":
        from trainlab.core.registry import get_scenario, list_scenarios

        for name in list_scenarios():
            cls = get_scenario(name)
            doc = (sys.modules[cls.__module__].__doc__ or "").strip().split("\n")[0]
            tag = "  <- sample (default)" if name == SAMPLE_SCENARIO else ""
            print(f"  {name:28s} levels={len(cls.CURRICULUM)}  {doc[:70]}{tag}")
    elif a.cmd == "play":
        from trainlab.play.keyboard import play

        play(a.scenario, a.level, a.wind)
    elif a.cmd == "train":
        from trainlab.agents.train import train

        rd = train(a.scenario, a.algo, a.steps, a.envs, a.seed, a.level, not a.no_curriculum, a.wind,
                   run_dir=a.run_dir, resume=a.resume, live=not a.no_live)
        print(f"\nDone. Run folder: {rd}\n  watch:  trainlab watch \"{rd}\"\n  record: trainlab record \"{rd}\"")
    elif a.cmd == "resume":
        from trainlab.agents.train import resume_run

        rd = resume_run(a.run, live=not a.no_live)
        print(f"\nRun folder: {rd}")
    elif a.cmd == "status":
        import json as _json

        from trainlab.agents.train import RUNS_DIR, is_unfinished, latest_save
        from trainlab.core.resources import free_ram_gb, profile, total_ram_gb

        prof = profile()
        print(f"RAM {total_ram_gb():.1f} GB ({free_ram_gb():.1f} GB free) -> profile '{prof['name']}': "
              f"{prof['n_envs']} parallel envs, {prof['torch_threads']} CPU threads, pause below "
              f"{prof['min_free_gb']} GB free")
        runs = [r.parent for r in RUNS_DIR.glob("*/*/run.json") if is_unfinished(r.parent)]
        if not runs:
            print("No unfinished runs.")
        for r in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True):
            m = _json.loads((r / "run.json").read_text())
            print(f"  unfinished: {r}  {latest_save(r)[2]:,} / {m['steps']:,} steps  ->  trainlab resume \"{r}\"")
    elif a.cmd == "watch":
        from trainlab.agents.runner import TrainedPolicy, run_episodes

        pol = TrainedPolicy(a.model)
        env = pol.make_env(render_mode="human", level=a.level)
        if a.camera:
            env.unwrapped.camera = a.camera
        for r in run_episodes(pol, env, a.episodes, on_step=lambda e, i: e.unwrapped.viewer_running()):
            print(r)
        env.close()
    elif a.cmd == "record":
        from pathlib import Path

        from trainlab.record import record

        out = a.out or str(Path(a.model if Path(a.model).is_dir() else Path(a.model).parent) / "clip.gif")
        record(a.model, out, a.episodes, a.level, camera=a.camera)
    elif a.cmd == "compare":
        from trainlab.agents.compare import compare

        compare(a.scenario, a.algos, a.seeds, a.steps, a.level, a.curriculum)
    elif a.cmd == "new":
        from trainlab.generator.scaffold import new_scenario

        print(f"Created {new_scenario(a.name)}")
    elif a.cmd == "generate":
        from trainlab.generator.pipeline import generate_cli

        return generate_cli(a.prompt, a.provider, a.model, a.base_url, a.name)
    elif a.cmd == "studio":
        from trainlab.studio.app import launch

        launch(port=a.port, share=a.share, open_browser=not a.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
