"""TrainLab Studio: describe a simulation, review it, train it, watch the result.

    trainlab studio            ->  http://localhost:7860
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from trainlab.agents.train import RUNS_DIR
from trainlab.core.registry import list_scenarios, load_all
from trainlab.cli import SAMPLE_SCENARIO
from trainlab.generator.llm_client import (PROVIDERS, GenerationError, MissingKeyError, default_model,
                                           default_provider, env_base_url, get_provider, key_status,
                                           save_settings_to_env)
from trainlab.generator.mesh_provider import get_mesh_provider

EXAMPLES = [
    "The Roadster with thrusters must hop over three walls in a row and stop in a garage.",
    "A quadcopter drone that must land gently on a ship deck that pitches up and down in waves, with gusty wind.",
    "A two-wheeled balancing robot that has to drive to a flag without falling over.",
    "A robotic arm with 3 joints that pushes a cube into a target square on a table.",
]
PROVIDER_CHOICES = [(info.label, name) for name, info in PROVIDERS.items()]


def _provider_md(provider: str) -> str:
    info, status = PROVIDERS[provider], key_status()[provider]
    lines = [f"**{info.label}**: key {status}" + (f" · get a key: {info.key_url}" if info.key_url else "")]
    if info.note:
        lines.append(info.note)
    return "  \n".join(lines)


def _llm_label(llm: dict) -> str:
    return (f"Simulations are written by **{PROVIDERS[llm['provider']].label} · {llm['model'] or '(pick a model)'}**"
            f" (key {'from Settings' if llm.get('key') else key_status()[llm['provider']]}). Change it in **Settings**.")


# ------------------------------------------------------------------ helpers
def _session_status(session) -> str:
    if session is None or not session.data:
        return "No scenario generated yet."
    r = session.report
    ok = "✅ passed all checks" if r.get("ok") else "❌ failing checks"
    return (f"**{session.data['title']}** (`{session.data['name']}`): {ok}  \n"
            f"obs {r.get('obs_dim', '?')} values · actions {r.get('act_dim', '?')} · "
            f"{r.get('levels', '?')} difficulty levels · random-play returns {r.get('random_returns', [])}")


def _controls_md(data: dict) -> str:
    rows = "\n".join(f"| {c['name']} | {c['meaning']} |" for c in data.get("controls", []))
    return f"| control | meaning |\n|---|---|\n{rows}"


def _review_outputs(session):
    d = session.data
    folder = session.folder
    gif = folder / "preview.gif" if folder and (folder / "preview.gif").exists() else None
    png = folder / "preview.png" if folder and (folder / "preview.png").exists() else None
    return (_session_status(session), str(gif or png) if (gif or png) else None,
            d.get("summary", ""), _controls_md(d), d.get("reward_explanation", ""),
            d.get("mjcf_xml", ""), d.get("scenario_py", ""))


def _read_progress(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "progress.csv"
    if not path.exists():
        return pd.DataFrame({"steps": [], "value": [], "series": []})
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame({"steps": [], "value": [], "series": []})
    out = []
    for col, label in (("rollout/ep_rew_mean", "training reward"), ("eval/mean_reward", "evaluation reward")):
        if col in df and "time/total_timesteps" in df:
            part = df[["time/total_timesteps", col]].dropna()
            out.append(pd.DataFrame({"steps": part.iloc[:, 0], "value": part.iloc[:, 1], "series": label}))
    return pd.concat(out) if out else pd.DataFrame({"steps": [], "value": [], "series": []})


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _live_frame(run_dir: Path):
    """(latest real-time frame as PIL image or None, status markdown, finished?)."""
    from PIL import Image

    live = run_dir / "live"
    spec = _load_json(live / "spectator.json", {})
    meta = _load_json(run_dir / "run.json", {})
    finished = bool(spec.get("finished")) or "finished" in meta
    frame = None
    try:
        with Image.open(live / "frame.jpg") as im:
            frame = im.convert("RGB").copy()
    except (OSError, ValueError):
        pass
    paused = _load_json(live / "paused.json", {})
    if paused and not finished and time.time() - paused.get("since", 0) < 30:
        return frame, (f"### ⏸ Paused: laptop is low on memory\nOnly **{paused['free_gb']} GB** RAM free "
                       f"(needs {paused['need_gb']:.2f} GB). Progress is saved. Close a few apps (videos, Spotify, "
                       "Discord, extra browser tabs) and training continues by itself. If memory stays low for "
                       "10 minutes it stops safely, then use **Resume unfinished run**."), False
    if not spec:
        msg = ("Waiting for the first live frame…" if not finished else
               "This run wasn't started with live view. In the terminal, add `--spectate` (or `--view` for a 3D window).")
        return frame, msg, finished
    age = time.time() - spec.get("time", 0)
    state = "⏹ training finished" if finished else ("🔴 LIVE" if age < 10 else f"⏸ no new frame for {age:.0f}s (paused?)")
    result = f"\n\nLast result: **{spec['result']}**" if spec.get("result") else ""
    md = (f"### {state}\n**Try #{spec['episode']}** · score {spec['score']} · level {spec['level']}  \n"
          f"{spec['steps'] / 1000:.0f}k steps learned  \n"
          f"Live tries so far: ✅ {spec['wins']} won · ❌ {spec['fails']} failed{result}")
    return frame, md, finished


def _live_outputs(run_dir: Path):
    """(latest clip, scoreboard markdown, win-rate dataframe, gallery items) for the live panel."""
    live = run_dir / "live"
    attempts = _load_json(live / "attempts.json", [])
    stats = _load_json(live / "stats.json", {})
    empty = pd.DataFrame({"steps": [], "value": [], "series": []})
    if not attempts and not stats:
        return None, "Waiting for the first attempt (recorded right after training starts)…", empty, []

    lines = []
    if attempts:
        a = attempts[-1]
        verdict = "✅ **WON**" if a["success"] else "❌ **FAILED**"
        lines.append(f"### Attempt #{a['attempt']}: {verdict} ({a['reason'].replace('_', ' ')})\n"
                     f"after {a['steps'] / 1000:.0f}k steps of learning · level {a['level']} · score {a['return']}")
        won = sum(x["success"] for x in attempts)
        lines.append(f"Recorded attempts: **{len(attempts)}** (✅ {won} · ❌ {len(attempts) - won})")
    if stats:
        total = stats["wins"] + stats["fails"]
        lines.append(f"**Training episodes so far: {total:,}**: ✅ {stats['wins']:,} won · ❌ {stats['fails']:,} failed"
                     f" · curriculum level {stats.get('level', 0)}")
        wins = set(stats.get("win_reasons", []))
        rows = "\n".join(f"| {'✅' if r in wins else '❌'} {r.replace('_', ' ')} | {n:,} |"
                         for r, n in stats.get("by_reason", {}).items())
        if rows:
            lines.append(f"| how episodes ended | count |\n|---|---|\n{rows}")
    tl = stats.get("timeline", [])
    df = pd.DataFrame({"steps": [t["steps"] for t in tl], "value": [100 * t["success_rate"] for t in tl],
                       "series": "win rate"}) if tl else empty

    latest = None
    if attempts:
        last = attempts[-1]
        latest = str(live / (last["gif"] or last["png"]))
    gallery = []
    for a in reversed(attempts[-24:]):
        f = live / (a["gif"] or a["png"])
        if f.exists():
            mark = "✅" if a["success"] else "❌"
            gallery.append((str(f), f"{mark} #{a['attempt']} · {a['steps'] / 1000:.0f}k · {a['reason'].replace('_', ' ')}"))
    return latest, "\n\n".join(lines), df, gallery


def _run_summary(run_dir: Path, proc) -> str:
    meta = json.loads((run_dir / "run.json").read_text()) if (run_dir / "run.json").exists() else {}
    if proc is not None:
        state = "running" if proc.poll() is None else "finished"
    else:
        state = "finished" if "finished" in meta else "running"
    lines = [f"**{state}** · `{run_dir}`"]
    ev = run_dir / "evaluations.npz"
    if ev.exists():
        try:
            d = np.load(ev)
            lines.append(f"last eval: reward {d['results'][-1].mean():.1f}"
                         + (f", success {d['successes'][-1].mean():.0%}" if d["successes"].size else ""))
        except Exception:
            pass
    if "final_level" in meta:
        lines.append(f"curriculum level reached: {meta['final_level']} / {meta.get('max_level')}")
    if state == "finished" and proc is not None and proc.returncode not in (0, None):
        log = run_dir / "studio_train.log"
        tail = log.read_text(errors="replace")[-1200:] if log.exists() else ""
        lines.append(f"❌ training stopped with an error (exit {proc.returncode}):\n```\n{tail}\n```")
    return "  \n".join(lines)


# ------------------------------------------------------------------- app
def build() -> gr.Blocks:
    with gr.Blocks(title="TrainLab Studio") as demo:
        session_state = gr.State(None)
        start_provider = default_provider()
        llm_state = gr.State({"provider": start_provider, "key": "", "model": default_model(start_provider),
                              "base_url": ""})
        train_state = gr.State({"proc": None, "run_dir": None})

        gr.Markdown("# TrainLab Studio\nDescribe a 3D simulation in plain English. TrainLab builds it with the AI "
                    "model of your choice, checks it, trains a reinforcement-learning agent on it, and shows you "
                    "the result.")
        with gr.Row(equal_height=True):
            gr.Markdown("**New here?** Start with the built-in sample: the **Roadster GapJump**. A sports car with "
                        "thrusters learns to jump a pit and park. No API key needed.")
            sample_btn = gr.Button("Train the sample Roadster", variant="secondary", scale=0, min_width=230)

        with gr.Tabs() as tabs:
            # ---------------------------------------------------------- Create
            with gr.Tab("1 · Create", id="create"):
                llm_md = gr.Markdown(_llm_label(llm_state.value))
                prompt = gr.Textbox(label="What should the simulation be?", lines=4,
                                    placeholder="e.g. " + EXAMPLES[0])
                gr.Examples(EXAMPLES, inputs=prompt)
                with gr.Row():
                    robot = gr.Dropdown(["let the AI decide", "use the Roadster car", "drone", "legged robot",
                                         "robot arm", "wheeled robot", "rocket"], value="let the AI decide",
                                        label="Agent type")
                    gravity = gr.Dropdown(["Earth", "Moon", "Mars", "zero-g (space)"], value="Earth", label="Gravity")
                    difficulty = gr.Dropdown(["gentle", "normal", "hard"], value="normal", label="Difficulty")
                    budget = gr.Dropdown(["quick (~10 min CPU)", "medium (~30 min)", "long (1h+)"],
                                         value="quick (~10 min CPU)", label="Training budget")
                gen_btn = gr.Button("Generate simulation", variant="primary")
                gen_log = gr.Textbox(label="Progress", lines=8, interactive=False)

            # ---------------------------------------------------------- Review
            with gr.Tab("2 · Review", id="review"):
                status = gr.Markdown("No scenario generated yet.")
                with gr.Row():
                    preview = gr.Image(label="Preview (random actions)", type="filepath", height=300)
                    with gr.Column():
                        summary = gr.Markdown()
                        controls = gr.Markdown()
                reward_md = gr.Markdown(label="Reward design")
                with gr.Accordion("Generated code (read before training, editable)", open=False):
                    xml_code = gr.Code(label="model.xml", language="html", lines=20)
                    py_code = gr.Code(label="scenario.py", language="python", lines=30)
                    recheck_btn = gr.Button("Re-check my edits")
                with gr.Row():
                    change_req = gr.Textbox(label="Ask for a change", placeholder="make the wind stronger, add a second ship...",
                                            scale=4)
                    change_btn = gr.Button("Revise", scale=1)
                with gr.Row():
                    regen_btn = gr.Button("Regenerate from scratch")
                    accept_btn = gr.Button("Accept & go to training", variant="primary")
                review_log = gr.Textbox(label="Log", lines=4, interactive=False)

            # ----------------------------------------------------------- Train
            with gr.Tab("3 · Train", id="train"):
                with gr.Row(equal_height=True):
                    scenario_dd = gr.Dropdown(list_scenarios(), label="Scenario", value=SAMPLE_SCENARIO,
                                              allow_custom_value=False, scale=3)
                    refresh_btn = gr.Button("↻", scale=0, min_width=48)
                    algo_dd = gr.Dropdown(["ppo", "sac", "td3", "a2c", "dqn"], value="ppo", label="Algorithm", scale=1)
                    curriculum_cb = gr.Checkbox(True, label="Difficulty curriculum", scale=1)
                    window_cb = gr.Checkbox(False, label="Also open interactive 3D window", scale=1)
                steps_sl = gr.Slider(50_000, 5_000_000, value=300_000, step=50_000, label="Training steps")
                with gr.Row():
                    start_btn = gr.Button("Start training", variant="primary")
                    resume_btn = gr.Button("Resume unfinished run")
                    stop_btn = gr.Button("Stop (progress is saved)")
                with gr.Row(equal_height=True):
                    watch_dd = gr.Dropdown([], label="…or watch a run started from the terminal", scale=4,
                                           allow_custom_value=True)
                    watch_btn = gr.Button("Watch live", scale=0, min_width=120)
                train_md = gr.Markdown()

                gr.Markdown("### 🔴 Live 3D: watch it learn right now\nThe current, still-learning agent drives "
                            "non-stop in a spare copy of the world (updates about 2× a second, costs roughly 5–10% "
                            "training speed). Tick *interactive 3D window* to rotate and zoom with the mouse "
                            "(that window uses more CPU).")
                with gr.Row():
                    live_now = gr.Image(label="LIVE", type="pil", height=380, scale=3)
                    live_now_md = gr.Markdown("Start a training run to watch it live.")
                gr.Markdown("### Trial and error so far\nEvery ~25k steps one full attempt is saved as a clip, "
                            "so you can see it go from crashing to winning.")
                with gr.Row():
                    live_clip = gr.Image(label="Latest attempt (3D)", type="filepath", height=300, scale=3)
                    with gr.Column(scale=2):
                        live_md = gr.Markdown("Start a training run to see attempts here.")
                        win_plot = gr.LinePlot(pd.DataFrame({"steps": [], "value": [], "series": []}), x="steps",
                                               y="value", color="series", title="Win rate of training episodes",
                                               height=200, x_title="environment steps", y_title="% won",
                                               y_lim=[0, 100])
                live_gallery = gr.Gallery(label="All attempts, newest first (✅ won · ❌ failed)", columns=4,
                                          height=360, object_fit="contain", preview=False)
                plot = gr.LinePlot(pd.DataFrame({"steps": [], "value": [], "series": []}), x="steps", y="value",
                                   color="series", title="Reward during training", height=280,
                                   x_title="environment steps", y_title="mean episode reward")
                timer = gr.Timer(3.0, active=False)
                live_timer = gr.Timer(0.5, active=False)

            # --------------------------------------------------------- Results
            with gr.Tab("4 · Results", id="results"):
                with gr.Row(equal_height=True):
                    run_dd = gr.Dropdown([], label="Run", allow_custom_value=True, scale=4)
                    runs_refresh = gr.Button("↻", scale=0, min_width=48)
                    record_btn = gr.Button("Record video of the trained agent", variant="primary")
                result_gif = gr.Image(label="Trained agent", type="filepath", height=420)
                result_files = gr.File(label="Download model / clip / scenario", file_count="multiple")
                result_log = gr.Textbox(label="Log", lines=3, interactive=False)

            # -------------------------------------------------------- Settings
            with gr.Tab("Settings", id="settings"):
                gr.Markdown("### AI model that writes your simulations\nUse **any provider you have an API key for**. "
                            "No key? Pick **Ollama** to run a free model on your own PC, or get a free-tier key from "
                            "Google Gemini or Groq. A chat subscription (ChatGPT Plus, Claude Pro, ...) is not an API "
                            "key: keys come from each provider's developer console. Keys typed here are used for this "
                            "browser session only, unless you tick *save to .env* (TrainLab/.env is git-ignored). "
                            "Training and the sample Roadster never need a key.")
                with gr.Row():
                    provider_dd = gr.Dropdown(PROVIDER_CHOICES, value=start_provider, label="Provider", scale=2)
                    model_dd = gr.Dropdown(PROVIDERS[start_provider].models, value=default_model(start_provider),
                                           label="Model", allow_custom_value=True, scale=3)
                    models_btn = gr.Button("Load models", scale=0, min_width=130)
                provider_info = gr.Markdown(_provider_md(start_provider))
                key_box = gr.Textbox(label="API key", type="password", placeholder="paste your key (leave empty to use .env)")
                base_url_box = gr.Textbox(label="Base URL (only for Ollama on another port/PC or a custom server)",
                                          value=env_base_url(start_provider) or "",
                                          visible=start_provider in ("ollama", "custom"))
                save_cb = gr.Checkbox(False, label="save to .env")
                key_btn = gr.Button("Use these settings", variant="primary")
                settings_log = gr.Markdown()
                mesh = get_mesh_provider()
                gr.Markdown(f"### Optional: realistic 3D meshes\nText-to-3D provider: **{mesh.name}** "
                            f"({'ready' if mesh.available() else 'not configured'}). Set TEXT_TO_3D_PROVIDER, "
                            "TEXT_TO_3D_API_KEY and TEXT_TO_3D_ENDPOINT in .env and wire up "
                            "`trainlab/generator/mesh_provider.py`. Physics always uses simple shapes.")

        # ============================================================ logic
        def on_provider(provider):
            info = PROVIDERS[provider]
            return (gr.update(choices=info.models, value=default_model(provider)), _provider_md(provider),
                    gr.update(visible=provider in ("ollama", "custom"), value=env_base_url(provider) or ""))

        provider_dd.change(on_provider, [provider_dd], [model_dd, provider_info, base_url_box])

        def load_models(provider, key, base_url):
            try:
                models = get_provider(provider, model="probe", api_key=(key or "").strip() or None,
                                      base_url=(base_url or "").strip() or None).list_models()
            except (MissingKeyError, GenerationError) as e:
                return gr.update(), f"🔑 {e}"
            except Exception as e:  # network / auth / provider without a models endpoint
                return gr.update(), f"❌ Could not list models: {e}"
            if not models:
                return gr.update(), "The provider returned no models; type the model name instead."
            return gr.update(choices=models), f"Found {len(models)} models. Pick one in the Model box."

        models_btn.click(load_models, [provider_dd, key_box, base_url_box], [model_dd, settings_log])

        def apply_settings(provider, model, key, base_url, save):
            llm = {"provider": provider, "model": (model or "").strip(), "key": (key or "").strip(),
                   "base_url": (base_url or "").strip()}
            msg = f"Using **{PROVIDERS[provider].label} · {llm['model'] or '?'}** for this session."
            if save:
                msg += f" Saved to `{save_settings_to_env(provider, llm['key'], llm['model'], llm['base_url'])}`."
            return llm, msg, _llm_label(llm), _provider_md(provider)

        key_btn.click(apply_settings, [provider_dd, model_dd, key_box, base_url_box, save_cb],
                      [llm_state, settings_log, llm_md, provider_info])

        def do_generate(text, robot_v, gravity_v, diff_v, budget_v, llm):
            from trainlab.generator.pipeline import GenerationSession

            log: list[str] = []
            if not (text or "").strip():
                yield None, "Type a description first.", *([gr.update()] * 7)
                return
            try:
                session = GenerationSession(provider=llm["provider"], model=llm["model"] or None,
                                            api_key=llm["key"] or None, base_url=llm["base_url"] or None,
                                            log=log.append)
                options = {"agent type": None if robot_v == "let the AI decide" else robot_v,
                           "gravity": gravity_v, "difficulty": diff_v, "training budget": budget_v}
                log.append("Designing the world, the reward and the curriculum (this takes 1-3 minutes) ...")
                yield None, "\n".join(log), *([gr.update()] * 7)
                session.create(text, options)
                log.append("Done! Open the Review tab.")
                yield session, "\n".join(log), *_review_outputs(session)
            except MissingKeyError as e:
                yield None, f"🔑 {e}", *([gr.update()] * 7)
            except GenerationError as e:
                log.append(f"❌ {e}")
                yield None, "\n".join(log), *([gr.update()] * 7)

        review_outputs = [status, preview, summary, controls, reward_md, xml_code, py_code]
        gen_btn.click(do_generate, [prompt, robot, gravity, difficulty, budget, llm_state],
                      [session_state, gen_log, *review_outputs])
        regen_btn.click(do_generate, [prompt, robot, gravity, difficulty, budget, llm_state],
                        [session_state, review_log, *review_outputs])

        def do_revise(session, feedback):
            if session is None:
                return session, "Generate a scenario first.", *([gr.update()] * 7)
            try:
                session.revise(feedback)
                return session, "Revised and re-checked.", *_review_outputs(session)
            except (GenerationError, MissingKeyError) as e:
                return session, f"❌ {e}", *_review_outputs(session)

        change_btn.click(do_revise, [session_state, change_req], [session_state, review_log, *review_outputs])

        def do_recheck(session, xml, py):
            if session is None:
                return session, "Generate a scenario first.", *([gr.update()] * 7)
            errors = session.apply_edits(xml, py)
            msg = "✅ edits pass all checks" if not errors else "❌ " + "\n".join(errors[:5])
            return session, msg, *_review_outputs(session)

        recheck_btn.click(do_recheck, [session_state, xml_code, py_code], [session_state, review_log, *review_outputs])

        def do_accept(session):
            if session is None:
                return "Generate a scenario first.", gr.update(), gr.update()
            try:
                name = session.accept()
            except GenerationError as e:
                return f"❌ {e}", gr.update(), gr.update()
            return (f"Saved as scenario '{name}'. Try it yourself: trainlab play {name}",
                    gr.update(choices=list_scenarios(), value=name), gr.Tabs(selected="train"))

        accept_btn.click(do_accept, [session_state], [review_log, scenario_dd, tabs])

        def refresh_scenarios():
            load_all(reload_generated=True)
            return gr.update(choices=list_scenarios())

        refresh_btn.click(refresh_scenarios, None, scenario_dd)
        sample_btn.click(lambda: (gr.update(value=SAMPLE_SCENARIO), gr.Tabs(selected="train")), None,
                         [scenario_dd, tabs])

        def start_training(state, scenario, algo, steps, curriculum, window):
            if state.get("proc") is not None and state["proc"].poll() is None:
                return state, "A training run is already going. Stop it first.", gr.update(), gr.Timer(active=True)
            if not scenario:
                return state, "Pick a scenario.", gr.update(), gr.Timer(active=False)
            run_dir = RUNS_DIR / scenario / f"{algo}_studio_{time.strftime('%Y%m%d-%H%M%S')}"
            cmd = [sys.executable, "-m", "trainlab.cli", "train", scenario, "--algo", algo,
                   "--steps", str(int(steps)), "--run-dir", str(run_dir), "--spectate"]
            if not curriculum:
                cmd.append("--no-curriculum")
            if window:
                cmd.append("--view")
            run_dir.mkdir(parents=True, exist_ok=True)
            log = open(run_dir / "studio_train.log", "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            state = {"proc": proc, "run_dir": run_dir}
            return state, f"Started. Logs: `{run_dir / 'studio_train.log'}`", gr.update(), gr.Timer(active=True)

        start_btn.click(start_training, [train_state, scenario_dd, algo_dd, steps_sl, curriculum_cb, window_cb],
                        [train_state, train_md, plot, timer]).then(lambda: gr.Timer(active=True), None, live_timer)

        def resume_training(state):
            from trainlab.agents.train import find_unfinished

            if state.get("proc") is not None and state["proc"].poll() is None:
                return state, "A training run is already going.", gr.Timer(active=True)
            run_dir = find_unfinished()
            if run_dir is None:
                return state, "Nothing to resume: every run has finished.", gr.Timer(active=False)
            log = open(run_dir / "studio_train.log", "a")
            proc = subprocess.Popen([sys.executable, "-m", "trainlab.cli", "resume", str(run_dir), "--spectate"],
                                    stdout=log, stderr=subprocess.STDOUT)
            return {"proc": proc, "run_dir": run_dir}, f"Resuming `{run_dir}`", gr.Timer(active=True)

        resume_btn.click(resume_training, [train_state], [train_state, train_md, timer]).then(
            lambda: gr.Timer(active=True), None, live_timer)

        def stop_training(state):
            if state.get("run_dir"):
                (Path(state["run_dir"]) / "STOP").touch()
                return "Stopping after the current update (the model is saved)..."
            return "Nothing is running."

        stop_btn.click(stop_training, [train_state], [train_md])

        def poll(state):
            run_dir, proc = state.get("run_dir"), state.get("proc")
            if not run_dir:
                return gr.update(), gr.update(), *([gr.update()] * 4), gr.Timer(active=False)
            run_dir = Path(run_dir)
            if proc is not None:
                done = proc.poll() is not None
            else:  # a run started from the terminal: it's done once run.json says so
                meta = run_dir / "run.json"
                done = meta.exists() and "finished" in json.loads(meta.read_text())
            return (_run_summary(run_dir, proc), _read_progress(run_dir), *_live_outputs(run_dir),
                    gr.Timer(active=not done))

        live_outputs = [live_clip, live_md, win_plot, live_gallery]
        timer.tick(poll, [train_state], [train_md, plot, *live_outputs, timer])

        def poll_live(state):
            run_dir = state.get("run_dir")
            if not run_dir:
                return gr.update(), gr.update(), gr.Timer(active=False)
            frame, md, finished = _live_frame(Path(run_dir))
            return (frame if frame is not None else gr.update()), md, gr.Timer(active=not finished)

        live_timer.tick(poll_live, [train_state], [live_now, live_now_md, live_timer])

        def watch_run(run):
            if not run:
                return {"proc": None, "run_dir": None}, "Pick a run.", gr.Timer(active=False)
            return {"proc": None, "run_dir": Path(run)}, f"Watching `{run}`", gr.Timer(active=True)

        watch_btn.click(watch_run, [watch_dd], [train_state, train_md, timer]).then(
            lambda: gr.Timer(active=True), None, live_timer)
        tabs.select(lambda: gr.update(choices=[str(p.parent) for p in sorted(
            RUNS_DIR.glob("*/*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)]), None, watch_dd)

        def list_runs():
            runs = sorted((p.parent for p in RUNS_DIR.glob("*/*/run.json")), key=lambda p: p.stat().st_mtime,
                          reverse=True)
            return gr.update(choices=[str(r) for r in runs], value=str(runs[0]) if runs else None)

        runs_refresh.click(list_runs, None, run_dd)
        tabs.select(lambda: list_runs(), None, run_dd)

        def record_run(run):
            if not run:
                return None, None, "Pick a run."
            run = Path(run)
            out = run / "clip.gif"
            cmd = [sys.executable, "-m", "trainlab.cli", "record", str(run), "--out", str(out), "--episodes", "2"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out.exists():
                return None, None, (proc.stderr or proc.stdout)[-1500:]
            files = [str(p) for p in (run / "best_model.zip", run / "final_model.zip", run / "vecnormalize.pkl", out)
                     if p.exists()]
            return str(out), files, proc.stdout.strip().splitlines()[-1] if proc.stdout else "recorded"

        record_btn.click(record_run, [run_dd], [result_gif, result_files, result_log])
    return demo


def launch(port: int = 7860, share: bool = False, open_browser: bool = True) -> None:
    build().queue().launch(server_name="127.0.0.1", server_port=port, share=share, inbrowser=open_browser)
