# TrainLab: briefing for a new AI agent

You're joining an existing project called **TrainLab**. Read this whole briefing before
touching any code. The owner will give you instructions **one step at a time**: do only
the step you're given, report back, and wait for the next one.

## 1. What TrainLab is

TrainLab is a general tool for training reinforcement-learning (RL) agents in 3D
simulations. It takes the "Lunar Lander + reward shaping" tutorial idea, moves it into full
3D physics, and makes it work for any simulation. It's a beginner-friendly AI resume project.

- **The default sample** is a Roadster-style electric sports car with a cold-gas thruster
  pack (Starman rides along). It's a fan project with no connection to Tesla or SpaceX.
- **Any scenario** is one Python file (four hooks) plus one MuJoCo XML model.
- **Studio** is a Gradio web app. The user types an idea, an LLM of *their* choice builds
  the simulation, TrainLab validates it, trains an agent, and shows live 3D attempts and videos.

## 2. Machine and environment (important)

- **Machine:** Windows 11 laptop, Ryzen 5 5600H (6 cores), **8 GB RAM** (usually only
  0.4–0.8 GB free with a browser open), 1 TB disk, **no NVIDIA GPU** (CPU-only PyTorch).
- **The laptop must never freeze.** Run one heavy job at a time. Never run training and a
  big local LLM together.
- **Python:** `C:\Users\JESUS\AppData\Local\Programs\Python\Python311\python.exe`, or `py -3.11`.
  Plain `python` on PATH points at an unrelated venv (hermes-agent). Don't use it.
- **Install:** `pip install -e ".[studio,dev]"` from `D:\MY works\TrainLab`.
- **Run:** `py -3.11 -m trainlab.cli <command>`.
- **Tests:** `py -3.11 -m pytest -q -p no:cacheprovider`. 53 tests pass today.
- **PowerShell 5.1 quirks:**
  - Inline `python -c "..."` scripts lose their quotes. Write scripts to files instead.
  - `Get-Content`/`Set-Content` re-encode UTF-8 (they turned `·` into `Â·`). Edit files
    with Python or an editor, not PowerShell text replacement.
- **Not a git repo.** Don't initialise git unless the owner asks.
- **Versions:** MuJoCo 3.17, Gymnasium 1.4, Stable-Baselines3 2.9, torch 2.11 (CPU),
  Gradio 6.30, anthropic 2.3, openai 2.32, pygame, imageio, psutil.
- **Ollama 0.40** is installed locally with gemma4 (8B, about 7.4 GB loaded, too big for
  8 GB: it swaps), mistral and tinyllama.

## 3. Code map (`D:\MY works\TrainLab\trainlab\`)

### core/
- `base_env.py`: `MujocoScenarioEnv(gym.Env)`.
  - **Hooks:** `build_xml` (or `xml_file`), `setup`, `randomize(params)`, `get_obs`,
    `check_done() -> (terminated, success, reason)`, and
    `compute_reward(action, terminated, success) -> (reward, terms_dict)`.
  - **Optional hooks:** `apply_action`, `substep_control` (runs every physics step), `action_dim`.
  - **Timing:** 0.005 s physics timestep, `frame_skip=10`, so control runs at 20 Hz.
  - **`CURRICULUM`:** a list of dicts, one per level. Values are fixed numbers or
    `[lo, hi]` ranges sampled at every reset. `set_level` / `get_level` switch levels.
  - **Wind:** an Ornstein-Uhlenbeck gust model plus air drag on `wind_body`.
  - **Rendering:** `rgb_array` uses `mujoco.Renderer`, with size set by
    `render_width`/`render_height` (not `width`, which TrackRace uses for the track).
    `human` uses the MuJoCo passive viewer.
  - **`set_box(body_name, pos, quat, half_size)`** moves or resizes obstacles. The body
    **must be `mocap="true"` and hold exactly one box geom**. Moving geoms of the static
    world body breaks MuJoCo's precomputed collision tree, and the car falls through.
  - **Helpers:** `body`, `geom`, `joint`, `actuator` (ids), `body_velocity`, `contacts()`.
- `registry.py`: `@register_scenario("snake_name")` registers the class with Gymnasium as
  `TrainLab/<Camel>-v0`. It auto-loads `scenarios/*.py` (except `_*.py`) and
  `scenarios/generated/<name>/scenario.py` (except `_drafts`).
- `rewards.py`: `PotentialShaper` (reward = Φ(now) − Φ(before)) and `braking_envelope`.
- `curriculum.py`:
  - `CurriculumCallback` raises the level when the evaluation success rate reaches 80%.
  - `SaveVecNormalizeCallback`.
  - `AutosaveCallback` saves to `run/autosave/` every 20k steps, atomically.
  - `StopFileCallback` stops training when a STOP file appears in the run folder.
- `live.py`: `LiveViewCallback`.
  - Every 25k steps, the still-learning policy plays one rendered attempt. It's saved to
    `run/live/attempt_NNNN.gif/.png` with a ✅ SUCCESS or ❌ FAILED banner. GIFs are small:
    at most 120 frames and a 128-colour palette.
  - It also tallies every training episode by end reason and keeps a win-rate timeline,
    written to `stats.json` and `attempts.json`.
- `resources.py`:
  - `profile()` sets 8 GB machines to 2 parallel envs, 2 torch threads, and a pause below
    0.2 GB free RAM.
  - `be_nice()` sets below-normal process priority.
  - `MemoryGuardCallback` pauses training (after an autosave) when free RAM is low, and
    stops cleanly after 10 minutes.
- `factory.py`: `EnvFactory`, the picklable env constructor. It must stay light: **no torch
  or SB3 imports**, because subprocess workers import it and each torch import costs ~300 MB.
- `wind.py`, and `wrappers.py` (`DiscretizeAction` lets DQN run on continuous scenarios).

### scenarios/
- `_roadster.py`: `RoadsterMixin` plus `roadster_xml(world, gravity, start, ...)`, which fills
  `assets/roadster.xml` using `string.Template` placeholders.
  - **Controls:** every control is in [-1, 1]: `throttle steer brake lift pitch roll yaw
    thrust_x thrust_y thrust_z`. `brake` and `lift` are one-sided (0 means off). A scenario
    chooses its controls with `CONTROLS`.
  - **Car model:** about 4.2 × 1.9 m and ~1,900 kg, with a low battery ballast for a low
    centre of mass and all-wheel drive (1,500 N·m per wheel). Front wheels steer through a
    stiff position servo (kp 30,000). Each wheel has slide-joint suspension.
  - **Wheel control:** traction control and anti-lock braking run in `substep_control`, at
    200 Hz, because at 20 Hz the wheels spin up.
  - **Thrusters:** site actuators on `cg` (lift 32 kN, torques of 6–9 kN·m). They burn fuel,
    with capacity set by `FUEL_SECONDS`.
  - **Collision classes:** `world` collides with `car`; `deco` never collides.
  - **Contact settings:** elliptic friction cones (impratio 5), and wheel contacts use
    margin/gap 0.03 so the wheel-touch sensor doesn't flicker.
  - **Helpers:** `place_car`, `car_pos`, `car_vel`, `car_vel_body`, `car_angvel_body`,
    `up_z`, `heading`, `car_obs()` (16 values), `wheel_contacts()`, `shell_hit()`,
    `wheel_slip()`, `fuel_frac()`.
- `roadster_gap_jump.py`: **the default sample** (`SAMPLE_SCENARIO` in `cli.py`).
  - **Task:** speed up, cross a pit (thrusters for big gaps), land on a ledge (a mocap
    body), and park in a green box.
  - **Six curriculum levels.** Level 0 is a 0.3–1 m crack. Before that level existed, PPO
    learned to stop at the pit edge and never jump.
  - **Reward:** potential shaping on distance, tilt and speed over the braking envelope;
    fuel and time costs; +100 plus a fuel bonus for parking; −100 for a crash (body_hit,
    flipped, fell, hard_landing, overshot).
- `roadster_launch_sprint.py`: stop exactly on a line, with random grip and wind.
- `roadster_track_race.py`: a random closed 3D track every episode, built from 60 mocap
  slab segments with hills and banking. Observations include 5 look-ahead centreline points.
- `roadster_space_dock.py`: zero gravity, 6-DoF thrusters, dock with a station. It needs
  `<statistic extent="20"/>` because a huge Earth sphere had pushed the camera's near
  clip plane past the car.
- `_template.py` with `assets/template_model.xml`: a ball-pusher starter used by `trainlab new`.

### agents/
- `train.py`: `train(...)` builds SubprocVecEnv → VecMonitor → VecNormalize.
  - **Callbacks:** EvalCallback (best model), curriculum, checkpoints, autosave, memory
    guard, stop file, and live view.
  - **Output:** each run writes `runs/<scenario>/<run>/run.json`.
  - **Resuming:** `latest_save`, `is_unfinished`, `find_unfinished` and `resume_run`.
    `resume` continues from the newest autosave or final save.
- `runner.py`: `TrainedPolicy`, which loads a model with its matching VecNormalize stats,
  and `run_episodes`.
- `compare.py`: runs several algorithms across seeds and plots `compare.png` and `summary.csv`.

### Other modules
- `play/keyboard.py`: pygame manual driving. Scores go to `runs/<scenario>/human_scores.csv`.
- `record.py`: GIF/MP4 export with a HUD.
- `generator/`:
  - `llm_client.py`: providers anthropic, openai, gemini, groq, openrouter, deepseek,
    mistral, ollama and custom.
    - Anthropic uses its own SDK (default model `claude-opus-5-5`, adaptive thinking,
      JSON-schema output, and `fallbacks="default"` behind the
      `server-side-fallback-2026-07-01` beta).
    - Ollama uses its native `/api/chat` endpoint with `num_ctx` 12,288, JSON-schema
      output, thinking off and `keep_alive` 2m. Its OpenAI-compatible endpoint can't raise
      the context size, and the default silently truncated the prompt. It warns when a model
      is too big for the RAM.
    - All other providers go through the OpenAI-compatible API, falling back from JSON
      schema to JSON object to plain text.
    - Keys live in `.env`; see `.env.example`.
  - `prompts/system.md`: the scenario contract the LLM follows, with GapJump as a worked example.
  - `validate.py`:
    - An AST allowlist: imports only from numpy, math and trainlab.core / `_roadster`; no
      os, files, eval or dunder attributes.
    - XML rules, and a runtime check in a **separate process with a timeout**.
  - `check_scenario.py`: the child process that runs the Gymnasium env checker, 200
    random steps, and saves `preview.png` and `preview.gif`.
  - `pipeline.py`: `GenerationSession` drafts into `generated/_drafts/<name>`, repairs
    automatically (up to 3 rounds), and only becomes a real scenario on accept.
  - `scaffold.py`, and `mesh_provider.py` (a placeholder for text-to-3D meshes).
- `studio/app.py`:
  - **Tabs:** Create, Review (code you can edit), Train, Results, Settings.
  - **Train tab:** a live trial-and-error panel (latest 3D attempt, scoreboard, win-rate
    chart, gallery), Watch live for runs started from the terminal, a Resume button, and Stop.
  - **Sample banner:** "Train the sample Roadster".
- `cli.py`: `list`, `play`, `train`, `resume`, `status`, `watch`, `record`, `compare`,
  `new`, `generate`, and `studio`.
- `tests/`: pytest covering env checks, determinism, physics sanity, validator safety,
  providers, live view and resume.

### Body + Task framework (added after the first handoff)

- `bodies/base.py`: the `Body` base class. It holds state helpers (pos, rot, vel, up_z,
  world_contacts) and hooks: `xml(start)`, `bind`, `place`, `apply_action`,
  `substep_control`, `obs`, `crash_reason`, `effort`, `update_visuals`.
- `tasks/base.py`:
  - `Task` base class: `world_xml`, `actuators_xml`, `randomize`, `obs`, `check_done`,
    `compute_reward`, and `substep(t)` for moving platforms. `span(key)` returns the raw
    curriculum range; `randomize()` receives each range already sampled once.
  - `BodyTaskEnv` and `body_task_env(body, task)` build the scenario class.
  - Shared MJCF: `assets/world_template.xml`.
- `bodies/drone.py`: the `DroneBody`.
  - A 1.16 kg X-frame quad, 4 site actuators (7 N max each, plus a yaw drag torque).
  - Controls: thrust, roll, pitch, yaw in "angle mode". An attitude PD loop plus a motor
    mixer run in `substep_control` at 200 Hz.
  - `DroneFX` (render only) spins the props and adds downwash dust.
- `tasks/waypoints.py` and `tasks/landing.py`.
  - The landing pad is a **real body moved by stiff position actuators (gravcomp)**, not a
    mocap. With mocap, a moving deck didn't carry the drone, because mocap surfaces have
    zero velocity for friction.
- `scenarios/body_tasks.py`: the PAIRS list, registered as `drone_waypoints` and `drone_landing`.
- **Verified:** a scripted autopilot wins all 4 levels of both tasks (48/48), including the
  moving ship deck. 65 tests pass.
- `bodies/rover.py`: the `RoverBody`.
  - About 61 kg, 6 wheels with slide suspension, skid steer, and a wheel-speed controller in
    `substep_control`. Controls: drive, turn. Top speed 2.5 m/s.
  - `LaserEyes` (in `bodies/base.py`): 7 `mj_ray` beams that only see group 0 (the world).
    The beams are drawn through **mocap bodies**: MuJoCo re-reads a world-body geom's
    position but not its rotation.
- `WaypointsTask` adapts to the body. `curriculum_for(body)` returns the flying table or
  `GROUND_CURRICULUM`; `start_for(body)` uses `body.spawn_z`. Ground targets sit at spawn
  height, distance is 2D, and levels 2–3 place up to 12 mocap ellipsoid rocks near the
  route. Body traits: `flies`, `spawn_z`, `reach`.
- **Rover autopilot results:** levels 0–1 6/6, level 2 8/10, level 3 3/10 (hard but winnable). 71 tests pass.
- **Next:** a Roadster body adapter (the existing roadster_* scenarios are
  standalone and still use `_roadster.py`), then make the Studio generator write only the
  task for a chosen library body.

## 4. Current results

- **Sample run** (`runs/roadster_gap_jump/ppo_studio_20261025-074748`): PPO, 300k steps,
  about 7 minutes. Reward rose from −110 to +125. Level 0 parks 100% of the time, and the
  curriculum reached level 1, which isn't solved yet. `level0_model.zip` parks 2 of 2 times
  (see `sample_level0.gif`). It still burns all its thruster fuel.
- **Speed:** about 600–900 environment steps per second with 2 envs on this laptop.
- **LLM generation:** gemma4 via Ollama wrote a full scenario in about 7.5 minutes, but
  round 2 was interrupted, so it isn't verified end to end. No cloud provider has been tested.

## 5. Next phase (what we're about to build, step by step on the owner's instructions)

**Goal:** make the Roadster robust through variety. One fixed road just gets memorized, so
every episode should be a new random layout.

### A. Laser "eyes" (build first; both scenarios use them)
- About 11 forward ray sensors (`mujoco.mj_ray`, or MuJoCo rangefinder sensors), fanned
  across roughly ±60°, plus 2–3 rays angled down to spot gaps and ramps.
- Normalised distances become part of the observation.
- Put this in a reusable helper, for example `RoadsterMixin.ray_obs()`, and ignore the
  car's own geoms when casting.

### B. `roadster_obstacle_course`
- Each episode procedurally builds a road of about 200 m with random obstacles, all mocap
  bodies through `set_box` and pre-declared in the XML (for example about 30 slots):
  - traffic cones and barriers to steer around
  - speed bumps and potholes
  - kicker ramps (tilted boxes)
  - gaps
  - walls to hop over with the lift thrusters
  - at the top level, a moving crossing obstacle
- **Goal:** reach the finish zone and stop in it.
- **Reward:** potential shaping on progress along the road, small costs for fuel and time,
  a crash penalty, and a finish bonus.
- **Curriculum:** (0) a few cones → (1) bumps and ramps → (2) gaps → (3) a dense mixed
  course with wind.

### C. `roadster_ramp_flip` (a stunt like the one the owner saw at the Roadster reveal)
- A curved quarter-pipe ramp, built from mocap slab segments like TrackRace, launches the car.
- In the air, the agent uses the pitch and roll thrusters to complete a flip, lands on all
  four wheels, and stops in a landing zone.
- **Reward:** rotation completed while airborne (accumulated pitch), an upright landing, a
  soft touchdown and stopping in the zone. A roof landing or crash gives −100.
- **Curriculum:** (0) a clean jump and landing → (1) a higher ramp → (2) a half rotation → (3) a full flip.

### D. Demonstrations ("more training data")
- Record the owner's keyboard runs from `trainlab play` as (obs, action) pairs.
- Pre-train the policy with behaviour cloning, then fine-tune with PPO.

### Constraints for this phase
- Keep the 8 GB profile: 2 envs, low priority, autosave and resume, one heavy job at a time.
- Expect about 2–6M steps (hours). Use autosave plus `trainlab resume` to run overnight or
  across several sessions.
- The new observations change the network's input size, so the GapJump model can't be
  loaded directly. Train from scratch, using the easy early levels.

## 6. How to work with the owner

- Talk simply and briefly. The owner is a beginner and writes informally.
- Do exactly the step you're given, run the tests, check the result yourself (render a
  frame, run a short training run), and report honestly what works and what doesn't.
- Before any training longer than a few minutes, say how long it will take on this laptop.
