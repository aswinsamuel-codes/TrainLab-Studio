import json

from trainlab.agents import train as tr
from trainlab.core import resources


def test_profile_for_8gb_laptop(monkeypatch):
    monkeypatch.setattr(resources, "total_ram_gb", lambda: 7.3)
    p = resources.profile()
    assert p["n_envs"] == 2 and p["torch_threads"] == 2 and p["min_free_gb"] <= 0.25


def test_profile_for_bigger_machines(monkeypatch):
    monkeypatch.setattr(resources, "total_ram_gb", lambda: 16)
    assert resources.profile()["n_envs"] == 4
    monkeypatch.setattr(resources, "total_ram_gb", lambda: 64)
    assert resources.profile()["n_envs"] >= 2


def _fake_run(root, steps_target, autosave_steps=None, final_steps=None):
    run = root / "roadster_gap_jump" / "ppo_fake"
    (run / "autosave").mkdir(parents=True)
    meta = {"scenario": "roadster_gap_jump", "algo": "ppo", "steps": steps_target, "seed": 0, "level": 0}
    if final_steps is not None:
        meta["timesteps"] = final_steps
        (run / "final_model.zip").write_bytes(b"x")
    (run / "run.json").write_text(json.dumps(meta))
    if autosave_steps is not None:
        (run / "autosave" / "model.zip").write_bytes(b"x")
        (run / "autosave" / "state.json").write_text(json.dumps({"timesteps": autosave_steps, "level": 1}))
    return run


def test_latest_save_prefers_newest(tmp_path):
    run = _fake_run(tmp_path, 100_000, autosave_steps=60_000, final_steps=40_000)
    model, _, steps, level = tr.latest_save(run)
    assert steps == 60_000 and level == 1 and model.parent.name == "autosave"


def test_unfinished_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "RUNS_DIR", tmp_path)
    run = _fake_run(tmp_path, 100_000, autosave_steps=20_000)  # laptop switched off mid-run
    assert tr.is_unfinished(run) and tr.find_unfinished() == run


def test_finished_run_is_not_resumed(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "RUNS_DIR", tmp_path)
    _fake_run(tmp_path, 100_000, autosave_steps=100_000, final_steps=100_352)
    assert tr.find_unfinished() is None
