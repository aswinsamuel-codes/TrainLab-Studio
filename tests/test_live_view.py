import json

import numpy as np
from PIL import Image

from trainlab.core.live import _hud


def test_hud_draws_banner():
    frame = np.zeros((272, 480, 3), dtype=np.uint8)
    out = _hud(frame, "attempt #1", "FAILED: FELL", ok=False)
    assert out.shape == frame.shape and out[136, 240, 0] > 150  # red banner across the middle


def test_live_outputs_reads_run_folder(tmp_path):
    from trainlab.studio.app import _live_outputs

    live = tmp_path / "live"
    live.mkdir()
    Image.new("RGB", (48, 27)).save(live / "attempt_0001.png")
    Image.new("RGB", (48, 27)).save(live / "attempt_0002.png")
    (live / "attempts.json").write_text(json.dumps([
        {"attempt": 1, "steps": 0, "level": 0, "return": -100, "success": False, "reason": "fell",
         "gif": None, "png": "attempt_0001.png"},
        {"attempt": 2, "steps": 25000, "level": 0, "return": 150, "success": True, "reason": "parked",
         "gif": None, "png": "attempt_0002.png"},
    ]))
    (live / "stats.json").write_text(json.dumps({
        "steps": 25000, "wins": 3, "fails": 40, "by_reason": {"fell": 30, "time_up": 10, "parked": 3},
        "win_reasons": ["parked"], "level": 0,
        "timeline": [{"steps": 2000, "episodes": 5, "success_rate": 0.0},
                     {"steps": 25000, "episodes": 5, "success_rate": 0.6}]}))
    latest, md, df, gallery = _live_outputs(tmp_path)
    assert latest.endswith("attempt_0002.png")
    assert "WON" in md and "43" in md and "✅ parked" in md and "❌ fell" in md
    assert list(df["value"]) == [0.0, 60.0]
    assert gallery[0][1].startswith("✅ #2") and gallery[1][1].startswith("❌ #1")


def test_live_frame_states(tmp_path):
    import time

    from trainlab.studio.app import _live_frame

    live = tmp_path / "live"
    live.mkdir()
    (tmp_path / "run.json").write_text("{}")
    assert "Waiting" in _live_frame(tmp_path)[1]

    Image.new("RGB", (48, 27)).save(live / "frame.jpg")
    (live / "spectator.json").write_text(json.dumps({"episode": 4, "steps": 30000, "level": 0, "score": 12.5,
                                                     "wins": 1, "fails": 3, "result": None, "time": time.time()}))
    frame, md, finished = _live_frame(tmp_path)
    assert frame is not None and "LIVE" in md and "Try #4" in md and not finished

    (live / "paused.json").write_text(json.dumps({"free_gb": 0.05, "need_gb": 0.35, "since": time.time()}))
    assert "Paused" in _live_frame(tmp_path)[1]

    (tmp_path / "run.json").write_text(json.dumps({"finished": "now"}))
    assert _live_frame(tmp_path)[2] is True


def test_live_outputs_empty_run(tmp_path):
    from trainlab.studio.app import _live_outputs

    latest, md, df, gallery = _live_outputs(tmp_path)
    assert latest is None and gallery == [] and "Waiting" in md
