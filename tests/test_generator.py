from pathlib import Path

import pytest

from trainlab.generator import validate
from trainlab.generator.pipeline import EXAMPLE, system_prompt

GOOD = EXAMPLE.read_text(encoding="utf-8")


def test_example_passes_static_check():
    assert validate.static_check_python(GOOD, "roadster_gap_jump") == []


@pytest.mark.parametrize("snippet", [
    "import os",
    "import subprocess",
    "from pathlib import Path",
    "x = open('secrets.txt').read()",
    "eval('1+1')",
    "__import__('os')",
    "y = (1).__class__.__bases__",
    "getattr(object, 'x')",
])
def test_dangerous_code_is_rejected(snippet):
    errors = validate.static_check_python(GOOD + "\n" + snippet + "\n", "roadster_gap_jump")
    assert errors, snippet


def test_wrong_registered_name_is_rejected():
    assert validate.static_check_python(GOOD, "something_else")


def test_xml_rules():
    assert validate.static_check_xml('<mujoco><asset><mesh file="../../x.stl"/></asset></mujoco>')
    assert validate.static_check_xml('<mujoco><include file="a.xml"/></mujoco>')
    assert validate.static_check_xml('<mujoco><worldbody/></mujoco>')  # no chase camera
    assert validate.static_check_xml("") == []  # Roadster scenarios ship no XML


def test_names():
    assert validate.check_name("drone_ship_landing") == []
    assert validate.check_name("Drone Landing")
    assert validate.check_name("../evil")


def test_system_prompt_includes_example():
    text = system_prompt()
    assert "register_scenario(\"roadster_gap_jump\")" in text and "{example}" not in text


def test_template_scenario_runs_in_checker(tmp_path, monkeypatch):
    """`trainlab new` output must pass the same sandboxed check generated code gets."""
    from trainlab.generator import scaffold

    monkeypatch.setattr(scaffold, "GENERATED_DIR", tmp_path)
    folder = scaffold.new_scenario("ball_pusher_test")
    assert validate.static_check_python((folder / "scenario.py").read_text(), "ball_pusher_test") == []
    report = validate.runtime_check(folder)
    assert report["ok"], report["errors"]
    assert Path(folder / "preview.gif").exists()
