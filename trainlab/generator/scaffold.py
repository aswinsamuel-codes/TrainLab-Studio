"""`trainlab new <name>`: copy the starter template into scenarios/generated/<name>/."""

from __future__ import annotations

from pathlib import Path

from trainlab.core.registry import GENERATED_DIR, camel
from trainlab.generator.validate import check_name

PKG = Path(__file__).resolve().parent.parent


def new_scenario(name: str) -> Path:
    errors = check_name(name)
    if errors:
        raise SystemExit(errors[0])
    folder = GENERATED_DIR / name
    if folder.exists():
        raise SystemExit(f"{folder} already exists")
    folder.mkdir(parents=True)
    title = name.replace("_", " ").title()
    py = (PKG / "scenarios" / "_template.py").read_text(encoding="utf-8")
    py = py.replace("__NAME__", name).replace("__CLASS__", camel(name)).replace("__TITLE__", title)
    xml = (PKG / "assets" / "template_model.xml").read_text(encoding="utf-8").replace("__NAME__", name)
    (folder / "scenario.py").write_text(py, encoding="utf-8")
    (folder / "model.xml").write_text(xml, encoding="utf-8")
    (folder / "config.yaml").write_text("# training overrides, e.g.\n# algos:\n#   ppo:\n#     n_steps: 512\n",
                                        encoding="utf-8")
    return folder
