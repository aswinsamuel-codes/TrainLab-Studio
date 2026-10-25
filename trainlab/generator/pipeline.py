"""Prompt -> scenario pipeline: generate with an LLM, validate, auto-repair, review, accept.

Drafts live in scenarios/generated/_drafts/<name>/ and only become a real scenario
(scenarios/generated/<name>/) when the user accepts them.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from trainlab.core.registry import GENERATED_DIR, load_all
from trainlab.generator import validate
from trainlab.generator.llm_client import GenerationError, get_provider

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
EXAMPLE = Path(__file__).resolve().parent.parent / "scenarios" / "roadster_gap_jump.py"
DRAFTS = GENERATED_DIR / "_drafts"
MAX_ROUNDS = 3


def system_prompt() -> str:
    example = "```python\n" + EXAMPLE.read_text(encoding="utf-8") + "\n```"
    return (PROMPT_DIR / "system.md").read_text(encoding="utf-8").replace("{example}", example)


def write_draft(data: dict) -> Path:
    folder = DRAFTS / data["name"]
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    (folder / "scenario.py").write_text(data["scenario_py"], encoding="utf-8")
    if data.get("mjcf_xml", "").strip():
        (folder / "model.xml").write_text(data["mjcf_xml"], encoding="utf-8")
    (folder / "config.yaml").write_text(data.get("config_yaml") or "", encoding="utf-8")
    meta = {k: data.get(k) for k in ("name", "title", "summary", "reward_explanation", "controls")}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def check_all(data: dict, folder: Path | None = None) -> tuple[list[str], dict]:
    errors = validate.check_name(data.get("name", ""))
    errors += validate.static_check_python(data.get("scenario_py", ""), data.get("name"))
    errors += validate.static_check_xml(data.get("mjcf_xml", ""))
    if errors:
        return errors, {}
    folder = folder or write_draft(data)
    report = validate.runtime_check(folder)
    return report.get("errors", []), report


class GenerationSession:
    """One conversation with the LLM about one scenario (create, then any number of revisions)."""

    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, log=print):
        self.provider = get_provider(provider, model, api_key, base_url)
        warn = getattr(self.provider, "memory_warning", lambda: None)()
        if warn:
            log(f"⚠️ {warn}")
        self.system = system_prompt()
        self.messages: list = []
        self.data: dict | None = None
        self.report: dict = {}
        self.log = log

    @property
    def folder(self) -> Path | None:
        return DRAFTS / self.data["name"] if self.data else None

    def create(self, prompt: str, options: dict | None = None) -> dict:
        text = f"Build this simulation:\n\n{prompt.strip()}"
        if options:
            text += "\n\nUser preferences:\n" + "\n".join(f"- {k}: {v}" for k, v in options.items() if v)
        return self._run(text)

    def revise(self, feedback: str) -> dict:
        if not self.data:
            raise GenerationError("Generate a scenario first.")
        return self._run(f"Please change the scenario: {feedback.strip()}\nReturn the full updated JSON.")

    def _run(self, user_text: str) -> dict:
        errors: list[str] = []
        for round_no in range(1, MAX_ROUNDS + 1):
            self.log(f"Round {round_no}: asking {self.provider.name} / {self.provider.model} ...")
            self.messages.append({"role": "user", "content": user_text})
            t0 = time.time()
            data, content = self.provider.generate(self.system, self.messages)
            self.messages.append({"role": "assistant", "content": content})
            self.log(f"  got '{data.get('name')}' in {time.time() - t0:.0f}s, checking ...")
            errors, report = check_all(data)
            self.data, self.report = data, report
            if not errors:
                self.log("  passed all checks.")
                return data
            self.log(f"  {len(errors)} problem(s), asking for a fix.")
            user_text = ("The scenario failed TrainLab's checks. Fix the root cause and return the full JSON.\n\n"
                         + "\n\n".join(f"- {e}" for e in errors))
        raise GenerationError("Still failing after %d rounds:\n%s" % (MAX_ROUNDS, "\n".join(errors[:5])))

    def apply_edits(self, mjcf_xml: str | None, scenario_py: str | None) -> list[str]:
        """Re-validate after the user hand-edits the code in the Review tab."""
        if mjcf_xml is not None:
            self.data["mjcf_xml"] = mjcf_xml
        if scenario_py is not None:
            self.data["scenario_py"] = scenario_py
        errors, self.report = check_all(self.data)
        return errors

    def accept(self) -> str:
        if not self.data or not self.report.get("ok"):
            raise GenerationError("Only a scenario that passed the checks can be accepted.")
        name = self.data["name"]
        target = GENERATED_DIR / name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(DRAFTS / name, target)
        load_all(reload_generated=True)
        return name


def generate_cli(prompt: str, provider: str | None = None, model: str | None = None,
                 base_url: str | None = None, name: str | None = None) -> int:
    from trainlab.generator.llm_client import MissingKeyError

    try:
        session = GenerationSession(provider=provider, model=model, base_url=base_url)
        options = {"scenario name": name} if name else None
        data = session.create(prompt, options)
    except (MissingKeyError, GenerationError) as e:
        print(f"\n{e}")
        return 1
    print(f"\n{data['title']}  ({data['name']})\n\n{data['summary']}\n")
    print("Controls:")
    for c in data["controls"]:
        print(f"  - {c['name']}: {c['meaning']}")
    print(f"\nReward design:\n{data['reward_explanation']}\n")
    print(f"Draft files (read them before training!): {session.folder}")
    print(f"Random-play check: {session.report.get('random_returns')}  preview: {session.folder / 'preview.gif'}")
    if input("\nAccept this scenario? [y/N] ").strip().lower() == "y":
        accepted = session.accept()
        print(f"Saved. Next:\n  trainlab play {accepted}\n  trainlab train {accepted} --steps 300000")
    return 0
