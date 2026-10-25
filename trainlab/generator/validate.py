"""Safety and sanity checks for generated scenarios.

Static checks run in-process (they only parse, never execute). Anything that executes
generated code runs in a separate Python process with a timeout (check_scenario.py).
This is a guard rail, not a hardened sandbox: always read generated code before training.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ALLOWED_IMPORTS = {"__future__", "numpy", "math", "trainlab.core", "trainlab.core.rewards",
                   "trainlab.scenarios._roadster"}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
                 "locals", "vars", "getattr", "setattr", "delattr", "memoryview", "help", "exit", "quit"}
ALLOWED_DUNDERS = {"__init__", "__future__", "__name__", "__doc__"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


def check_name(name: str) -> list[str]:
    return [] if NAME_RE.match(name or "") else [
        f"name '{name}' must be snake_case, 3-40 chars, starting with a letter"]


def static_check_python(source: str, expected_name: str | None = None) -> list[str]:
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"scenario.py syntax error line {e.lineno}: {e.msg}"]

    registered = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    errors.append(f"line {node.lineno}: import '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "") not in ALLOWED_IMPORTS or node.level:
                errors.append(f"line {node.lineno}: 'from {node.module} import' is not allowed")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            errors.append(f"line {node.lineno}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS:
            errors.append(f"line {node.lineno}: dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(f"line {node.lineno}: global/nonlocal is not allowed")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "register_scenario":
            if node.args and isinstance(node.args[0], ast.Constant):
                registered.append(node.args[0].value)

    if not registered:
        errors.append("no @register_scenario(\"...\") decorator found")
    elif expected_name and expected_name not in registered:
        errors.append(f"@register_scenario uses {registered}, expected '{expected_name}'")
    return errors


def static_check_xml(xml: str) -> list[str]:
    if not xml.strip():
        return []
    errors = []
    if re.search(r"\bfile\s*=", xml):
        errors.append("model.xml must not reference files (file=...); use builtin textures and primitives")
    if re.search(r"<\s*include\b", xml):
        errors.append("model.xml must not use <include>")
    if 'name="chase"' not in xml:
        errors.append('add a camera named "chase" on the controlled body (mode="trackcom")')
    return errors


def runtime_check(folder: Path, timeout: float = 180.0) -> dict:
    """Load, check_env, random rollout and preview render in a child process."""
    cmd = [sys.executable, "-I", "-m", "trainlab.generator.check_scenario", str(folder)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "errors": [f"scenario check timed out after {timeout:.0f}s (infinite loop or very slow physics?)"]}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        tail = (proc.stderr or proc.stdout)[-2500:]
        return {"ok": False, "errors": [f"checker crashed:\n{tail}"]}
    return json.loads(lines[-1])
