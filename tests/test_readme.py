# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The README's runnable snippets run as written (#178)."""

import re
from pathlib import Path

import pytest

README = (Path(__file__).resolve().parent.parent / "README.md").read_text()


def _first_python_block(heading: str) -> str:
    section = README.split(f"\n{heading}\n", 1)[1].split("\n## ", 1)[0]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


def test_quick_start_without_a_simulator_runs():
    exec(compile(_first_python_block("## Quick start without a simulator"), "README quick start", "exec"), {})


def test_plan_in_mujoco_runs(capsys):
    pytest.importorskip("mujoco")
    pytest.importorskip("ssik")
    pytest.importorskip("sscbirrt_assets")
    from sscbirrt.backends import native_mujoco

    if not native_mujoco.available():
        pytest.skip(native_mujoco.unavailable_reason())
    exec(compile(_first_python_block("## Plan in MuJoCo"), "README MuJoCo", "exec"), {})
    assert capsys.readouterr().out.startswith("True native")


def test_validating_a_path_runs():
    namespace: dict = {}
    exec(compile(_first_python_block("## Quick start without a simulator"), "README quick start", "exec"), namespace)
    exec(compile(_first_python_block("## Validating a path"), "README validation", "exec"), namespace)
    assert namespace["report"].valid and namespace["report"].subject == "path"
