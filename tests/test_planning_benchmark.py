# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The planning benchmark (tools/planning_benchmark.py) runs a demo problem and records what comparisons rely on."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("mujoco")
pytest.importorskip("ssik")
pytest.importorskip("sscbirrt_assets")


def _tool():
    spec = importlib.util.spec_from_file_location("planning_benchmark", ROOT / "tools" / "planning_benchmark.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules["planning_benchmark"] = tool
    spec.loader.exec_module(tool)
    return tool


def test_benchmark_records_time_work_and_path(tmp_path, capsys):
    tool = _tool()
    out = tmp_path / "bench.json"
    assert tool.main(["run", "--problems", "pick", "--seeds", "1", "--quiet", "--output", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["versions"]["sscbirrt"] and data["machine"]["cpus"]
    pick = data["problems"]["pick"]
    assert len(pick["calibration_seconds"]) == 2
    (run,) = pick["runs"]
    assert run["success"] and run["backend"] == "native"
    # solve time is root collection plus the search: planning_time alone excludes the initial roots
    assert run["solve_seconds"] == pytest.approx(run["roots_seconds"] + run["planning_seconds"])
    assert run["roots"] >= 2 and run["state_checks"] > 0 and run["path_length"] > 0
    assert set(pick["summary"]["work_median"]) == set(tool.WORK_KEYS)
    assert sum(pick["summary"]["goals_by_label"].values()) == 1

    assert tool.main(["compare", str(out), str(out)]) == 0
    assert "solve median s" in capsys.readouterr().out
