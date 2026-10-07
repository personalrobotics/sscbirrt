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


def test_sweep_resumes_and_analyze_applies_the_decision_rule(tmp_path):
    """#186: a sweep runs every (setting, seed) pair once, resumes without repeating, and analyze picks a setting."""
    tool = _tool()
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "fixed": {"timeout": 10, "sample_draws": 1000},
                "factors": {"num_tree_roots": [5], "sample_probability": [0.1, 0.25]},
                "baseline": {"num_tree_roots": 20, "sample_probability": 0.1},
            }
        )
    )
    out = tmp_path / "sweep.jsonl"
    argv = ["sweep", str(spec), "--problems", "pick", "--seeds", "2", "--output", str(out), "--quiet"]
    assert tool.main(argv) == 0
    lines = out.read_text().splitlines()
    runs = [json.loads(line) for line in lines if '"cell"' in line]
    assert len(runs) == 3 * 2 and {r["timeout"] for r in runs} == {10.0}
    out.write_text("\n".join(lines[:4]) + "\n")  # interrupted after two runs
    assert tool.main(argv) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    pairs = [(json.dumps(r["cell"], sort_keys=True), r["seed"]) for r in rows if "cell" in r]
    assert len(pairs) == 6 and len(set(pairs)) == 6  # resumed, nothing repeated
    result = tool.analyze(rows[1:], rows[0]["header"]["spec"], "pick")
    assert result["baseline"] is not None and result["winner"] is not None
    assert set(result["effects"]) == {"num_tree_roots", "sample_probability"}


def test_run_applies_and_records_overrides(tmp_path):
    tool = _tool()
    out = tmp_path / "set.json"
    argv = ["run", "--problems", "pick", "--seeds", "1", "--quiet", "--output", str(out), "--set", "num_tree_roots=5"]
    assert tool.main(argv) == 0
    data = json.loads(out.read_text())
    assert data["overrides"] == {"num_tree_roots": 5}
    run = data["problems"]["pick"]["runs"][0]
    assert run["roots"] - run["search_roots"] == 1 + 5  # HOME, and the 5 goal roots collected before the search
