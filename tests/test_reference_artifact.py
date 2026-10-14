# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The Python reference artifact pins the planner's semantics on a fixed matrix (#94).

Regenerates the artifact in-process and compares its semantic view with the
checked-in file: status, failure category, provenance, and the independent
validation report. Paths and iteration counts are inspectable but not
compared, so the test is stable across platforms while behavior is pinned.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "reference_artifact.py"
ARTIFACT = ROOT / "tests" / "reference" / "python_reference.json"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh(tool):
    return tool.generate()


@pytest.fixture(scope="module")
def stored():
    return json.loads(ARTIFACT.read_text())


def test_artifact_is_checked_in(stored):
    assert stored["cases"], "artifact has no cases"
    assert {"sscbirrt", "sstsr", "numpy", "python"} <= set(stored["versions"])


def test_semantics_match_the_stored_artifact(tool, fresh, stored):
    assert tool.semantic_mismatches(stored, fresh) == []
    assert {c["name"] for c in fresh["cases"]} == {c["name"] for c in stored["cases"]}


def test_every_success_passes_independent_validation(fresh):
    for case in fresh["cases"]:
        if case["status"] != "success":
            continue
        v = case["validation"]
        for key in (
            "all_in_space",
            "first_in_start_set",
            "last_in_goal_set",
            "all_admissible",
            "path_valid",
            "report_matches_planning",
            "raw_steps_within_step_size",
        ):
            assert v[key], f"{case['name']}: {key} failed"


def test_expected_statuses(fresh):
    by_name = {c["name"]: c for c in fresh["cases"]}
    assert by_name["timeout"]["failure_category"] == "timeout"
    assert by_name["cancellation"]["failure_category"] == "aborted"
    assert by_name["unreachable"]["failure_category"] == "max_iterations"
    for name in (
        "fixed_to_fixed",
        "multiple_roots",
        "nested_finite_goal",
        "wrapped_seam",
        "rejection_constraint",
        "projected_constraint",
        "tsr_goal_union",
        "tsr_chain_goal",
        "allof_constraint",
    ):
        assert by_name[name]["status"] == "success", name
    assert by_name["nested_finite_goal"]["goal_source"] == [0, 1]  # the filtered member of the first alternative


def test_regeneration_is_deterministic(tool):
    a, b = tool.generate(), tool.generate()
    a.pop("versions"), b.pop("versions")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_check_mode_passes(tool, capsys):
    assert tool.main(["--check"]) == 0
    assert "semantics match" in capsys.readouterr().out


def test_native_backend_matches_the_artifact(tool):
    pytest.importorskip("sscbirrt._native")
    native = tool.generate("native")
    supported = [c["name"] for c in native["cases"] if c["status"] != "unsupported"]
    assert {"fixed_to_fixed", "multiple_roots", "wrapped_seam", "timeout", "cancellation", "unreachable"} <= set(
        supported
    )
    by_name = {c["name"]: c for c in native["cases"]}
    # The planar TSR cases use a Python IK, so they fall back with that reason, the planar chain case included: chains
    # themselves are native since sstsr 3.3 (#184).
    assert by_name["tsr_chain_goal"]["unsupported_reasons"][0].startswith("goal: IK PlanarIK is a Python object")
    assert by_name["projected_constraint"]["unsupported_reasons"][0].startswith(
        "path_constraint: IK PlanarIK is a Python object"
    )
    if "ur5e_tsr_goal_union_with_path_tsr" in by_name and pytest.importorskip("sscbirrt._native").has_ssik():
        ur5e = by_name["ur5e_tsr_goal_union_with_path_tsr"]
        assert ur5e["status"] == "success" and ur5e["native_python_calls"] == 0  # the no-callback proof (#91)
    # the v1.7.0 release cases (#88) and the native TSR chain (#184)
    for name in ("ur5e_mujoco_tsr_goal_among_obstacles", "ur5e_mujoco_held_object", "ur5e_mujoco_tsr_chain_crank"):
        case = by_name[name]
        if case["status"] != "skipped":
            assert case["status"] == "success" and case["native_python_calls"] == 0, name
    assert tool.parity_mismatches(json.loads(ARTIFACT.read_text()), native) == []
    assert tool.main(["--check", "--backend", "native"]) == 0


def test_default_selection_matches_the_artifact_and_explains_every_python_choice(tool):
    """Under the default backend, native runs wherever it can and Python only with a stated reason (#86)."""
    pytest.importorskip("sscbirrt._native")
    auto = tool.generate("auto")
    ran = {c["name"]: c for c in auto["cases"] if c["status"] != "skipped"}
    assert ran["fixed_to_fixed"]["backend"] == "native" and ran["fixed_to_fixed"]["backend_reasons"] == []
    assert ran["tsr_chain_goal"]["backend"] == "python"  # its IK is Python's; the chain itself would lower (#184)
    assert ran["tsr_chain_goal"]["backend_reasons"][0].startswith("goal: IK PlanarIK is a Python object")
    if "ur5e_mujoco_tsr_chain_crank" in ran:
        assert ran["ur5e_mujoco_tsr_chain_crank"]["backend"] == "native"
    assert all(c["backend_reasons"] for c in ran.values() if c["backend"] == "python")
    assert tool.main(["--check", "--backend", "auto"]) == 0
