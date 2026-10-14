# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#230's evidence: a fresh run of tools/validation_artifact.py agrees with the stored artifact."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
from sscbirrt.backends import native_mujoco  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validation_artifact", ROOT / "tools" / "validation_artifact.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def test_a_fresh_run_has_the_stored_verdicts():
    stored = json.loads(tool.ARTIFACT.read_text())
    fresh = json.loads(json.dumps(tool.generate()))
    assert tool.semantic_view(fresh) == tool.semantic_view(stored)


def test_the_evidence_shows_what_230_asks():
    a = json.loads(tool.ARTIFACT.read_text())
    assert all(a["python_equals_native_verdicts"].values())
    for b in a["backends"].values():
        assert b["planning_equals_public"] == {"path": True, "trajectory": True}
        assert b["repeated_equal"] and b["reordered_equal"]
        public = b["public"]
        assert public["accepted_path"]["valid"] and public["reparameterization"]["valid"]
        w = public["rejected_path"]["witness"]
        assert (w["failure"], w["segment"]) == ("collision", 0) and w["contacts"]
        assert public["trace_changing"]["witness"]["time_s"] > 0


def test_the_geodude_edge_evidence_is_consistent():
    """ssrobot#132's edge, made inside ssrobot (tools/geodude_132_evidence.py): planning and the public call agree."""
    e = json.loads((ROOT / "tests" / "reference" / "geodude_132_evidence.json").read_text())
    rec = e["recorded_plan"]
    assert rec["result_equals_ci_artifact"] and rec["snapshot_fingerprint_matches"]
    assert rec["smoother_3_7_rerun_reproduces_trajectory"] and rec["python_equals_native"]
    assert rec["dense_recheck_3_7"]["invalid"] > 0 and rec["dense_recheck_3_7"]["pieces"] == [3]
    assert all(why is None for why in rec["rule_3_7_per_piece"])  # the 3.7 rule accepted every piece ...
    assert rec["this_validator"]["native"]["trajectory"]["valid"]  # ... and so does this one: the policy (#222)
    planned = e["planned_with_this_sscbirrt"]
    assert planned["planning_equals_public"] and planned["smoother_rerun_reproduces_trajectory"]
