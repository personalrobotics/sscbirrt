# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""#254 and #256's benchmark runs: on its cluttered scene the filtered validator equals a full collision, the
first-class clearance equals the model margins it stands for, and the filter skips work."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("sscbirrt_assets")
from sscbirrt.backends import native_mujoco  # noqa: E402

if not native_mujoco.available():
    pytest.skip(native_mujoco.unavailable_reason(), allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent


def test_the_benchmark_agrees_and_filters(tmp_path):
    spec = importlib.util.spec_from_file_location("collision_benchmark", ROOT / "tools" / "collision_benchmark.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules["collision_benchmark"] = tool
    spec.loader.exec_module(tool)
    out = tmp_path / "bench.json"
    assert tool.main(["--configs", "200", "--output", str(out)]) == 0  # 0: every agreement holds
    report = json.loads(out.read_text())
    assert report["agreement"] == {
        "filtered_equals_all_0mm": True,
        "filtered_equals_all_5mm": True,
        "first_class_equals_model_margin_5mm": True,  # the clearance is the model margins it stands for
    }
    v = report["variants"]
    # With a clearance the resting cans come within it of the table and each other; the filtered check skips them.
    assert v["filtered_5mm"]["mean_contacts_generated"] < v["all_5mm"]["mean_contacts_generated"]
    assert (
        v["filtered_5mm"]["valid_fraction"] == v["all_5mm"]["valid_fraction"] == v["model_margin_5mm"]["valid_fraction"]
    )
