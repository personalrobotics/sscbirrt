# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""CBiRRTConfig rejects malformed ranges at construction (#108)."""

import pytest

from sscbirrt import CBiRRTConfig


def test_defaults_construct():
    CBiRRTConfig()


@pytest.mark.parametrize(
    "field, value, requirement",
    [
        ("timeout", 0.0, "positive"),
        ("timeout", -1.0, "positive"),
        ("step_size", 0.0, "positive"),
        ("progress_tolerance", 0.0, "positive"),
        ("projection_progress_tolerance", 0.0, "positive"),
        ("membership_tolerance", -1e-9, "nonnegative"),
        ("connection_tolerance", -1e-9, "nonnegative"),
        ("max_iterations", 0, "at least 1"),
        ("sample_draws", 0, "at least 1"),
        ("num_tree_roots", 0, "at least 1"),
        ("max_per_draw", 0, "None or at least 1"),
        ("max_projection_iters", 0, "at least 1"),
        ("smoothing_iterations", -1, "nonnegative"),
        ("smoothing_patience", -1, "nonnegative"),
        ("edge_resolution", 0.0, "None or positive"),
        ("extend_steps", 0, "None or at least 1"),
        ("connect_steps", 0, "None or at least 1"),
        ("goal_sample_probability", 1.5, r"within \[0, 1\]"),
        ("start_sample_probability", -0.1, r"within \[0, 1\]"),
    ],
)
def test_out_of_range_field_is_rejected_by_name(field, value, requirement):
    with pytest.raises(ValueError, match=rf"^{field} must be {requirement}, got"):
        CBiRRTConfig(**{field: value})


@pytest.mark.parametrize(
    "field, value",
    [
        ("membership_tolerance", 0.0),
        ("connection_tolerance", 0.0),
        ("smoothing_iterations", 0),
        ("smoothing_patience", 0),
        ("goal_sample_probability", 0.0),
        ("start_sample_probability", 1.0),
        ("edge_resolution", None),
        ("extend_steps", 1),
        ("connect_steps", None),
    ],
)
def test_boundary_values_are_accepted(field, value):
    CBiRRTConfig(**{field: value})


def test_deprecated_alias_is_validated_too():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="membership_tolerance must be nonnegative"):
            CBiRRTConfig(tsr_tolerance=-1.0)


class TestRenamedNames:
    """#176: renamed configuration names keep working, with a warning, until 4.0."""

    @pytest.mark.parametrize(
        "old, new, value",
        [
            ("tsr_samples", "sample_draws", 7),
            ("max_ik_per_pose", "max_per_draw", 2),
            ("angular_joints", "continuous_joints", (True,)),
            ("start_bias", "start_sample_probability", 0.2),  # #196: renamed, and the meaning changed
            ("goal_bias", "goal_sample_probability", 0.3),
        ],
    )
    def test_old_keyword_and_attribute(self, old, new, value):
        with pytest.warns(DeprecationWarning, match=f"CBiRRTConfig.{old} is deprecated; use {new}"):
            cfg = CBiRRTConfig(**{old: value})
        assert getattr(cfg, new) == value
        with pytest.warns(DeprecationWarning):
            assert getattr(cfg, old) == value
        with pytest.raises(TypeError, match=f"both {old}"):
            CBiRRTConfig(**{old: value, new: value})

    def test_cbirrt_ik_solver_keyword(self):
        from sscbirrt import CBiRRT
        from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK

        with pytest.warns(DeprecationWarning, match=r"CBiRRT\(ik_solver=...\) is deprecated; use ik="):
            planner = CBiRRT(PlanarArm(), ik_solver=PlanarIK(), collision_checker=NoCollision())
        assert isinstance(planner.ik, PlanarIK)

    def test_configuration_planning_needs_no_ik(self):
        import numpy as np

        from sscbirrt import CBiRRT
        from sscbirrt.testing import NoCollision, PlanarArm

        planner = CBiRRT(PlanarArm(), collision_checker=NoCollision())
        assert planner.plan(start=np.zeros(2), goal=np.array([1.0, 0.5]), seed=0) is not None
        with pytest.raises(TypeError, match="needs a collision_checker"):
            CBiRRT(PlanarArm())

    def test_seeds_is_explicit_samples(self):
        import numpy as np

        from sscbirrt import FiniteSet, explicit_samples, seeds

        s = FiniteSet([np.zeros(2)])
        with pytest.warns(DeprecationWarning, match="use explicit_samples"):
            assert [m.source for m in seeds(s)] == [m.source for m in explicit_samples(s)]


def test_tsrs_without_ik_say_so():
    from tsr import TSR

    from sscbirrt import CBiRRT
    from sscbirrt.testing import NoCollision, PlanarArm

    planner = CBiRRT(PlanarArm(), collision_checker=NoCollision())
    with pytest.raises(ValueError, match="need IK: CBiRRT"):
        planner.plan(start=[0.0, 0.0], goal_tsrs=[TSR()])


def test_edge_resolution_defaults_to_005_independent_of_step_size():
    """#204: collision checks every 0.05 rad by default, whatever step_size is; None still means step_size."""
    assert CBiRRTConfig().edge_resolution == 0.05
    assert CBiRRTConfig(step_size=0.3).edge_resolution == 0.05
    native = pytest.importorskip("sscbirrt._native")
    assert native.PlannerConfig().edge_resolution == 0.05
