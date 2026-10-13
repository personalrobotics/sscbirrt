# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native backend: lowering, explicit fallback, no Python callbacks, and semantic parity (#118)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tsr import TSR

from sscbirrt import (
    AllStartConfigurationsInCollision,
    CBiRRT,
    CBiRRTConfig,
    FiniteSet,
    PlanningError,
    PlanningProblem,
    PredicateSet,
    TSRConfigurationSet,
)
from sscbirrt.backends import native
from sscbirrt.sets import explicit_samples
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK, Wall

pytest.importorskip("sscbirrt._native")

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def artifact_tool():
    spec = importlib.util.spec_from_file_location("reference_artifact", ROOT / "tools" / "reference_artifact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["reference_artifact"] = module
    spec.loader.exec_module(module)
    return module


def _planner(backend, collision=None, **kw):
    cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0, **kw)
    return CBiRRT(PlanarArm(), PlanarIK(), collision or NoCollision(), cfg, backend=backend)


def _finite(planner, configs):
    return FiniteSet(configs, tolerance=planner.config.membership_tolerance, metric=planner.space.distance)


class TestLowering:
    def test_finite_problem_lowers_and_solves_natively(self):
        planner = _planner("native")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert result.success and result.backend == "native" and result.backend_reasons == ()
        assert np.array_equal(result.path[0], np.zeros(2))
        assert planner.space.distance(result.path[-1], np.array([1.0, 0.5])) <= 1e-3
        assert result.start_source == (0,) and result.goal_source == (0,) and result.goal_index == 0
        assert result.tree_start is not None and len(result.tree_start) == result.tree_sizes[0]

    def test_wall_lowers_to_a_box(self):
        planner = _planner("native", collision=Wall(axis=0, lo=0.45, hi=0.55), max_iterations=300)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.0])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert not result.success and result.failure_reason.startswith("Max iterations")
        assert all(n.config[0] <= 0.45 for n in result.tree_start.nodes)

    def test_unsupported_components_are_all_reported(self):
        planner = _planner("native")
        tsr = TSR(
            T0_w=np.eye(4),
            Tw_e=np.eye(4),
            Bw=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]),
        )
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=TSRConfigurationSet(tsr, planner.robot, planner.ik, planner.space),
            validator=planner.collision,
            path_constraint=PredicateSet(lambda q: True, name="p"),
        )
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(problem, seed=0)
        reasons = info.value.reasons
        # The TSR set's IK is the planar Python solver, so that is the blocker named for the goal.
        assert any(r.startswith("goal: IK PlanarIK is a Python object") for r in reasons)
        assert any(r.startswith("path_constraint: PredicateSet") for r in reasons)
        assert len(reasons) == 2

    def test_auto_selects_python_and_records_why(self):
        planner = _planner("auto")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
            path_constraint=PredicateSet(lambda q: q[1] >= -1e-9, name="q1>=0"),
        )
        result = planner.solve(problem, seed=4)
        assert result.success and result.backend == "python"
        assert result.backend_reasons == ("path_constraint: PredicateSet has no native form",)

    def test_default_backend_is_auto_and_python_stays_explicit(self, caplog):
        cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0)
        planner = CBiRRT(PlanarArm(), PlanarIK(), NoCollision(), cfg)
        assert planner.backend == "auto"
        result = planner.plan(start=np.zeros(2), goal=np.array([1.0, 0.5]), seed=0, return_details=True)
        assert result.success and result.backend == "native" and result.backend_reasons == ()
        # An unsupported component under the default is Python with a diagnostic on the result and in the log.
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
            path_constraint=PredicateSet(lambda q: True, name="p"),
        )
        with caplog.at_level("INFO", logger="sscbirrt.planner"):
            result = planner.solve(problem, seed=0)
        assert result.backend == "python" and result.backend_reasons[0].startswith("path_constraint: PredicateSet")
        assert any("planning in Python" in r.message and "PredicateSet" in r.message for r in caplog.records)
        explicit = _planner("python")
        result = explicit.plan(start=np.zeros(2), goal=np.array([1.0, 0.5]), seed=0, return_details=True)
        assert result.success and result.backend == "python" and result.backend_reasons == ()
        with pytest.raises(ValueError, match="backend must be"):
            _planner("cuda")

    def test_no_roots_maps_to_the_python_exception(self):
        planner = _planner("native", collision=Wall(axis=0, lo=-1.0, hi=1.0))
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([2.0, 2.0])]),
            validator=planner.collision,
        )
        with pytest.raises(AllStartConfigurationsInCollision):
            planner.solve(problem, seed=0)

    @pytest.mark.parametrize(
        "starts",
        [
            [np.zeros(2)],  # one in collision
            [np.zeros(2), np.array([0.5, 0.0])],  # two in collision
            [np.zeros(2), np.array([9.0, 0.0])],  # one in collision, one outside the joint space
        ],
    )
    def test_no_roots_count_and_class_match_python(self, starts):
        """#170: the native count added explicit candidates to rejections, counting each explicit one twice."""
        raised = {}
        for backend in ("python", "native"):
            planner = _planner(backend, collision=Wall(axis=0, lo=-1.0, hi=1.0))
            problem = PlanningProblem(
                space=planner.space,
                start=_finite(planner, starts),
                goal=_finite(planner, [np.array([2.0, 2.0])]),
                validator=planner.collision,
            )
            with pytest.raises(PlanningError) as info:
                planner.solve(problem, seed=0)
            raised[backend] = (type(info.value), info.value.n_configs)
        assert raised["native"] == raised["python"]
        assert raised["python"][1] == len(starts)

    def test_abort_fn_is_honored(self):
        planner = _planner("native", abort_fn=lambda: True)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=0)
        assert not result.success and result.failure_reason.startswith("Aborted")


class TestNoCallbacks:
    def test_native_solve_makes_no_python_calls(self):
        planner = _planner("native")
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        lowered = native.lower(problem, planner.config)
        nplanner = native._native.Planner(lowered.config)
        calls = []

        def profiler(frame, event, arg):
            if event == "call":
                calls.append(frame.f_code.co_name)

        sys.setprofile(profiler)
        try:
            r = nplanner.solve(lowered.problem, 0, None, True)
        finally:
            sys.setprofile(None)
        assert r.success
        assert calls == []  # nothing in Python ran between entry and exit


class TestParityWithTheArtifact:
    """Native and Python agree on the semantic view for every artifact case the native core supports."""

    def test_semantic_parity(self, artifact_tool):
        lowered_any = False
        for case in artifact_tool.cases():
            if "skipped" in case:  # an environment-gated case (the Menagerie is absent here)
                continue
            cfg, problem, planner = case["config"], case["problem"], case["planner"]
            try:
                lowered = native.lower(problem, cfg)
            except native.NativeUnsupported:
                continue
            lowered_any = True
            py = planner.solve(problem, seed=case["seed"])
            nat = native.solve(lowered, case["seed"], cfg.abort_fn)
            assert nat.success == py.success, case["name"]
            assert artifact_tool.failure_category(nat.failure_reason) == artifact_tool.failure_category(
                py.failure_reason
            ), case["name"]
            if py.success:
                v = artifact_tool.validate(problem, cfg, nat.path)
                assert all(
                    v[k]
                    for k in (
                        "all_in_space",
                        "first_in_start_set",
                        "last_in_goal_set",
                        "all_admissible",
                        "edges_validated_at_resolution",
                        "raw_steps_within_step_size",
                    )
                ), (case["name"], v)
                # Provenance is compared only where the reached root is unique.
                if len(explicit_samples(problem.start)) == 1 and len(explicit_samples(problem.goal)) == 1:
                    assert (nat.start_source, nat.goal_source) == (py.start_source, py.goal_source), case["name"]
        assert lowered_any


class TestTSRLowering:
    """TSR sets lower through the native SSIK adapter, or say exactly why not (#130)."""

    @pytest.fixture(scope="class")
    def ur5e(self):
        ssik = pytest.importorskip("ssik")
        if not native._native.has_ssik():
            pytest.skip(native._native.ssik_unavailable_reason())
        from sscbirrt.backends.ssik import SSIKRobotModel, SSIKSolver

        ik = SSIKSolver(ssik.Manipulator.from_prebuilt("ur5e"))
        robot = SSIKRobotModel(ik)
        cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0, num_tree_roots=20)
        return robot, ik, cfg

    def _grasp(self, robot, ik, planner, q):
        box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        return TSRConfigurationSet(
            TSR(T0_w=robot.forward_kinematics(q), Tw_e=np.eye(4), Bw=box), robot, ik, planner.space
        )

    def test_tsr_goal_and_path_constraint_solve_natively_with_no_python_calls(self, ur5e):
        robot, ik, cfg = ur5e
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="native")
        q0 = np.array([0.0, -1.2, 1.0, -1.4, -1.57, 0.0])
        goal = self._grasp(robot, ik, planner, np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4]))
        above = TSRConfigurationSet(
            TSR(
                np.eye(4),
                np.eye(4),
                np.array([[-1.2, 1.2], [-1.2, 1.2], [0.05, 1.5], [-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]]),
            ),
            robot,
            ik,
            planner.space,
        )
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [q0]),
            goal=goal,
            validator=planner.collision,
            path_constraint=above,
        )
        lowered = native.lower(problem, cfg)
        nplanner = native._native.Planner(lowered.config)
        calls = []
        sys.setprofile(lambda frame, event, arg: calls.append(frame.f_code.co_name) if event == "call" else None)
        try:
            r = nplanner.solve(lowered.problem, 12, None, True)
        finally:
            sys.setprofile(None)
        assert r.success and calls == []
        result = native.convert(r)
        assert result.backend == "native"
        assert goal.contains(result.path[-1]) and all(above.contains(q) for q in result.path)
        assert np.array_equal(result.path[0], q0)

    def test_chain_lowers_natively_and_planar_ik_and_fk_disagreement_are_reported(self, ur5e):
        robot, ik, cfg = ur5e
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="native")
        q0 = np.array([0.0, -1.2, 1.0, -1.4, -1.57, 0.0])
        goal = self._grasp(robot, ik, planner, np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4]))
        base_problem = dict(space=planner.space, start=_finite(planner, [q0]), validator=planner.collision)

        from tsr import TSRChain

        # A chain lowers natively since sstsr 3.3 (#184): the grasp, then a link free to spin about the flange's axis.
        spin = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.array([[0, 0]] * 5 + [[-np.pi, np.pi]]))
        chain = TSRConfigurationSet(TSRChain(TSRs=[goal.tsr, spin]), robot, ik, planner.space)
        result = planner.solve(PlanningProblem(goal=chain, **base_problem), seed=0)
        assert result.success and result.backend == "native"
        assert chain.contains(result.path[-1])  # judged by the Python chain

        planar_ik_goal = TSRConfigurationSet(goal.tsr, robot, PlanarIK(), planner.space)
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(PlanningProblem(goal=planar_ik_goal, **base_problem), seed=0)
        assert info.value.reasons[0].startswith("goal: IK PlanarIK is a Python object")

        class Shifted:  # a robot model whose FK disagrees with SSIK's by 1 cm
            dof = robot.dof
            joint_limits = robot.joint_limits

            def forward_kinematics(self, q):
                T = robot.forward_kinematics(q).copy()
                T[0, 3] += 0.01
                return T

        shifted_goal = TSRConfigurationSet(goal.tsr, Shifted(), ik, planner.space)
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(PlanningProblem(goal=shifted_goal, **base_problem), seed=0)
        assert "robot model FK disagrees with the native IK model's" in info.value.reasons[0]

    def _native_set(self, s, planner):
        problem = PlanningProblem(space=planner.space, start=s, goal=s, validator=planner.collision)
        reasons: list[str] = []
        ctx = native._Lowering(problem, native._lower_space(planner.space))
        lowered = native._lower_tsr_set(s, "goal", ctx, reasons)
        assert reasons == []
        return lowered

    def test_native_chain_sets_keep_the_python_contract(self, ur5e):
        """A single-link chain is an exact path, so it matches Python; a cold chain is held to properties (#184)."""
        from tsr import TSRChain

        robot, ik, cfg = ur5e
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="native")
        grasp = self._grasp(robot, ik, planner, np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4])).tsr
        rng = np.random.default_rng(0)
        lo, hi = map(np.asarray, robot.joint_limits)
        qs = [rng.uniform(lo, hi) for _ in range(20)]

        single = TSRConfigurationSet(TSRChain(TSRs=[grasp]), robot, ik, planner.space)
        nsingle = self._native_set(single, planner)
        for q in qs:
            assert nsingle.distance(list(q)) == pytest.approx(single.distance(q), abs=1e-9)

        spin = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.array([[0, 0]] * 5 + [[-np.pi, np.pi]]))
        chain = TSRConfigurationSet(TSRChain(TSRs=[grasp, spin]), robot, ik, planner.space)
        nchain = self._native_set(chain, planner)
        samples = [q for seed in range(5) for q in nchain.sample(seed)]
        assert samples and all(chain.contains(np.array(q)) for q in samples)  # native samples are in Python's set
        for q in samples[:5] + qs:
            assert nchain.distance(list(q)) >= 0.0
            p = nchain.project(list(q))
            # projection lands in the set or reports failure, judged by the Python chain
            assert p is None or chain.contains(np.array(p))

    @pytest.mark.parametrize("name", ["panda", "iiwa14", "jaco2", "gen3", "xarm7"])
    def test_every_native_ssik_family_plans_tsr_goals_natively(self, name):
        """Since #220 every family ssik solves natively lowers: 6- and 7-joint arms alike."""
        ssik = pytest.importorskip("ssik")
        from sscbirrt.backends.ssik import SSIKRobotModel, SSIKSolver

        ik = SSIKSolver(ssik.Manipulator.from_prebuilt(name))
        robot = SSIKRobotModel(ik)
        unlimited = tuple(lim is None for lim in ik.solver.joint_limits)  # Kinova arms have continuous joints
        cfg = CBiRRTConfig(
            step_size=0.2, edge_resolution=0.05, timeout=30.0, num_tree_roots=20, continuous_joints=unlimited
        )
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="native")
        lo, hi = robot.joint_limits
        lo, hi = np.maximum(lo, -np.pi) + 0.1, np.minimum(hi, np.pi) - 0.1
        rng = np.random.default_rng(0)
        q0, qg = rng.uniform(lo, hi), rng.uniform(lo, hi)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [q0]),
            goal=self._grasp(robot, ik, planner, qg),
            validator=planner.collision,
        )
        result = planner.solve(problem, seed=1)
        assert result.success and result.backend == "native"
        assert problem.goal.contains(result.path[-1])

    def test_a_family_ssik_cannot_run_natively_says_so(self, ur5e, monkeypatch):
        ssik_cpp = pytest.importorskip("ssik.cpp")
        from sscbirrt.backends.native_ssik import unsupported_reason

        monkeypatch.setattr(ssik_cpp, "NATIVE_FAMILIES", ())
        robot, ik, cfg = ur5e
        assert "has no native solver" in unsupported_reason(ik.solver)

    def test_auto_backend_runs_tsr_problems_natively(self, ur5e):
        robot, ik, cfg = ur5e
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="auto")
        result = planner.plan(
            start=np.array([0.0, -1.2, 1.0, -1.4, -1.57, 0.0]),
            goal_tsrs=[self._grasp(robot, ik, planner, np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4])).tsr],
            seed=3,
            return_details=True,
        )
        assert result.success and result.backend == "native" and result.backend_reasons == ()


class TestIntegrations:
    """Validators and IK solvers lower through protocols, not class names (#147).

    Neither integration here is MuJoCo or SSIK; the lowering must accept them on the protocol alone and
    carry their provenance into the result.
    """

    def test_validator_integration_lowers_and_its_provenance_reaches_the_result(self):
        class BoxWall:  # a CollisionChecker with a native form, defined outside sscbirrt
            def __init__(self):
                self.fresh_calls = 0

            def is_valid(self, q):
                return not (0.45 <= q[0] <= 0.55)

            def fresh(self):
                self.fresh_calls += 1
                lo, hi = [0.45, -np.inf], [0.55, np.inf]
                return native._native.JointBoxObstacles([(lo, hi)])

            @property
            def provenance(self):
                return {"wall": "x in [0.45, 0.55]"}

        wall = BoxWall()
        assert isinstance(wall, native.ValidatorIntegration)
        planner = _planner("native", collision=wall, max_iterations=300)
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.0])]),
            validator=wall,
        )
        result = planner.solve(problem, seed=0)
        assert result.backend == "native" and wall.fresh_calls == 1
        assert not result.success and all(n.config[0] <= 0.45 for n in result.tree_start.nodes)
        assert result.provenance["wall"] == "x in [0.45, 0.55]"

    def test_validator_integration_that_hands_back_a_python_object_is_a_reason(self):
        class Bad:
            def is_valid(self, q):
                return True

            def fresh(self):
                return object()

            provenance = {}

        planner = _planner("native", collision=Bad())
        problem = PlanningProblem(
            space=planner.space,
            start=_finite(planner, [np.zeros(2)]),
            goal=_finite(planner, [np.array([1.0, 0.5])]),
            validator=planner.collision,
        )
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(problem, seed=0)
        assert info.value.reasons == ["validator: Bad.fresh() returned object, not a sscbirrt StateValidator"]

    def test_kinematics_integration_failures_are_forwarded_as_reasons(self):
        planner = _planner("native")
        tsr = TSR(T0_w=np.eye(4), Tw_e=np.eye(4), Bw=np.zeros((6, 2)))
        base = dict(space=planner.space, start=_finite(planner, [np.zeros(2)]), validator=planner.collision)

        class NoArm(PlanarIK):
            def native_kinematics(self):
                raise native.NativeUnsupported(["this arm has no native form today"])

            provenance = {}

        class WrongType(PlanarIK):
            def native_kinematics(self):
                return native._native.AcceptAll()

            provenance = {}

        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(PlanningProblem(goal=TSRConfigurationSet(tsr, planner.robot, NoArm(), planner.space), **base))
        assert info.value.reasons == ["goal: this arm has no native form today"]
        with pytest.raises(native.NativeUnsupported) as info:
            planner.solve(
                PlanningProblem(goal=TSRConfigurationSet(tsr, planner.robot, WrongType(), planner.space), **base)
            )
        assert info.value.reasons == [
            "goal: WrongType.native_kinematics() returned AcceptAll, not a sscbirrt ForwardKinematics and IKSolver"
        ]

    def test_kinematics_integration_lowers_without_the_ssik_adapter_class(self):
        ssik = pytest.importorskip("ssik")
        if not native._native.has_ssik():
            pytest.skip(native._native.ssik_unavailable_reason())
        from sscbirrt.backends import native_ssik
        from sscbirrt.backends.ssik import SSIKRobotModel, SSIKSolver

        manipulator = ssik.Manipulator.from_prebuilt("ur5e")
        reference = SSIKSolver(manipulator)

        class MyIK:  # an IKSolver from another package: not an SSIKSolver, but with a native form
            def solve(self, pose, q_init=None):
                return reference.solve(pose, q_init)

            def native_kinematics(self):
                return native_ssik.arm_from_manipulator(manipulator, T_base=None, T_ee=None)

            provenance = {"ik_backend": "MyIK"}

        robot = SSIKRobotModel(reference)
        ik = MyIK()
        cfg = CBiRRTConfig(step_size=0.2, edge_resolution=0.05, timeout=30.0, num_tree_roots=20)
        planner = CBiRRT(robot, ik, NoCollision(), cfg, backend="native")
        q0 = np.array([0.0, -1.2, 1.0, -1.4, -1.57, 0.0])
        q1 = np.array([0.8, -1.0, 0.8, -1.3, -1.57, 0.4])
        box = np.array([[-0.05, 0.05], [-0.05, 0.05], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]])
        goal = TSRConfigurationSet(TSR(robot.forward_kinematics(q1), np.eye(4), box), robot, ik, planner.space)
        result = planner.solve(
            PlanningProblem(space=planner.space, start=_finite(planner, [q0]), goal=goal, validator=planner.collision),
            seed=0,
        )
        assert result.success and result.backend == "native" and goal.contains(result.path[-1])
        assert result.provenance["ik_backend"] == "MyIK" and "ssik_solver_name" not in result.provenance


def test_benchmark_tool_runs_and_records_the_breakdown(tmp_path):
    spec = importlib.util.spec_from_file_location("benchmark_native", ROOT / "tools" / "benchmark_native.py")
    tool = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_native"] = tool
    spec.loader.exec_module(tool)
    out = tmp_path / "bench.json"
    assert tool.main(["--seeds", "1", "--quiet", "--output", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["cases"] and all("seconds_edge_checks" in row["native"]["stats"] for row in data["cases"])
    assert all(row["native"]["stats"]["state_checks"] > 0 for row in data["cases"])
