# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""The native SSIK adapter agrees with Python's SSIKSolver, on one arm of every native ssik family (#90, #129, #220).

ssik verifies its own native solvers against its Python, so these check sscbirrt's adapter: the SolverData hand-off,
the frame offsets, 6 and 7 joints, and seeds. Solution sets must be identical up to ordering, FK must agree, and every
solution must close; with the Menagerie available, the MuJoCo model is the independent FK oracle.
"""

import os
from pathlib import Path

import numpy as np
import pytest

ssik = pytest.importorskip("ssik")
_native = pytest.importorskip("sscbirrt._native")
if not _native.has_ssik():
    pytest.skip(_native.ssik_unavailable_reason(), allow_module_level=True)

from sscbirrt.backends.native_ssik import arm_from_manipulator  # noqa: E402
from sscbirrt.backends.ssik import SSIKSolver  # noqa: E402


def _sorted(sols):
    return sorted((tuple(np.round(q, 9)) for q in sols))


def _atol(arm):
    """How closely the backends' solutions agree. xarm7's polished family refines numerically, and the two builds of
    ssik's C++ converge to points up to 2.4e-6 apart on CI's runners (#282), not to the last bit; its distinct solutions
    are 1e-3 apart or more."""
    return 1e-5 if arm.solver_name == "seven_r.spherical_shoulder_polished" else 1e-6


def _on_a_limit(q, arm, margin=1e-6):
    """Whether ``q`` lies within ``margin`` of a joint limit. A solution on a limit, such as a joint at 0 wound to the
    UR5e's +-2pi, is kept or dropped by round-off, and the two builds of ssik's C++ may break that tie either way
    (#282)."""
    lo = np.array([lim[0] if lim else -np.inf for lim in arm.joint_limits])
    hi = np.array([lim[1] if lim else np.inf for lim in arm.joint_limits])
    return bool(np.any(q - lo < margin) or np.any(hi - q < margin))


def _difference(a, b, atol=1e-6, arm=None):
    """How solution sets ``a`` and ``b`` differ, or None for the same set: equal counts, and every solution of each
    matched by one of the other within ``atol``. Given ``arm``, a solution only one side has is excused when it lies on
    a joint limit, and the counts are compared without it."""
    a, b = [np.asarray(q) for q in a], [np.asarray(q) for q in b]

    def unmatched(xs, ys):
        gaps = [min((float(np.max(np.abs(x - y))) for y in ys), default=np.inf) for x in xs]
        missing = [(x, g) for x, g in zip(xs, gaps, strict=True) if g > atol]
        on_limit = [arm is not None and _on_a_limit(x, arm) for x, _ in missing]
        unexcused = [(x.round(9).tolist(), f"{g:.1e}") for (x, g), e in zip(missing, on_limit, strict=True) if not e]
        return unexcused, sum(on_limit)

    (only_a, excused_a), (only_b, excused_b) = unmatched(a, b), unmatched(b, a)
    if len(a) - excused_a == len(b) - excused_b and not only_a and not only_b:
        return None
    counts = f"{len(a)} against {len(b)} solutions, {excused_a} and {excused_b} unmatched on a limit"
    return f"{counts}; unmatched {len(only_a)}: {only_a[:3]}, and {len(only_b)}: {only_b[:3]}"


# One prebuilt arm per family ssik solves natively (ssik.cpp.NATIVE_FAMILIES, minus jointlock, which no prebuilt arm
# uses): the UR5e, ADA's JACO2, the iiwa, the Panda, and the polished variants.
FAMILIES = {
    "ur5e": "ikgeo.three_parallel",
    "jaco2": "ikgeo.general_6r",
    "iiwa14": "seven_r.srs",
    "gen3": "seven_r.srs_polished",
    "panda": "seven_r.spherical_shoulder",
    "xarm7": "seven_r.spherical_shoulder_polished",
}


def _random_configs(rng, arm, n):
    lo = np.array([lim[0] if lim else -np.pi for lim in arm.joint_limits])
    hi = np.array([lim[1] if lim else np.pi for lim in arm.joint_limits])
    return [rng.uniform(lo, hi) for _ in range(n)]


@pytest.fixture(scope="module", params=list(FAMILIES))
def pair(request):
    arm = ssik.Manipulator.from_prebuilt(request.param)
    return arm, SSIKSolver(arm), arm_from_manipulator(arm)


class TestEveryNativeFamily:
    def test_family_dof_and_fk_agree(self, pair):
        arm, py, nat = pair
        assert nat.family == arm.solver_name == FAMILIES[arm_name(arm)] and nat.dof == arm.dof
        rng = np.random.default_rng(0)
        for q in _random_configs(rng, arm, 50):
            assert np.allclose(nat.fk(list(q)), py.fk(q), atol=1e-12)

    def test_solution_sets_agree_and_close(self, pair):
        arm, py, nat = pair
        rng = np.random.default_rng(1)
        for q in _random_configs(rng, arm, 30):
            T = py.fk(q)
            a, b = py.solve(T), nat.solve(T.tolist())
            assert len(a) > 0, "reachable by construction"
            difference = _difference(a, b, _atol(arm))
            assert difference is None, f"solution sets differ at q={q}: {difference}"
            for sol in b:
                assert np.allclose(nat.fk(list(sol)), T, atol=1e-8)

    def test_seeded_solves_agree(self, pair):
        arm, py, nat = pair
        rng = np.random.default_rng(2)
        for q in _random_configs(rng, arm, 20):
            T = py.fk(q)
            seed = q + rng.normal(scale=0.3, size=arm.dof)
            difference = _difference(py.solve(T, q_init=seed), nat.solve(T.tolist(), list(seed)), _atol(arm))
            assert difference is None, f"seeded solution sets differ at q={q}, seed={seed}: {difference}"

    def test_a_wrong_length_seed_is_refused(self, pair):
        arm, py, nat = pair
        T = py.fk(_random_configs(np.random.default_rng(3), arm, 1)[0])
        with pytest.raises(ValueError, match="seed must have"):
            nat.solve(T.tolist(), [0.0] * (arm.dof + 1))


def arm_name(arm):
    return next(name for name, family in FAMILIES.items() if family == arm.solver_name)


class TestPrebuiltUR5e:
    @pytest.fixture(scope="class")
    def pair(self):
        arm = ssik.Manipulator.from_prebuilt("ur5e")
        return arm, SSIKSolver(arm), arm_from_manipulator(arm)

    def test_unreachable_and_singular_poses_agree(self, pair):
        arm, py, nat = pair
        far = np.eye(4)
        far[:3, 3] = [5.0, 0.0, 0.0]
        assert py.solve(far) == [] and nat.solve(far.tolist()) == []
        # Wrist singularity: joints 4 and 6 aligned (q5 = 0); shoulder singularity: wrist over the base. Both put
        # solutions on the +-2pi limits, where only one backend may keep them.
        for q in (
            [0.0, -1.2, 1.0, 0.2, 0.0, 0.7],
            [0.0, -np.pi / 2, 0.0, 0.0, 0.0, 0.0],
            [0.3, -0.8, 0.8, 0.0, 1e-7, 0.0],
        ):
            T = py.fk(np.array(q))
            a, b = py.solve(T), nat.solve(T.tolist())
            assert len(a) > 0 and len(b) > 0, "reachable by construction"
            difference = _difference(a, b, _atol(arm), arm)
            assert difference is None, f"singular pose disagreement at q={q}: {difference}"


@pytest.mark.skipif("MUJOCO_MENAGERIE_PATH" not in os.environ, reason="needs the MuJoCo Menagerie")
class TestMenagerieUR5e:
    def test_mjcf_arm_agrees_with_mujoco_fk_and_python(self):
        mujoco = pytest.importorskip("mujoco")
        from sscbirrt.backends.mujoco import MuJoCoRobotModel, site_offset_in_body

        xml = Path(os.environ["MUJOCO_MENAGERIE_PATH"]) / "universal_robots_ur5e" / "ur5e.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)
        joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        robot = MuJoCoRobotModel(model, data, "attachment_site", joints)
        arm = ssik.Manipulator.from_mjcf(str(xml), base="world", ee="wrist_3_link")
        T_ee = site_offset_in_body(model, "attachment_site")
        py = SSIKSolver(arm, T_ee=T_ee)
        nat = arm_from_manipulator(arm, T_ee=T_ee)
        rng = np.random.default_rng(3)
        lo, hi = robot.joint_limits
        for _ in range(30):
            q = rng.uniform(lo, hi)
            T_mj = robot.forward_kinematics(q)
            assert np.allclose(nat.fk(list(q)), T_mj, atol=1e-9)
            assert _sorted(py.solve(T_mj)) == _sorted(nat.solve(T_mj.tolist()))
