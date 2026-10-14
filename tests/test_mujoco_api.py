# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""sscbirrt.mujoco: Arm and plan, the user-facing one-call MuJoCo API (#175)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from sscbirrt.backends import native_mujoco as nm  # noqa: E402

if not nm.available():
    pytest.skip(nm.unavailable_reason(), allow_module_level=True)

from tsr import TSR  # noqa: E402

from sscbirrt import FiniteSet, PlanResult  # noqa: E402
from sscbirrt.mujoco import Arm, plan  # noqa: E402

XML = """
<mujoco><compiler angle="radian"/><worldbody>
  <geom name="floor" type="plane" size="3 3 0.1"/>
  <geom name="wall" type="box" pos="0.47 0 0.3" size="0.05 0.5 0.3"/>
  <body name="base" pos="0 0 0.1"><joint name="j0" type="hinge" axis="0 0 1" range="-3 3"/>
    <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.1"/>
    <body name="link1" pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 1 0" range="-2 2"/>
      <geom type="capsule" size="0.03" fromto="0 0 0 0.4 0 0"/>
      <body name="hand" pos="0.4 0 0"><geom type="box" size="0.02 0.03 0.02"/><site name="tip" pos="0.02 0 0"/></body>
    </body>
  </body>
  <body name="block" pos="1 1 0.05"><freejoint/><geom type="box" size="0.03 0.03 0.03"/></body>
</worldbody></mujoco>"""


@pytest.fixture
def world():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[:2] = [0.0, -1.2]
    mujoco.mj_forward(model, data)
    return model, data, Arm(model, ["j0", "j1"], "tip")


class TestPlan:
    def test_start_defaults_to_the_arm_now_and_result_is_a_plan_result(self, world):
        model, data, arm = world
        r = plan(model, data, arm, goal=[2.5, -1.2], seed=0)
        assert isinstance(r, PlanResult) and r.success and r.backend == "native"
        assert np.allclose(r.path[0], [0.0, -1.2]) and np.allclose(r.path[-1], [2.5, -1.2])

    def test_goal_forms(self, world):
        model, data, arm = world
        two = plan(model, data, arm, goal=[[2.5, -1.2], [-2.5, -1.2]], seed=0)
        assert two.success and two.goal_index in (0, 1)
        as_set = plan(model, data, arm, goal=FiniteSet([np.array([2.5, -1.2])]), seed=0)
        assert as_set.success

    def test_backend_is_passed_through(self, world):
        model, data, arm = world
        assert plan(model, data, arm, goal=[2.5, -1.2], backend="python", seed=0).backend == "python"

    @pytest.mark.parametrize("backend", ["native", "python"])
    def test_trajectory_is_timed_named_and_leaves_the_path(self, world, backend):
        from sscbirrt import TrajectoryOptions

        model, data, arm = world
        plain = plan(model, data, arm, goal=[2.5, -1.2], seed=0, backend=backend)
        r = plan(model, data, arm, goal=[2.5, -1.2], seed=0, backend=backend,
                 trajectory=TrajectoryOptions(1.57, 4.0))  # fmt: skip
        assert plain.trajectory is None and all(np.array_equal(a, b) for a, b in zip(plain.path, r.path))
        traj = r.trajectory
        assert traj.joint_names == ("j0", "j1") and traj.backend == backend and traj.seed == 0
        assert np.allclose(traj.start[0], r.path[0]) and np.allclose(traj.goal[0], r.path[-1])
        assert all(p.validated in ("planned", "shortcut") for p in traj.pieces)

    @pytest.mark.parametrize("backend", ["native", "python"])
    def test_the_validator_is_the_plans(self, world, backend):
        """#230: sscbirrt.mujoco.validator with plan's arguments answers exactly as the plan's own reports."""
        from sscbirrt import TrajectoryOptions
        from sscbirrt.mujoco import validator

        model, data, arm = world
        r = plan(model, data, arm, goal=[2.5, -1.2], seed=0, backend=backend, holding="block",
                 trajectory=TrajectoryOptions(1.57, 4.0))  # fmt: skip
        v = validator(model, data, arm, holding="block", backend=backend)
        assert r.validation.valid and r.trajectory.validation.valid
        assert v.validate_path(r.path) == r.validation
        assert v.validate_trajectory(r.trajectory) == r.trajectory.validation
        assert not v.validate_state([0.0, 0.0]).valid  # the hand in the wall
        assert v.validate_state([0.0, 0.0]).witness.contacts

    def test_holding_takes_the_grasp_from_the_current_poses(self, world):
        model, data, arm = world
        assert plan(model, data, arm, goal=[2.5, -1.2], holding="block", seed=0).success
        with pytest.raises(ValueError, match="holding: body 'nope' not found"):
            plan(model, data, arm, goal=[2.5, -1.2], holding="nope")


class TestErrors:
    def test_pose_regions_need_ik(self, world):
        model, data, arm = world
        with pytest.raises(ValueError, match="need IK: build the Arm with mjcf="):
            plan(model, data, arm, goal=TSR())

    def test_arm_arguments_are_checked(self, world):
        model, data, _ = world
        with pytest.raises(ValueError, match="site 'nope' not found in the model; its sites are: tip"):
            Arm(model, ["j0", "j1"], "nope")
        with pytest.raises(ValueError, match="joint 'jX' not found"):
            Arm(model, ["j0", "jX"], "tip")
        with pytest.raises(TypeError, match="list of joint names"):
            Arm(model, "j0", "tip")

    def test_malformed_goal_and_constraint(self, world):
        model, data, arm = world
        with pytest.raises(ValueError, match=r"goal: expected one configuration of length 2"):
            plan(model, data, arm, goal=[1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="constraint: expected a TSR"):
            plan(model, data, arm, goal=[2.5, -1.2], constraint=42)

    def test_arm_from_another_model(self, world):
        model, data, arm = world
        other = mujoco.MjModel.from_xml_string(XML)
        with pytest.raises(ValueError, match="different model"):
            plan(other, mujoco.MjData(other), arm, goal=[2.5, -1.2])


def _ur5e_xml():
    """The Menagerie UR5e, from sscbirrt-assets or MUJOCO_MENAGERIE_PATH; skip if neither."""
    import os
    from pathlib import Path

    if os.environ.get("MUJOCO_MENAGERIE_PATH"):
        return Path(os.environ["MUJOCO_MENAGERIE_PATH"]) / "universal_robots_ur5e" / "ur5e.xml"
    assets = pytest.importorskip("sscbirrt_assets")
    return assets.ur5e_xml()


class TestSSIKArm:
    JOINTS = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    HOME = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

    def _moved_world(self, xml):
        """The UR5e attached with a prefix at an offset, rotated base: the frames must be derived, not assumed."""
        world = mujoco.MjSpec()
        frame = world.worldbody.add_frame()
        frame.pos = [0.4, -0.3, 0.2]
        frame.quat = [np.cos(0.4), 0.0, 0.0, np.sin(0.4)]
        frame.attach_body(mujoco.MjSpec.from_file(str(xml)).worldbody.first_body(), "ur_", "")
        model = world.compile()
        data = mujoco.MjData(model)
        data.qpos[:6] = self.HOME
        mujoco.mj_forward(model, data)
        return model, data

    @pytest.mark.parametrize("backend", ["native", "python"])
    def test_tsr_goal_on_an_attached_arm(self, backend):
        pytest.importorskip("ssik")
        xml = _ur5e_xml()
        model, data = self._moved_world(xml)
        arm = Arm(model, ["ur_" + j for j in self.JOINTS], "ur_attachment_site", mjcf=xml, ik_end_body="wrist_3_link")
        T = np.eye(4)
        T[:3, :3] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        T[:3, 3] = [0.8, -0.1, 0.5]
        bounds = np.array([[-0.02, 0.02], [-0.02, 0.02], [0, 0.05], [-0.02, 0.02], [-0.02, 0.02], [-np.pi, np.pi]])
        region = TSR(T0_w=T, Bw=bounds)
        r = plan(model, data, arm, goal=region, backend=backend, seed=0)
        assert r.success and r.backend == backend
        assert region.distance(arm.robot_model(data).forward_kinematics(r.path[-1]))[0] < 1e-3

    def test_a_mismatched_mjcf_is_refused(self):
        pytest.importorskip("ssik")
        xml = _ur5e_xml()
        model, _ = self._moved_world(xml)
        joints = ["ur_" + j for j in self.JOINTS]
        with pytest.raises(ValueError, match="pass ik_end_body="):
            Arm(model, joints, "ur_attachment_site", mjcf=xml)
        with pytest.raises(ValueError, match="does not match this arm"):
            Arm(model, joints[::-1], "ur_attachment_site", mjcf=xml, ik_end_body="wrist_3_link")
