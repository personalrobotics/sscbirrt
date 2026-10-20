# Migrating mj_manipulator onto the native planning boundary

Written for the mj_manipulator maintainers, ahead of personalrobotics/mj_manipulator#174 and #175.

Since 1.7, sscbirrt offers one boundary for planning against a live MuJoCo world: an
owned scene, an immutable snapshot, a collision checker with mj_manipulator's
contact policy implemented in C++, and one call that runs the whole solve
natively. This guide maps what `Arm.create_planner` builds today onto that
boundary, says what sscbirrt does not take over, and states the rules a
caller must keep.

## The boundary

| Today, in `Arm.create_planner` | On the boundary |
|---|---|
| `self.env.fork()` for an isolated `MjData` | `NativeScene.from_model(model, joint_names, extra_arm_bodies)`: an `mjModel` sscbirrt owns, built once from the compiled model's MJB bytes and cached per model object and by their hash; nothing borrows the live model. |
| `ContextRobotModel(model, data, ...)` | `MuJoCoRobotModel(model, data, ee_site, joint_names)`, or nothing: `sscbirrt.mujoco.Arm(model, joint_names, ee_site, mjcf=...)` builds it, with the SSIK solver. Its forward kinematics must agree with the SSIK model's; lowering checks that at the start configurations and refuses with a reason if they differ by more than 1e-6. |
| `CollisionChecker(model, data, joint_names, grasped_objects=..., attachments=..., extra_arm_body_names=...)` in snapshot mode | `Snapshot.capture(scene, data, attachments={object: (gripper_body, T_gripper_object)})` on the owner thread, then `NativeCollisionChecker(scene, snapshot)`. The gripper-base rule (`<prefix>/base` when it exists, else the attachment body) is resolved to body ids at capture. |
| `continuous_joints(model, joint_names)` | Unchanged: `CBiRRTConfig(continuous_joints=...)`. Unlimited joints are reported as ±∞ by both robot models and the space demands the declaration. |
| `CBiRRT(robot, ik, collision, config).plan(...)` | `sscbirrt.mujoco.plan(model, data, arm, start=..., goal=..., constraint=..., holding=..., config=..., seed=...)`, or the same `CBiRRT(..., backend="native")` with a `NativeCollisionChecker` as the validator. `holding` takes body names (the grasp from the current poses) or the `{object: (gripper_body, T_gripper_object)}` dict. `plan_native(model, data, joint_names, ...)`, the 1.7 form of this call, is deprecated since 3.1.0. |

The IK is SSIK built from the same MJCF as the MuJoCo model.
`Arm(model, joint_names, ee_site, mjcf=ur5e_xml)` builds it and computes both
frame offsets from the model (so an arm attached anywhere, with a name prefix,
works), then checks that the two forward kinematics agree. By hand:
`SSIKSolver(ssik.Manipulator.from_mjcf(ur5e_xml, base="world",
ee="wrist_3_link"), T_ee=site_offset_in_body(model, "attachment_site"))`.

## What the result carries

`PlanResult.backend` is `"native"` or `"python"`; `backend_reasons` says why
Python was chosen under `backend="auto"`, the default since sscbirrt 2.0 (it
was `"python"` in 1.x; pass it explicitly to keep the reference). `PlanResult.provenance` records the
dependency versions, the scene's model signature and MJB hash, the snapshot
hash, and the SSIK family. `PlanResult.stats` breaks the cost down by
component. Zero Python functions run during a native solve; the reference
artifact records that count for the UR5e cases.

## What stays in mj_manipulator

Retiming, execution, visualization, grasp verification, recovery, and the
decision to replan. sscbirrt plans a geometric path against a snapshot and
returns it; it never touches the live `MjData` after capture (mj_manipulator#174's
first criterion), and it does not own the simulator or its threads.

## Rules for the caller

1. **Capture on the owner thread.** `Snapshot.capture` reads `qpos`, mocap
   poses, and computes attachment transforms from `data`; it must see a
   consistent world. After capture the solve is independent of `data`.
2. **Treat the model as immutable after first use.** `NativeScene.from_model`
   captures the model the first time it sees that object and returns the
   cached scene for it afterward without inspecting it, so an in-place edit,
   structural or numeric, is not seen. After an intentional edit, pass
   `refresh=True` or a new model object; either exports and hashes the MJB
   again. A new model object with the same content shares the scene.
   `mjModel.signature` is not a safe key (it hashes structure only).
3. **Revalidate before executing.** The path was valid against the snapshot.
   Capture a fresh snapshot before execution and compare `sha256`; if it
   differs, run the path (or at least its executable prefix) through a
   `NativeCollisionChecker` on the fresh snapshot, and replan if it fails.
   The same policy that planned it validates it.
4. **One validator per solve.** Lowering creates a fresh validator with its
   own `mjData` for every solve; two solves in parallel never share one. Do
   not hand one `NativeCollisionChecker` to two threads' Python code either.
5. **Pin `mujoco==3.14.0`** for sscbirrt 1.7 through 3.2, the versions built against it; the
   native scene refuses another MuJoCo with a message naming the three
   versions it sees.
6. **Unsupported means Python.** An IK solver or validator without a native
   form (the README's "Adding an integration" says how to give it one) makes
   `backend="native"` raise `NativeUnsupported` listing every blocker, and `backend="auto"` (the
   default, also for `sscbirrt.mujoco.plan`) selects the Python planner and records
   the reasons. The semantics are the same; the parity gate checks that.

## Downstream milestone

mj_manipulator#175 introduces the backend boundary in `Arm`, #174 bridges
snapshots and live-world revalidation, and #173 is the adoption artifact:
zero callbacks, independent validation, agreement with today's Python path
on a corpus, repeatable seeds, and full provenance. The corpus that gates
sscbirrt's validator against mj_manipulator's checker already exists
(`tests/reference/mujoco_collision_corpus.json`).
