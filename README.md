# sscbirrt

[![PyPI](https://img.shields.io/pypi/v/sscbirrt.svg?v=1)](https://pypi.org/project/sscbirrt/)
[![Python](https://img.shields.io/pypi/pyversions/sscbirrt.svg?v=1)](https://pypi.org/project/sscbirrt/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A planner over sets of configurations. A planning problem names a start set,
a goal set, a set the whole path must stay inside, and a validity predicate;
a solution is a path that begins in the start set, ends in the goal set, and
stays inside the path set with every configuration valid. A set is anything
that answers `contains(q)`. The algorithm is CBiRRT (Berenson et al., 2009),
a bidirectional RRT that grows trees from many roots and projects onto
constraints. Task Space Regions are one representation of a set; the planner
does not depend on it, and sets you define yourself take the same roles.

<table>
<tr><td><img src="docs/images/pick_yellow_seed5.gif" alt="UR5e reaching the yellow can, from the near side" width="100%"></td><td><img src="docs/images/pick_green_seed22.gif" alt="UR5e reaching the green can, straight in" width="100%"></td><td><img src="docs/images/pick_blue_seed12.gif" alt="UR5e reaching the blue can, over the boxes" width="100%"></td></tr>
<tr><td align="center"><sub>yellow can, from the near side (seed 5)</sub></td><td align="center"><sub>green can, straight in (seed 22)</sub></td><td align="center"><sub>blue can, over the boxes (seed 12)</sub></td></tr>
<tr><td><img src="docs/images/pick_yellow_seed15.gif" alt="UR5e reaching the yellow can, over the boxes" width="100%"></td><td><img src="docs/images/pick_green_seed57.gif" alt="UR5e reaching the green can, another grasp" width="100%"></td><td><img src="docs/images/pick_yellow_seed54.gif" alt="UR5e reaching the yellow can, a wide arc" width="100%"></td></tr>
<tr><td align="center"><sub>yellow can, over the boxes (seed 15)</sub></td><td align="center"><sub>green can, another grasp (seed 57)</sub></td><td align="center"><sub>yellow can, a wide arc (seed 54)</sub></td></tr>
</table>

Six runs of one call, `plan(model, data, arm, goal=grasps)`, where `grasps` is
every side grasp of every can: 18 regions. Each seed lets the planner choose a
different can, grasp, and route around the red boxes, planned natively in
0.02 to 0.12 s. `sscbirrt-demo pick --seed N` renders any of them.

<table>
<tr><td><img src="docs/images/transport.gif" alt="UR5e carrying a can upright over a box" width="100%"></td><td><img src="docs/images/door.gif" alt="UR5e opening a door: a TSR chain" width="100%"></td></tr>
<tr><td align="center"><sub>carrying a can upright: a path constraint</sub></td><td align="center"><sub>opening a door: a TSR chain</sub></td></tr>
</table>

A path constraint holds at every point of the path: the can is lifted over
the box and set down without tipping. The door's constraint is a TSR chain,
the hinge and then the grasp on the handle, so the gripper follows the
handle's arc as the door swings open. `sscbirrt-demo transport door`
renders both.

## Install

```bash
pip install sscbirrt                                  # the planner, its C++ core, and Task Space Regions
pip install "sscbirrt[mujoco,ssik]" sscbirrt-assets   # plus MuJoCo, analytical IK, and the UR5e model below
pip install "sscbirrt[demo]" && sscbirrt-demo         # everything above, and the rendered demos (MP4s in ./sscbirrt-demo)
```

Wheels for Linux x86_64 and macOS arm64, Python 3.10 through 3.14, include the
native core with the SSIK and MuJoCo adapters. `numpy` and `sstsr` (Task Space
Regions, imported as `tsr`) are installed as dependencies; the `mujoco` extra
pins the exact MuJoCo the extension is built against. `sscbirrt-assets` packages
the MuJoCo Menagerie UR5e and Robotiq 2F-85 so the demos need no clone.

From a checkout or the sdist, the extension compiles on install and needs
CMake 3.16+ and a C++20 compiler; the build fetches scikit-build-core,
pybind11, ninja, ssik, sstsr, and mujoco itself, and Eigen if none is installed:

```bash
uv pip install -e ".[all]"          # every extra, the example dependencies, and the dev tools
```

## Plan in MuJoCo

Describe the arm once, then plan from where it is to a region:

```python
import mujoco
import numpy as np
import sscbirrt_assets
from tsr import TSR

from sscbirrt.mujoco import Arm, plan

xml = sscbirrt_assets.ur5e_xml()                       # the Menagerie UR5e, packaged
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
data.qpos[:] = [0, -1.57, 1.57, -1.57, -1.57, 0]       # where the arm is now

joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
arm = Arm(model, joints, ee_site="attachment_site", mjcf=xml)   # SSIK built from the same MJCF

# Goal: the flange pointing down, within 2 cm of (0.5, 0.2, 0.3) and up to 5 cm above it, any yaw
T = np.eye(4)
T[:3, :3] = np.diag([1.0, -1.0, -1.0])
T[:3, 3] = [0.5, 0.2, 0.3]
goal = TSR(T0_w=T, Bw=[[-0.02, 0.02], [-0.02, 0.02], [0, 0.05], [0, 0], [0, 0], [-np.pi, np.pi]])

result = plan(model, data, arm, goal=goal, seed=0)
print(result.success, result.backend, len(result.path), f"{result.planning_time:.3f} s")
```

`plan(model, data, arm, goal=..., start=..., constraint=..., holding=...)`
takes each role in whatever form you have it: a configuration, a list of
configurations, a TSR, a list of TSRs (any of them, for a start or goal; all
of them, for a constraint), or any [set](#what-a-set-is). `start` defaults to
where the arm is now. `holding="can"` carries a body rigidly with the gripper
and lets it touch the gripper, with the grasp taken from the current poses.
The goal's `Bw` bounds the pose in the frame `T`, one `[min, max]` row each
for x, y, z, roll, pitch and yaw ([Task Space Regions](#task-space-regions)).
Collision checking runs on a native copy of the MuJoCo world. The search runs
natively when every piece of the problem has a native form, and in Python
otherwise. `result.backend` says which ran; `result.backend_reasons` is empty
for a native solve and otherwise names each piece that kept it in Python
([Backends](#backends-native-and-python)).

For a gripper, the `tsr` package's hand models give the grasp regions:
`tsr.Robotiq2F85().grasp_cylinder_side(radius, height)` returns templates for
every side grasp of a cylinder, and `template.instantiate(T_world_cylinder_bottom)`
places each as a TSR on the gripper's grasp frame. Pass them all as the goal
and the planner chooses among them; `result.goal_index` says which.

## Quick start without a simulator

Three roles, three kinds of set: a finite start, a goal induced by a pose
region, and a path constraint written for this problem that only implements
membership. Run against the two-link reference arm in `sscbirrt.testing`.

```python
import numpy as np
from tsr import TSR
from sscbirrt import CBiRRT, CBiRRTConfig, FiniteSet, PlanningProblem, TSRConfigurationSet
from sscbirrt.testing import NoCollision, PlanarArm, PlanarIK

robot, ik, collision = PlanarArm(), PlanarIK(), NoCollision()
planner = CBiRRT(robot, ik, collision, CBiRRTConfig(step_size=0.1, timeout=10.0))

# Goal: end effector within 10 cm of (0, 1.5), orientation free
T = np.eye(4)
T[:3, 3] = [0.0, 1.5, 0.0]
near = TSR(T0_w=T, Tw_e=np.eye(4), Bw=np.array([[-0.1, 0.1], [-0.1, 0.1], [0, 0], [0, 0], [0, 0], [-np.pi, np.pi]]))


class RightOfWall:
    """The end effector stays at x >= -0.2 along the whole path. Membership only."""

    def contains(self, q):
        return robot.forward_kinematics(q)[0, 3] >= -0.2


problem = PlanningProblem(
    space=planner.space,
    start=FiniteSet([np.array([-0.5, 0.5])], metric=planner.space.distance),
    goal=TSRConfigurationSet(near, robot, ik, planner.space),
    validator=collision,
    path_constraint=RightOfWall(),
)
result = planner.solve(problem, seed=0)
result.success        # True
result.path           # 47 waypoints; every one has end-effector x >= -0.2
result.goal_source    # () : the goal is a single leaf, no choice to record
result.backend        # "python": result.backend_reasons names PlanarIK and RightOfWall, which have no native form
```

The path constraint has no projector, so the planner rejects any extension
that would leave it. Give the class a `project` method and extensions are
pulled onto the set instead, which is what `TSRConfigurationSet` does.

For problems whose sets are all TSRs or configuration lists, `CBiRRT.plan(...)`
is the shorthand. It builds the `PlanningProblem` for you: configuration lists
become a `FiniteSet` whose members are roots, TSR lists an `AnyOf` weighted by
TSR volume, and several `constraint_tsrs` an `AllOf` with
`MostViolatedProjection`. Its arguments differ from `sscbirrt.mujoco.plan`'s:
TSRs go in `start_tsrs`, `goal_tsrs` and `constraint_tsrs`. In this fragment,
`q_current`, `grasp_above`, `object_pose` and `upright` stand for your own values:

```python
result = planner.plan(
    start=q_current,                       # or a list of configurations
    goal_tsrs=[grasp_above(object_pose)],  # any of these regions
    constraint_tsrs=[upright],             # all of these, along the whole path
    seed=0,
    return_details=True,
)
```

Two search-strategy components are replaceable on the problem, each with a
default. A custom `motion_validator` owns the whole edge check, and
`RestrictedMotionValidator(base, accepts)` adds a restriction on top of the
default discretized check. A custom `sampler` proposes the free-space
targets the trees grow toward and defaults to the space's uniform sampling;
replacing it trades away probabilistic completeness unless it has full
support over the space. [docs/design.md](docs/design.md) has the
definitions, the composition rules, what the planner requires of each role,
the tolerances, and the reference behavior artifact that pins the planner's
semantics (`python tools/reference_artifact.py --check`).

## Result

`sscbirrt.mujoco.plan` and `CBiRRT.solve` return a `PlanResult`; `CBiRRT.plan`
returns the path, or the `PlanResult` with `return_details=True`:

| Field | Meaning |
|---|---|
| `path` | joint waypoints from a start member to a goal member, or `None` |
| `success`, `failure_reason` | `failure_reason` is `None` on success |
| `start_source`, `goal_source` | provenance through the start and goal set expressions |
| `start_index`, `goal_index` | index into `CBiRRT.plan`'s `start`/`goal` list or TSR list; 0 for single inputs |
| `iterations`, `planning_time`, `tree_sizes` | search statistics |
| `tree_start`, `tree_goal` | the two trees, for inspection |
| `backend`, `backend_reasons` | `"native"` or `"python"`, and why not native when `"auto"` chose Python |
| `provenance` | dependency versions, and for MuJoCo the scene and snapshot hashes and the SSIK family |
| `stats` | the cost of the solve by component (counts and seconds) |

### When planning fails

A plan fails in one of two ways.

**No start or goal configuration survives.** Before searching, the planner
collects the roots of each tree: every explicit configuration, and every IK
solution of every pose drawn from a region. A root must be inside the joint
limits, valid (collision-free), and inside the path constraint, if there is
one. If none survives for a role, `solve`, `CBiRRT.plan` and
`sscbirrt.mujoco.plan` raise a `PlanningError`:
`AllStartConfigurationsInCollision` or `AllGoalConfigurationsInCollision` when
every candidate was in collision, and `AllStartConfigurationsInvalid` or
`AllGoalConfigurationsInvalid` otherwise. The message counts each cause, for
example `12 IK unreachable, 3 in collision`:

| Cause in the message | What to check |
|---|---|
| IK unreachable | the region is out of the arm's reach, or the IK's frames do not match the robot model's |
| outside joint space | the region needs joint values beyond the limits |
| in collision | the scene at the start or goal, or a held object (`holding=`) touching something |
| constraint violated | the start or goal is outside the path constraint, which every point of the path, ends included, must satisfy |

**The search ends without connecting the trees.** The result has
`success=False` and a `failure_reason` that starts with `Timeout`,
`Max iterations` or `Aborted by user`, and gives the iteration count and tree
sizes. Raise `timeout`, or `num_tree_roots` to start from more members of the
sets. With a path constraint, a small tree that stops growing usually means
projection onto the constraint keeps failing: check that the constraint
leaves room to move between the start and the goal.

`backend="native"` raises `NativeUnsupported`, listing every component without
a native form, instead of planning; the default `backend="auto"` plans in
Python with that list in `result.backend_reasons`.

## How it works

<p align="center">
  <img src="docs/images/example1_result.png" alt="Basic planning" width="600">
</p>

Two trees grow at once, blue from the start set and green from the goal set.
The right panel is configuration space; red regions are in collision.

1. **Grow a set, or sample.** The trees take turns. On a tree's turn, with
   probability `start_sample_probability` (start tree) or
   `goal_sample_probability` (goal tree), the turn draws new members of that
   tree's own set and adds them as roots, and that is the whole turn. This is
   CBiRRT's `P_sample`: the trees keep gaining start and goal members
   throughout the search, not only before it. Otherwise the turn samples a
   random configuration. A finite set's members are all roots from the start,
   so its tree always samples.
2. **Extend** the tree whose turn it is toward the random configuration, in
   steps of `step_size`.
3. **Project** each new configuration onto the path-admissible set when a
   path constraint is present.
4. **Connect** the trees when one reaches the other within
   `connection_tolerance` along a validated edge.
5. **Smooth** by shortcutting; a shortcut is kept only if it is shorter and
   passes the same validation as a tree edge.

<p align="center">
  <img src="docs/images/example3_result.png" alt="Constrained planning" width="600">
  <br>
  <em>With a path constraint, the end effector stays within the yellow band throughout the motion.</em>
</p>

## Why sets

A point-to-point planner takes one start configuration and one goal
configuration. Manipulation tasks are not specified that way. A grasp is valid
anywhere around the rim of a mug. A placement is valid anywhere on a shelf.
The arm can begin from any IK solution of its current end-effector pose. A
carried cup must stay upright at every point of the motion, not only at its
ends. Each of these is a set of configurations, and using a point planner
means choosing one member of each set before planning.

That choice is made with the least information available. Whether a given
grasp configuration is reachable from a given start, under a given path
constraint, is what the search determines. Choosing first means the chosen
grasp may collide with the shelf, or lie on an IK branch the arm cannot reach
without leaving the constraint set, and the failure is found only after
planning. The usual remedy is an outer loop over IK solutions and grasp
candidates, replanning each time and rebuilding the same start tree.

Taking the sets moves the choice into the search. Both trees grow from many
roots at once, one per member drawn from the start and goal sets, so every
alternative is explored in one search and the reachable member is found
rather than guessed. Path constraints are enforced during tree growth, by
projecting each new configuration onto the constraint set or rejecting it,
not by filtering the path afterward. The result records which start and goal
member the path uses. And because the planner asks a set for nothing beyond
membership, the sets are open: a pose region, a list of configurations, a
predicate on forward kinematics, or a set of your own design are all admitted
on equal terms.

## What a set is

A state set is anything with `contains(q) -> bool`. The planner never
branches on a set's concrete type. Four capabilities are separate protocols;
a set provides them by defining the method, and `supports(s, Capability)`
asks.

| Capability | Method | Contract |
|---|---|---|
| `SetSampler` | `sample(rng) -> list[Sample]` | candidate members from one draw, unfiltered; the planner validates them |
| `SetDistance` | `distance(q) -> float` | nonnegative geometric distance to the set, before tolerance |
| `SetViolation` | `violation(q) -> float` | nonnegative and exactly zero iff `contains(q)` |
| `SetProjector` | `project(q_previous, q_proposed) -> q or None` | move a configuration onto the set, or give up |

What each role requires:

| Role | Requires | Uses if present |
|---|---|---|
| start, goal | explicit members (a finite set anywhere in the expression) or `SetSampler`, so the tree has roots | `SetDistance` for nearest-member queries |
| path constraint | membership only; an extension that leaves the set is rejected | `SetProjector`, so extensions are projected instead of rejected |
| validator | `is_valid(q)`; applied to every root and every configuration along every edge | |

Composition is explicit. `AnyOf(children, weights)` is the union: a single
child delegates every capability; with several, sampling draws a child by
weight and each `Sample` records which one. `AllOf(children, projection=,
sampling=)` is the intersection: a single child delegates; with several, the
caller names how to project (`MostViolatedProjection`) and how to sample
(`RejectionSampling(source)`), because there is no canonical way to do either
for an intersection. An intersection of finite sets is finite and
enumerable. A `Sample` carries the configuration and a provenance tuple,
the sequence of choices that produced it, which is what
`PlanResult.start_source` and `goal_source` report; a leaf contributes no
choices, so provenance is empty when the set expression has none.

Leaves provided: `FiniteSet(configs, tolerance, metric)`, `PredicateSet(fn)`,
`EmptySet()`, and `TSRConfigurationSet(tsr, robot, ik, space)`, which is
$\{q : \mathrm{FK}(q) \in \mathrm{TSR}\}$ with all four capabilities.

## Defining your own set

A set is a class. Implement `contains`; add capabilities as the role needs
them.

```python
class MySet:
    def contains(self, q: np.ndarray) -> bool:
        ...                                   # required; the only thing every role needs

    def sample(self, rng: np.random.Generator) -> list[Sample]:
        ...                                   # start/goal roots: every candidate from one draw, unfiltered

    def violation(self, q: np.ndarray) -> float:
        ...                                   # zero iff contains(q); lets strategies rank children

    def project(self, q_previous: np.ndarray, q_proposed: np.ndarray) -> np.ndarray | None:
        ...                                   # path constraints: move q_proposed onto the set, or None
```

Rules the planner relies on: `sample` returns every candidate a draw
produces (for example every IK branch of one pose) and applies no collision
or other external filter, since which branch is valid is the validator's
call; `violation` is on the set's own scale and agrees with `contains` at
zero; `project` may use `q_previous` to seed an iterative solve or to reject
a result that moved too far. Every configuration a set returns is checked
against the joint space and the validator before it is stored, so a set
cannot put an invalid configuration into a tree.

## Task Space Regions

A TSR is a pose region in SE(3): a reference frame `T0_w`, an end-effector
offset `Tw_e`, and bounds `Bw` on x, y, z, roll, pitch, yaw in the reference
frame. `TSRConfigurationSet` lifts it through the robot into the set of
configurations whose end effector lies in the region, and the TSR's geometry
supplies every capability: sampling (draw a pose, solve IK for every
branch), distance and violation (TSR distance of the forward kinematics),
and projection (the closest pose in the region, then IK seeded from the
previous configuration).

```python
from tsr import TSR

grasp_tsr = TSR(
    T0_w=object_pose,     # reference frame at the object
    Tw_e=gripper_offset,  # gripper offset from that frame
    Bw=np.array([
        [-0.01, 0.01], [-0.01, 0.01], [0, 0],  # position: ±1 cm in x and y, exact z
        [0, 0], [0, 0], [-np.pi, np.pi],       # roll and pitch fixed; yaw free
    ]),
)
```

Several TSRs in `goal_tsrs` or `start_tsrs` form a union; sampling is
proportional to each TSR's volume. Configuration lists and TSRs can be mixed
in the same role (the names below stand for your own configurations and
TSRs):

```python
path = planner.plan(start, goal_tsrs=[top_grasp_tsr, side_grasp_tsr])
path = planner.plan(start=[home1, home2, home3], goal_tsrs=[grasp_tsr])
path = planner.plan(start=current, goal=[ik_sol1, ik_sol2, ik_sol3])
path = planner.plan(start=[home], start_tsrs=[start_region], goal=[q_grasp], goal_tsrs=[grasp_region])
```

A `TSRChain` couples TSRs in series (a handle on a swinging door) and defines
one region of end-effector poses. It goes anywhere a TSR goes. A complete,
runnable chain is the door demo's `door_chain` in
[`src/sscbirrt/demo/scenarios/door.py`](src/sscbirrt/demo/scenarios/door.py):

```python
from tsr import TSRChain

door = TSRChain(TSRs=[hinge_tsr, handle_tsr])
path = planner.plan(start_config, goal_tsrs=[door])
```

For a chain of two or more TSRs, membership and projection search
numerically for the links' values (sstsr's bounded inverse). The search can
miss a pose that is in the chain, but never accepts one that is not, so every
configuration on a returned path is in the set. A miss costs coverage: fewer
start or goal roots, and some extensions under a chain path constraint
rejected. If a chain problem finds no roots or does not connect, raise
`sample_draws` and `num_tree_roots`, or widen the chain's bounds where the
task allows. Chains plan natively like single TSRs: the door demo's chain
plans in about 0.3 s natively, against 5 to 7 s in Python.

## Configuration

```python
from sscbirrt import CBiRRTConfig

config = CBiRRTConfig(
    # Termination
    timeout=30.0,                       # Wall-clock seconds
    max_iterations=100000,              # Safety limit
    abort_fn=None,                      # Callable returning True to stop early

    # Tolerances
    membership_tolerance=1e-3,          # TSR distance at which a configuration is in the set
    connection_tolerance=1e-3,          # Joint-space distance at which growth has reached its target
    edge_resolution=None,               # Spacing of validity checks along an edge; None = step_size
    progress_tolerance=1e-6,            # Growth stops when it gains less than this per step
    projection_progress_tolerance=1e-6, # Projection gives up when the violation shrinks less than this

    # Tree growth
    step_size=0.1,                      # Max joint-space step per iteration
    start_sample_probability=0.1,       # On the start tree's turn, probability of adding start roots instead
    goal_sample_probability=0.1,        # On the goal tree's turn, probability of adding goal roots instead
    max_projection_iters=50,            # Iterations to project onto the constraint set

    # Set sampling
    sample_draws=100,                    # Pose samples to try from each TSR
    num_tree_roots=100,                 # Root configs to seed each tree before the search
    max_per_draw=3,                  # IK solutions to take per pose sample

    # Extension behavior (None = connect until blocked)
    extend_steps=None,                  # Steps toward a random sample
    connect_steps=None,                 # Steps toward the other tree

    # Smoothing
    smooth_path=True,
    smoothing_iterations=50,
    smoothing_patience=15,              # Stop early after this many attempts without improvement

    # Joints with no limits (see below); None = every joint is bounded
    continuous_joints=None,
)
```

`tsr_tolerance` is a deprecated alias that sets both `membership_tolerance`
and `connection_tolerance` and warns.

`membership_tolerance`, `max_projection_iters` and
`projection_progress_tolerance` configure the sets that `plan(...)` builds
from its arguments. A set you construct yourself and pass to `solve` keeps
its own constructor values (`FiniteSet(tolerance=1e-6)`,
`TSRConfigurationSet(tolerance=1e-3, ...)`).

### Continuous joints

Mark a joint continuous (`continuous_joints`) only if it has **no limits**. The space never infers
this: a bounded joint must have finite limits, and a robot model that
reports an unlimited joint (the MuJoCo model reports ±∞) fails at planner
construction until you either mark the joint continuous or give it finite
planning limits. A continuous joint ignores whatever limits were stored for
it, samples over one full turn, and its distance wraps at 2π so the planner
may join the trees across the seam. Joints with limits
wider than one turn, such as the UR5e's ±2π joints, are not continuous: their
windings are distinct configurations and the planner respects the limits.
Returned paths are unwrapped forward from the start, so on a continuous joint
consecutive waypoints never differ by more than a step and an executor can
interpolate them directly. The goal may therefore be re-expressed by a
multiple of 2π.

### Planning variants

| extend_steps | connect_steps | Behavior |
|---|---|---|
| None | None | **CON-CON**: both trees march until blocked (default) |
| 5 | 5 | **EXT-EXT**: both trees take limited steps |
| 5 | None | **EXT-CON**: extend limited, connect unlimited |
| None | 5 | **CON-EXT**: extend unlimited, connect limited |

## Interfaces

```python
class RobotModel(Protocol):
    @property
    def dof(self) -> int: ...

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]: ...

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 end-effector pose."""


class IKSolver(Protocol):
    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Return every IK solution, unfiltered; q_init is an optional seed."""


class CollisionChecker(Protocol):
    def is_valid(self, q: np.ndarray) -> bool:
        """Return True if collision-free."""
```

Together with `StateSet`, these are the planner's extension points. Each has
a native counterpart in the C++ core (`StateSet`, `StateValidator`,
`ForwardKinematics`, `IKSolver`), and the shipped integrations below are
implementations of them, not special cases; see
[Adding an integration](#adding-an-integration).

## Backends: native and Python

The C++20 core in `cpp/` implements the same contract as the Python planner
([docs/native-design.md](docs/native-design.md)). It is built into the wheel
as `sscbirrt._native`, and since 2.0 the planner selects it by default:

```python
planner = CBiRRT(robot, ik, collision, config)                    # backend="auto": native where it can, else Python
planner = CBiRRT(robot, ik, collision, config, backend="native")  # native or NativeUnsupported; never a silent fallback
planner = CBiRRT(robot, ik, collision, config, backend="python")  # the reference implementation, always
result = planner.solve(problem, seed=0)
result.backend          # "native" or "python"
result.backend_reasons  # under "auto": why Python was chosen, one entry per component; also logged at INFO
```

Selection is decided by the problem's components, never by which optional
packages happen to import: a missing extension or adapter is itself one of
the stated reasons. The native core plans problems whose components all have
a native form: finite sets, `AnyOf`/`AllOf` with the named strategies,
`EmptySet`, the validators in `sscbirrt.testing`, any validator or IK solver
that implements the integration protocols below (the MuJoCo scene and SSIK
do), and `TSRConfigurationSet`s whose region is a `TSR` or `TSRChain` and whose IK
has a native form (SSIK around an `ssik.Manipulator` of a verified family,
the UR family `ikgeo.three_parallel`). TSRs and SSIK then run entirely in
C++: TSRs and TSR chains are sstsr's own C++ core (sstsr 3.3), and the SSIK
adapter is checked against the Python one on the UR5e. Anything else
(Python-only IK, predicates, Python-only validators, samplers, or motion
validators) makes `backend="native"` raise `NativeUnsupported` listing every
blocker, and the default fall back to Python with the same list on the
result; [Adding an integration](#adding-an-integration) is how to give a
component a native form. Before a native solve, sscbirrt translates the
problem into the core's C++ objects ("lowering"); lowering also checks that the robot model's forward kinematics
agrees with the native IK model's on the problem's explicit configurations.
The native solve releases the GIL and calls no Python after entry. Same
seed, same path within a backend; the two backends agree on outcomes and
validated paths but not on waypoints, because they use different
random-number engines. `tools/reference_artifact.py --backend {python,native,auto} --check`
runs the behavior artifact through each selection.

## Integrations

sscbirrt ships three integrations. Each implements one interface the core
defines and each lives in its own module, so none is required by the others:

| Integration | Implements | C++ target | Python |
|---|---|---|---|
| Task Space Regions (`sstsr`) | `StateSet` | `sscbirrt::tsr` | `sscbirrt.tsr_set` |
| SSIK | `ForwardKinematics`, `IKSolver` | `sscbirrt::ssik` (the only target that uses Eigen) | `sscbirrt.backends.ssik`, `native_ssik` |
| MuJoCo | `StateValidator` | `sscbirrt::mujoco`, module `sscbirrt._native_mujoco` | `sscbirrt.backends.mujoco`, `native_mujoco` |

`sscbirrt::core` depends on none of them and on nothing but the C++ standard
library. `import sscbirrt` and the native planner work with no simulator and
no IK library installed; a missing one is reported as a reason, never an
import error. Another simulator or IK library is a fourth module of the same
shape ([Adding an integration](#adding-an-integration)).

### MuJoCo

`sscbirrt.mujoco` ([above](#plan-in-mujoco)) is the way in. The pieces it is
built from are usable on their own:

```python
from sscbirrt.backends.mujoco import MuJoCoCollisionChecker, MuJoCoIKSolver, MuJoCoRobotModel

robot = MuJoCoRobotModel(model, data, ee_site="attachment_site", joint_names=joints)
collision = MuJoCoCollisionChecker(model, data, joint_names=joints)   # any contact is a collision
ik = MuJoCoIKSolver(model, data, ee_site="attachment_site", joint_names=joints, collision_checker=collision, seed=0)
```

Name the arm's joints: a model with free bodies (objects on a table) refuses
`joint_names=None`. `MuJoCoRobotModel(joint_limits=(lower, upper))` replaces the
model's limits, for example to give an unlimited joint finite planning limits.
The differential solver is stateful and, when called without a seed
configuration, tries a few random restarts within each joint's limits. Pass
`seed=` for reproducible runs; `CBiRRT.plan(seed=...)` seeds the planner only.

### Native MuJoCo scene (collision checking in C++)

With `sscbirrt[mujoco]` (pinned to `mujoco==3.14.0`, the version the extension
is built against), collision checking can run natively against a MuJoCo world
the planner owns:

```python
from sscbirrt.backends.native_mujoco import NativeScene, Snapshot, NativeCollisionChecker

scene = NativeScene.from_model(model, joint_names)              # an owned mjModel from the compiled model's MJB bytes
snap = Snapshot.capture(scene, data, attachments={"can": ("robot/gripper/base", T_gripper_can)})
checker = NativeCollisionChecker(scene, snap)                   # a CollisionChecker for either backend
```

`sscbirrt.mujoco.plan` builds the scene, the snapshot, and the checker for
you. `plan_native(model, data, joint_names, ...)` in this module did the same
before 3.1.0 and is deprecated.

The snapshot is a value: `qpos`, mocap poses, and attachments copied at capture,
so later changes to `data` do not reach a running solve. Downstream integration
is described in [docs/migration-mj-manipulator.md](docs/migration-mj-manipulator.md). The contact policy is
mj_manipulator's (a grasped object may touch its gripper; everything else that
touches the robot is a collision) and is checked against it on a checked-in
corpus. `PlanResult.provenance` records the scene's MJB hash and the snapshot
hash alongside the dependency versions; `PlanResult.stats` breaks the solve's
cost down by component on both backends.

### SSIK (analytical IK, recommended)

```bash
uv pip install "sscbirrt[ssik]"   # ssik >= 7.0
```

`SSIKRobotModel(ik)` is a `RobotModel` whose forward kinematics and limits
come from the wrapped manipulator, for planning without a simulator.

SSIK is the analytical IK backend. The EAIK backend, deprecated in 1.3.0, was
removed in 2.0 with its `eaik` extra; `SSIKSolver(ssik.Manipulator.from_prebuilt("ur5e"))`
replaces `EAIKSolver.for_ur5e(...)`.

SSIK solves 6R and 7R arms in closed form, accepts a seed, and returns every
in-limit winding of each geometric branch on joints wider than one turn, so
the planner sees the complete TSR-induced configuration set. The adapter does
no collision checking and applies no solution cap; joint limits are enforced
by `JointSpace` and collision by the planner's validator.

SSIK runs on the native backend for a 6-DOF `ssik.Manipulator` of a verified
family, today the UR family (`ikgeo.three_parallel`). Any other arm, a 7R arm
such as a Franka included, plans with SSIK on the Python backend, and
`result.backend_reasons` says so.

`sscbirrt.mujoco.Arm(..., mjcf=path)` builds SSIK for you, computes both
frame offsets from the MuJoCo model (so the arm can stand anywhere in the
world), and checks that the two agree. By hand:

```python
import ssik
import sscbirrt_assets
from sscbirrt.backends.mujoco import site_offset_in_body
from sscbirrt.backends.ssik import SSIKSolver

# From the same MJCF as the MuJoCo model, so the frames agree to machine precision
arm = ssik.Manipulator.from_mjcf(sscbirrt_assets.ur5e_xml(), base="world", ee="wrist_3_link")
ik = SSIKSolver(arm, T_ee=site_offset_in_body(model, "attachment_site"))  # the site is on wrist_3_link

# Or the prebuilt model (vendor nominal geometry); ssik.Manipulator.from_urdf takes a URDF
ik = SSIKSolver(ssik.Manipulator.from_prebuilt("ur5e"))
```

The SSIK model and the `RobotModel` must agree on joint order and sign, base
frame, end-effector frame, and which joints are continuous. If the frames
differ by fixed transforms, pass `T_base` and `T_ee` so that
`robot.forward_kinematics(q) == ik.fk(q)`, and assert that before planning.
Prebuilt models use the vendor's nominal geometry and can differ from a
simulator model by a millimeter, which matters at the default membership
tolerance.

### Adding an integration

The native backend runs a solve only when every component has a native form.
One Python-only validator, IK solver or set sends the whole solve, every
sample, extension and collision check, to the Python planner: the cost is the
whole speedup, not that component's share. The door demo shows the size. Its
TSR chain had no native form before 3.2.0; the same problem now plans in
about 0.3 s natively, against 5 to 7 s in Python. `result.backend_reasons`
names each component that kept a solve in Python, so it is the list of what
to port.

The native lowering (`sscbirrt.backends.native`) recognizes validators and IK
solvers by two protocols, never by type, so a new collision or IK backend
plugs in without a change to sscbirrt:

```python
from sscbirrt.backends.native import ValidatorIntegration, KinematicsIntegration

class MyChecker:                       # a CollisionChecker for the Python backend ...
    def is_valid(self, q) -> bool: ...
    def fresh(self):                   # ... and, for the native one, a sscbirrt StateValidator per solve
        return my_native_module.Validator(self.world_handle)
    @property
    def provenance(self) -> dict:      # what it checked against; merged into PlanResult.provenance
        return {"my_world_sha256": self.world_hash}

class MyIK:                            # an IKSolver ...
    def solve(self, pose, q_init=None) -> list: ...
    def native_kinematics(self):       # ... that is also a sscbirrt ForwardKinematics and IKSolver,
        return my_native_module.Arm(self.spec)   # or raises NativeUnsupported([reason])
    provenance = {"ik_backend": "mine"}
```

On the C++ side, subclass `sscbirrt::StateValidator` (one virtual, `is_valid`)
or `sscbirrt::ForwardKinematics` and `sscbirrt::IKSolver` in a target that
links `sscbirrt::core`, and bind it with pybind11 in your own extension
module, declaring the base registered by `sscbirrt._native` so a native
`PlanningProblem` accepts your object. `NativeCollisionChecker` and
`SSIKSolver` are the two shipped implementations of these protocols and are
the templates to copy. The rules that made them trustworthy apply to a new
one too: the native form and the Python form are one implementation or are
checked against each other on a corpus (MuJoCo: 490 configurations against
mj_manipulator; SSIK: the UR5e artifact), the library version is pinned and
verified at import, and per-solve scratch state comes from `fresh()` so a
validator is never shared between concurrent solves. Lowering itself checks
that your `native_kinematics()` agrees with the problem's `RobotModel` on the
explicit configurations, and refuses with a reason if it does not.

## Examples

```bash
# MuJoCo: a UR5e with a Robotiq 2F-85, rendered to MP4 (pip install "sscbirrt[demo]")
sscbirrt-demo                           # pick, transport, and door
sscbirrt-demo --list                    # what each one shows
sscbirrt-demo pick --seed 3             # one scenario, another seed

# A 2-DOF planar arm with matplotlib plots of the trees (no simulator)
uv pip install -e ".[examples]"
python examples/planar_arm.py           # all four
python examples/planar_arm.py -e 1      # basic planning
python examples/planar_arm.py -e 2      # start and goal TSRs
python examples/planar_arm.py -e 3      # a path constraint
python examples/planar_arm.py -e 4      # several start and goal configurations
```

The demos are the MuJoCo examples to read: each scenario in
[`src/sscbirrt/demo/scenarios/`](src/sscbirrt/demo/scenarios/) is one file
built on `sscbirrt.mujoco`, and `sscbirrt-demo` runs them.

## References

- Berenson, D., Srinivasa, S., Ferguson, D., & Kuffner, J. (2009). [Manipulation planning on constraint manifolds](https://www.ri.cmu.edu/pub_files/2009/5/berenson_icra09_cbirrt.pdf). *ICRA*.
- Berenson, D., Srinivasa, S., & Kuffner, J. (2011). [Task Space Regions: A framework for pose-constrained manipulation planning](https://www.ri.cmu.edu/pub_files/2011/10/pedestrian_ijrr.pdf). *IJRR*.

## License

MIT
