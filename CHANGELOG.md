# Changelog

All notable changes to sscbirrt (named pycbirrt before 3.0.0). The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **`max_per_draw` defaults to `None`, keeping every candidate of a draw
  (#186).** A parameter study chose this. It is in `benchmarks/ablation/`,
  run with `tools/planning_benchmark.py sweep` / `analyze`, with a decision
  rule fixed before the data. Stages:
  - screening: 240 settings of `num_tree_roots`, the sample probabilities,
    `max_per_draw` and `step_size` on pick, 20 seeds each;
  - confirmation: the shortlist on 100 fresh seeds;
  - generalization: transport and the door.

  Only `max_per_draw` beat today's defaults. On fresh seeds, pick's median
  solve is ×0.54 [0.46, 0.65] and its p90 ×0.58 [0.32, 0.92], with paths 12%
  shorter. Under the library's own defaults, pick is ×0.60 and transport
  ×0.53. Since #200, keeping all of a pose's IK solutions is cheap, and they
  give a tree roots on every branch. The other parameters were not better
  than today's values within the CI, so they stay. A sample probability of
  0 is the one setting that breaks the planner (72% success). The door
  keeps its explicit 400 roots: for a fixed grasp, fewer roots made its p90
  2.3–2.5× worse. `max_per_draw` takes `None` or an int (C++:
  `std::optional<int>`). A draw is now visited in random order whenever it
  offers more candidates than can be kept, including when the cap comes
  from `num_tree_roots` (#168).
- **The door demo reaches the handle first, then opens the door (#198).**
  It used to plan the opening first and then reach its start. That start is
  one IK solution, which can lie on a joint winding HOME cannot reach. The
  demo failed at seed 1 that way, and at seed 0 once candidate order
  changed. Reaching any grasp first, then opening from there, succeeded on
  seeds 0–9.
- **`edge_resolution` defaults to 0.05 rad, independent of `step_size` (#204).**
  It is the spacing of collision checks along an edge, a joint-space distance.
  It used to default to `step_size` (0.1 rad). On a UR5e with gripper, a point
  can move up to ~1.2 m per radian, so the old default let it move up to
  12 cm between checks and step over thin obstacles. 0.05 rad bounds that to
  ~6 cm. Every shipped demo and reference case already set 0.05, so they are
  unchanged. A problem that relied on the default gets finer checks, is
  slower, and plans different paths. Its returned paths are also denser: the
  planner stores each checked configuration, so there is a waypoint every
  0.05 rad. `edge_resolution=None` still means `step_size`. C++:
  `PlannerConfig::edge_resolution` defaults to 0.05.

## [3.3.0] - 2026-10-04

Root collection now follows CBiRRT, and it is much cheaper. Over 100 seeds
of the pick demo, every run succeeds (it was 97%), and the median solve drops
from 0.68 to 0.07 s (p90 from 1.69 to 0.18 s). Two changes, each measured
against the one before it with `tools/planning_benchmark.py`:
- The start and goal sets keep growing during the search (#196).
- IK windings of one physical configuration share a collision verdict
  (#200). This gives the same paths, faster.

Migrating from 3.2.0:
- The same seed gives a different path.
- `goal_bias` and `start_bias` are deprecated, and their meaning changed.
  Prefer `goal_sample_probability` and `start_sample_probability`; the old
  names map to them, with a warning, until 4.0.
- C++ consumers: the `PlannerConfig` fields are renamed the same way.

The start and goal sets keep growing during the search, as in CBiRRT (#196).
On each tree's turn a weighted coin, the paper's `P_sample`, either draws new
members of that tree's own start or goal set and adds them as roots, or takes
an ordinary turn toward a random configuration. Until now, roots were
collected once, before the search, and frozen. Measured over 100 seeds per
problem, with both versions interleaved on one machine
(`tools/planning_benchmark.py`):
- every run that used to time out now succeeds: pick goes from 97% to 100%
  of seeds, and the TSR-chain crank from 93% to 100%;
- the slow tail shrinks: pick's p90 goes from 1.69 to 0.84 s (worst case
  from a 30 s timeout to 1.75 s); the crank's p90 from 11.0 to 5.4 s (worst
  case from a 120 s timeout to 12.6 s); transport's p90 from 0.45 to 0.32 s;
- medians hold or improve, for example pick from 0.68 to 0.54 s, measured
  as root collection plus search;
- the door, whose up-front roots already connect it in one iteration, is
  unchanged.

### Changed
- **Behaviour.** The same seed now gives a different path, in both backends.
  Every reference-artifact case keeps its outcome and passes validation;
  native parity holds.
- **Parameters.** New `start_sample_probability` and `goal_sample_probability`
  (default 0.1) replace `start_bias` and `goal_bias`. The meaning changed: the
  old biases steered a turn's extension toward a member of the *other* tree's
  set. The new probabilities add roots to the tree's *own* set. A finite set
  never tosses the coin, because its members are all roots already. The old
  names still work until 4.0, mapped to the new ones, with a
  `DeprecationWarning` that says the meaning changed.
- **Faster root collection, same result (#200).** SSIK returns every winding
  of every IK branch: 320 solutions per pick pose, but only about 12
  physically distinct arm poses. Root collection now collision-checks each
  physical configuration once per draw instead of every winding. Pick's root
  collection goes from 0.41 s to 0.03 s, so its median solve drops from 0.53
  to 0.07 s (p90 from 0.80 to 0.18 s). Transport's median drops from 0.14 to
  0.05 s. All 700 benchmark runs give identical paths. This needs two
  declarations, never inferred:
  - the IK solver's `revolute_joints` (SSIK: every joint);
  - the collision checker's `full_turn_invariant` (the MuJoCo checkers, when
    every planned joint is a hinge).

  `stats["reused_verdicts"]` counts the shared verdicts. C++: `Sample::key`,
  `IKSolver::revolute_joints()`, `StateValidator::full_turn_invariant()` and
  `SolveStats::reused_verdicts` are new.
- **New stat.** `PlanResult.stats["search_roots"]` counts the roots added
  during the search, on both backends.
- **New tool.** `tools/planning_benchmark.py` measures planning time and work
  on realistic problems. `benchmarks/` holds the v3.2.0 baseline and this
  change's measurement.
- **C++ consumers.** `PlannerConfig::goal_bias` and `start_bias` are renamed
  to `goal_sample_probability` and `start_sample_probability`, with the new
  meaning. `Tree::add_root` is new. `SolveStats::search_roots` is new.

## [3.2.0] - 2026-10-02

TSR chains plan on the native backend (#184). sscbirrt's TSRs now come from
sstsr's own C++ core (sstsr 3.3), which carries chains, so a chain goal,
start or path constraint no longer falls back to Python. The door demo plans
its chain about 15-20 times faster than on Python.

Migrating from 3.1.0: nothing to change in Python. Install pulls
`sstsr>=3.3`. C++ consumers of the installed package need sstsr's CMake
package findable (`tsr.get_cmake_dir()`, from a built wheel; an editable
install has none).

### Changed
- `sstsr>=3.3,<4` is required, to build and to run. The native `TSR` is
  sstsr's C++, and sscbirrt's own C++ port of the TSR rules is removed.
- The native backend lowers `TSRChain` regions. Exact paths match the Python
  bit-for-bit. Elsewhere a chain's distance is sstsr's bounded inverse, an
  upper bound in both backends, held to the same properties rather than to
  the same numbers. A chain whose later link has a non-identity `T0_w` is
  refused by sstsr 3.3 itself (personalrobotics/tsr#169).
- The door demo plans only its chain and drops the single-TSR comparison.
  Its README GIF is re-rendered from the native plan.
- The reference artifact gains `ur5e_mujoco_tsr_chain_crank`: a UR5e turning
  a crank, with start, goal and path constraint one chain. It runs natively
  under `auto`.

### Removed
- C++ consumers: the RPY helpers in `sscbirrt/transform.hpp`
  (`rot_to_rpy`, `xyzrpy_to_trans` and their kin), which only the TSR port used,
  and their `sscbirrt._native` bindings. `sscbirrt/sscbirrt.hpp` no longer
  includes the TSR headers; include `sscbirrt/tsr/tsr_set.hpp`, which needs
  `sstsr_cpp` (found by `find_package(sscbirrt)`).

## [3.1.0] - 2026-09-30

sscbirrt becomes effortless to start with. `pip install "sscbirrt[demo]"`
then `sscbirrt-demo` renders MuJoCo demos with no Menagerie clone: the robot
models come from the new `sscbirrt-assets` wheel. `sscbirrt.mujoco` plans for
an arm in a MuJoCo world in one call:
`plan(model, data, Arm(model, joints, site, mjcf=...), goal=...)`. Errors
name what exists and how to fix it. Names are consistent across the API, and
the README opens with something you can run.

Migrating from 3.0.0: nothing breaks. Renamed names keep working with a
`DeprecationWarning` until 4.0:
- `ik_solver` → `ik`;
- `tsr_samples` / `max_ik_per_pose` → `sample_draws` / `max_per_draw`;
- `angular_joints` → `continuous_joints`;
- `seeds` → `explicit_samples`;
- `plan_native` → `sscbirrt.mujoco.plan`.

Two behaviours tighten:
- A MuJoCo model with free bodies now needs explicit `joint_names`.
- A malformed `start` or `goal` raises a `ValueError`. Seeded TSR plans on
  arms whose IK enumerates many solutions per pose (SSIK on the UR5e) take
  different, equally valid paths, because roots are now a random subset of
  each draw (#168).

### Added
- `sscbirrt-demo` (`pip install "sscbirrt[demo]"`): rendered MuJoCo demos on
  `sscbirrt.mujoco`, with no Menagerie clone and no system ffmpeg (#163-#165).
  - `pick`: a goal set of every side grasp of three cans among obstacles.
  - `transport`: a can carried upright over a box by a path constraint.
  - `door`: a door opened along its handle's arc by a TSR chain.
  - The README shows them as GIFs, regenerated by `tools/readme_gifs.py`.
- `sscbirrt-assets`, a separate data wheel in `assets/` that packages the
  MuJoCo Menagerie UR5e and Robotiq 2F-85 models (pinned upstream commit,
  SHA-256 manifest, upstream BSD licenses kept), released by its own
  `assets-v*` tag pipeline. It is the first step toward
  `pip install "sscbirrt[demo]"` running the demos without a Menagerie
  clone (#162).
- `sscbirrt.mujoco`, the one-call MuJoCo API (#175).
  - `Arm(model, joints, ee_site, mjcf=...)` describes the arm once and
    builds analytical IK from its MJCF. The frames that relate SSIK to the
    world are computed from the model, so an arm attached anywhere, with a
    name prefix, just works, and they are checked.
  - `plan(model, data, arm, goal=..., start=..., constraint=..., holding=...)`
    always returns a `PlanResult`.
  - `start` defaults to where the arm is now.
  - `start`, `goal` and `constraint` each take a configuration,
    configurations, a TSR, TSRs, or any set.
  - `holding="can"` takes the grasp from the current poses.

### Changed
- Errors name what exists: a missing site or joint lists the model's sites
  or joints, and the unlimited-joint error names the joint and both fixes.
  `MuJoCoRobotModel(joint_limits=(lower, upper))` gives finite planning
  limits. Install hints match pyproject. `NativeUnsupported` is exported
  from `sscbirrt` (#174).
- `CBiRRTConfig`'s set tolerances are documented as applying to the sets
  `plan(...)` builds (#177).

### Deprecated
- Consistent names; the old ones keep working, with a `DeprecationWarning`,
  until 4.0 (#176).
  - `CBiRRT(ik_solver=)` is now `ik=`. `ik` is optional: planning between
    configurations never uses it, and TSRs without it raise a clear error.
  - `CBiRRTConfig.tsr_samples` / `max_ik_per_pose` are now `sample_draws` /
    `max_per_draw`, the names the native core already used. The budgets
    apply to any sampleable set, not only TSRs.
  - `CBiRRTConfig.angular_joints` and `JointSpace(angular_joints=)` are now
    `continuous_joints`. Every revolute joint is angular; the flag means
    unlimited.
  - `sscbirrt.seeds` is now `explicit_samples`, because `seed` means an RNG
    seed everywhere else.
- `sscbirrt.backends.native_mujoco.plan_native`: use `sscbirrt.mujoco.plan`.
  It keeps working, with a `DeprecationWarning`, until 4.0 (#175).

### Removed
- The UR5e examples (`ur5e_mujoco.py`, `ur5e_transport.py`,
  `tsr_union_demo.py`), superseded by `sscbirrt-demo` on `sscbirrt.mujoco`.
  Their scene, which the reference artifact and the UR5e tests plan in, moved
  unchanged to `sscbirrt.testing.ur5e`; the artifact is bit-for-bit
  identical. `multi_config_demo.py` is now `planar_arm.py -e 4`, and
  `planar_arm.py` uses `sscbirrt.testing`'s arm and IK. The `examples` extra
  no longer installs mediapy (#166).

### Fixed
- Roots kept from one sampling draw are now a random subset of its
  candidates, as `max_ik_per_pose`'s "for diversity" promised, on both
  backends. Previously the first `max_ik_per_pose` candidates were kept,
  in the sampler's order. An IK solver that enumerates branches and joint
  windings (SSIK on the UR5e returns 256 solutions per pose) lists them in
  a fixed order, so every draw contributed near-duplicates from one
  corner of the solution set. Draws at or under the cap consume no
  randomness: the planar reference cases are bit-for-bit unchanged, and
  the three UR5e cases have new paths that pass every validation (#168).
- `dataclasses.replace(config, membership_tolerance=x)` kept the old value,
  because the deprecated `tsr_tolerance` was a stored field mirroring it.
  Deprecated names are no longer fields (#176).

- The native backend's "All N start/goal configuration(s)" count matches
  Python's; it counted each rejected explicit candidate twice (#170).
- `plan(start=[0.1, 0.2])`: a flat list is one configuration on every
  backend, and a malformed `start`/`goal` raises a `ValueError` naming the
  argument and the expected shape. It used to crash under the default
  backend with `TypeError: iteration over a 0-d array` (#171).
- `plan_native` checks its arguments before doing any work. Pose regions
  without `ik` raise a `ValueError` naming `ik`, instead of leaking `_NoIK`
  or claiming there are no pose regions. A wrong site or joint is reported
  before the scene is built. A missing native scene says that `fallback`
  cannot cover it (#172).
- `MuJoCoRobotModel`, `MuJoCoCollisionChecker` and `MuJoCoIKSolver` no
  longer take every joint by default when the model has free or ball joints,
  which silently gave the wrong DOF; they ask for the arm's joints and list
  them (#173).

### Documentation
- README images and links use relative paths, so they render on every
  branch, in pull requests and in local previews. The release build pins
  them to the release tag in the PyPI description (`tools/pypi_readme.py`),
  so PyPI shows each release's own images; before this, relative images
  were broken on PyPI.
- The README starts with installing and a runnable MuJoCo plan
  (`sscbirrt.mujoco` with the packaged UR5e), then the concepts. The
  `PlanResult` table lists `backend`, `backend_reasons`, `provenance` and
  `stats`. The MuJoCo snippets use real site names, and the tsr hand models
  are introduced for grasp regions. `tests/test_readme.py` runs the snippets
  (#178).
- The README starts with installing and a runnable MuJoCo plan
  (`sscbirrt.mujoco` with the packaged UR5e), then the concepts. The
  `PlanResult` table lists `backend`, `backend_reasons`, `provenance` and
  `stats`. The MuJoCo snippets use real site names, and the tsr hand models
  are introduced for grasp regions. `tests/test_readme.py` runs the snippets
  (#178).

## [3.0.0] - 2026-09-30

pycbirrt becomes **sscbirrt**, and this is the first release on PyPI:
`pip install sscbirrt`, with the `ssik` and `mujoco` extras. The name is the
one the C++ core, its targets, and its CMake package have carried since
1.5.0; the distribution, the import path, the extension modules, and the
repository now match it. Wheels for CPython 3.10 through 3.14 on Linux
x86_64 and macOS arm64 ship the native core with the SSIK and MuJoCo
adapters and are built by a tag-driven pipeline that runs the reference
artifact check on every wheel before publishing. Nothing about planning
changed since 2.0.0.

**Migrating from 2.0.0.** Replace `pycbirrt` with `sscbirrt` in dependencies
and imports (`import sscbirrt`, `sscbirrt.backends.native_mujoco`, and so
on); the API is identical. Repository URLs under `personalrobotics/pycbirrt`
redirect. Installing from source no longer needs an Eigen package: the build
downloads the pinned one when none is found.

### Added
- A tag-driven release pipeline (`.github/workflows/release.yml`): `v*rc*` tags
  build wheels and the sdist and publish to TestPyPI, `v*` tags publish to PyPI and
  create the GitHub release from this file's section, with Trusted Publishing and a
  guard that every artifact carries the tag's version. Wheels for CPython 3.10 to
  3.14 on manylinux_2_28 x86_64 and macOS arm64, each installed with `[mujoco,ssik]`
  and run through the reference artifact check before it is kept. libmujoco is not
  bundled: the module keeps linking the mujoco wheel's own library (#153).
- The CMake build downloads the pinned Eigen 3.4.0 (checked hash) when no Eigen3
  package is installed, so the sdist compiles on a bare machine and the wheels build
  inside the manylinux image. `SSCBIRRT_FETCH_EIGEN=OFF` restores the old requirement (#153).

### Changed
- **Renamed to `sscbirrt`.** The distribution is `sscbirrt`, the import path is
  `import sscbirrt`, the extension modules are `sscbirrt._native` and
  `sscbirrt._native_mujoco`, and the repository is `personalrobotics/sscbirrt`
  (old URLs redirect). The C++ core, targets, and CMake package were already
  `sscbirrt`. Entries below this one keep the old name as they were written (#152).

## [2.0.0] - 2026-09-30

The native core becomes the default. A `CBiRRT` built without a `backend`
argument now plans in C++ whenever every component of the problem has a
native form and in Python otherwise, saying why on the result and in the log;
`backend="native"` refuses rather than falls back and `backend="python"` is
the reference, both as in 1.x. The deprecation announced in 1.3.0 is carried
out: EAIK is gone and SSIK is the analytical IK backend. The lowering now
admits any collision or IK backend through two protocols, so MuJoCo and SSIK
are the shipped integrations of the core's interfaces rather than the only
possible ones.

**Migrating from 1.x.** Code that relied on the Python planner by default
should pass `backend="python"` explicitly; results under the default carry
`backend` and `backend_reasons`, and seeds reproduce paths within a backend
but not across the two, because their random-number engines differ. Replace
`EAIKSolver.for_ur5e(...)` with `SSIKSolver(ssik.Manipulator.from_prebuilt("ur5e"))`
(or a manipulator from the same MJCF as the simulator model) and drop the
`eaik` extra. Nothing else in the public API changed: `PlanningProblem`,
`solve`, the legacy `plan(...)`, the set types, and `plan_native` are as in
1.7.0. The mj_manipulator adoption artifact (personalrobotics/mj_manipulator#173)
named in the 2.0 gate had not been produced at release; the release stands on
pycbirrt's own gates, the behavior artifact under all three backend
selections and the MuJoCo and TSR parity corpora.

### Removed
- `EAIKSolver`, `pycbirrt.backends.eaik`, the `eaik` extra (and its place in
  `all`), and the EAIK-specific URDFs under `urdf/`. Deprecated in 1.3.0 with a
  stated 2.0 removal (#63). SSIK is the analytical IK backend:
  `SSIKSolver(ssik.Manipulator.from_prebuilt("ur5e"))` replaces
  `EAIKSolver.for_ur5e(...)`; the Python and native backends are unaffected (#81).

### Changed
- `CBiRRT` defaults to `backend="auto"`: the native core for every problem whose
  components have a native form, else the Python reference with the reasons on
  `PlanResult.backend_reasons` and logged at INFO. Pass `backend="python"` for the
  reference implementation or `backend="native"` to refuse instead of falling back.
  Selection depends on the problem's components, never on which optional packages
  import; `tools/reference_artifact.py --backend auto --check` runs the behavior
  artifact under the default and requires Python to be chosen only with a reason (#86).
- The native lowering recognizes validators and IK solvers by two runtime-checkable
  protocols in `pycbirrt.backends.native`, `ValidatorIntegration` (`fresh()`,
  `provenance`) and `KinematicsIntegration` (`native_kinematics()`, `provenance`),
  instead of by class name. `NativeCollisionChecker` and `SSIKSolver` implement them;
  the SSIK-specific checks moved into `SSIKSolver.native_kinematics()`. A third
  collision or IK backend now plugs in without a change to pycbirrt (#147).
- Reasons for a Python-only validator or IK solver name the protocol to implement;
  the FK-agreement reason says "native IK model's" rather than "SSIK model's".

### Documentation
- README: the shipped backends presented as three integrations of the core's
  interfaces, with an "Adding an integration" section; design doc: an
  "Integrations" section with the checklist for a new simulator or IK library (#147).

## [1.7.0] - 2026-09-30

Collision checking joins the native path. A MuJoCo world becomes a scene the
planner owns, loaded from the compiled model's MJB bytes, and every query runs
against an immutable snapshot with mj_manipulator's attachment-aware contact
policy implemented in C++ and checked against it on a corpus. One call,
`plan_native`, takes a live model and data to a native solve with SSIK and
native collision, and every result carries provenance and a cost breakdown.
The scene lives in its own extension module, so `import pycbirrt` needs no
MuJoCo, and the `mujoco` extra pins the exact version the module is built
against. The default backend is still Python. Released together with 1.6.0
from the same commit; the two milestones were planned and verified separately.

### Added

- **Native MuJoCo collision checking.**
  `pycbirrt.backends.native_mujoco`: `NativeScene.from_model` (an owned
  `mjModel` from the compiled model's MJB bytes, cached by content),
  `Snapshot.capture` (qpos, mocap poses, and attachments as a value), and
  `NativeCollisionChecker(scene, snapshot)`, a `CollisionChecker` with
  mj_manipulator's attachment-aware contact policy implemented in C++ and
  used by both backends; `backend="native"` lowers it to a per-solve
  validator with its own `mjData`. Decisions are checked against
  mj_manipulator's checker on a checked-in corpus
  (`tools/mujoco_collision_corpus.py`) (#93, #84, #137, #138).
- `plan_native(model, data, joint_names, ik=..., start=..., goal_tsrs=..., attachments=...)`:
  one call from a live MuJoCo world to a native solve (scene, snapshot,
  native collision, SSIK), with `fallback=True` for the Python planner on
  unsupported input. The reference artifact gains two Menagerie-gated UR5e
  cases in the example scene, a TSR goal among obstacles and a held object
  with the grasp contact allowed, recorded as skipped where the Menagerie is
  absent; `docs/migration-mj-manipulator.md` describes the boundary for
  mj_manipulator (#88, #140).
- `PlanResult.provenance` (dependency versions, backend, and for a native
  MuJoCo solve the scene's model signature, MJB hash, and snapshot hash;
  the SSIK family when SSIK lifted a set) and `PlanResult.stats` (counts
  and seconds per component: roots, search, smoothing, state checks, edge
  checks, set samples, set projections) on both backends.
  `tools/benchmark_native.py` records both backends' timings with that
  breakdown (#89, #139).

### Changed

- The `mujoco` extra pins `mujoco==3.14.0` exactly, and mujoco is a build
  requirement of the extension: the native MuJoCo scene
  (`pycbirrt._native_mujoco`, a separate module so `import pycbirrt` never
  loads MuJoCo) is built against that version and refuses another with a
  message naming the compiled, loaded, and installed versions (#137). The
  workspace lock moves to 3.14.0.

## [1.6.0] - 2026-09-30

TSR planning goes native. When a problem's regions are single TSRs and its
IK is SSIK on a verified family, the whole solve, sampling, projection, and
IK included, runs in C++ with no Python callback, and the reference
artifact records the proof. The TSR math is checked against sstsr on a
conformance corpus and the SSIK adapter against the Python one on the
UR5e. Everything else falls back to Python with a stated reason, and the
default backend is still Python, so `plan(...)` behaves as in 1.5.0.

### Added

- **Native TSR planning.** `backend="native"` and `"auto"` now run
  problems with `TSRConfigurationSet`s entirely in C++ when the region is a
  single `TSR` and the IK is an `SSIKSolver` around an `ssik.Manipulator`
  of a verified family (`ikgeo.three_parallel`, the UR family). The TSR
  math is sstsr 3.2.0's, checked on a checked-in conformance corpus
  (`tools/tsr_conformance.py`, 162 probes); the SSIK adapter calls ssik
  7.0's header-only solver and is verified identical to the Python adapter
  on the UR5e. Lowering checks that the robot model's FK agrees with
  SSIK's on the problem's explicit configurations. TSR chains, other IK
  backends, and other families fall back explicitly with a reason
  (#87, #90, #127, #128, #129, #130).
- `SSIKRobotModel`: a `RobotModel` from an `SSIKSolver`, for planning
  without a simulator. The reference artifact gains a UR5e case (finite
  start, `AnyOf` of two grasp TSRs by volume, a workspace path TSR) whose
  native run records zero Python calls during the solve.

### Fixed

- `examples/tsr_union_demo.py` built its side-grasp frame from a
  left-handed triad (a reflection, determinant −1), which sstsr 3.2's
  constructor now rejects; the gripper x axis is the right-handed
  completion of the approach and up directions.

### Changed

- The `ssik` extra requires `ssik>=7.0,<8`: 7.0 ships the `ssik_cpp`
  headers and `ssik.cpp.joint_data`, which the native SSIK adapter is built
  on, and ssik is a build requirement of the extension (#129;
  personalrobotics/ssik#641). The Python `SSIKSolver` adapter is unchanged.
- Depends on `sstsr>=3.2,<4`: a `TSR` now rejects a malformed region at
  construction (non-finite or non-rigid frames, non-finite bounds) with the
  tolerance exported as `tsr.FRAME_ATOL`. This is the shared construction
  contract with the native TSR runtime (#127; personalrobotics/tsr#162).

## [1.5.0] - 2026-09-29

The native core arrives as an explicit opt-in. A C++20 implementation of the
planner, specified by `docs/native-design.md` as one contract with the Python
reference, plans finite and composed problems with no Python in the loop and
is gated against the reference artifact. The default backend is still Python,
so `plan(...)` and `solve(...)` behave exactly as in 1.4.0. The Python
reference also adopted four rules the contract needed stated sharply: finite
limits on bounded joints, config range validation, three cancellation
points, and a replaceable free-space sampler. Installing from source now
compiles the extension and needs CMake and a C++20 compiler; wheels include
it.

### Added

- **Native core (opt-in).** A C++20 implementation of the planner under
  `cpp/` (`sscbirrt::core`, standard library only), built into the wheel as
  `pycbirrt._native` with scikit-build-core. `CBiRRT(..., backend="python"
  | "native" | "auto")`; default `"python"`, so nothing changes for
  existing callers. `"native"` plans problems whose components all have a
  native form (finite sets, `AnyOf`/`AllOf` with the named strategies,
  `EmptySet`, the `pycbirrt.testing` validators) with the GIL released and
  no Python callbacks, or raises `NativeUnsupported` listing every blocker;
  `"auto"` falls back to Python and records the reasons on
  `PlanResult.backend_reasons`. `PlanResult.backend` names the
  implementation. Same seed, same path within a backend; the backends agree
  on outcomes and validated paths, not on waypoints (#85, #116, #117, #118).
  The C++ package installs and exports as `sscbirrt::core` for standalone
  consumers (`find_package(sscbirrt)`); `cpp/examples/consumer/` is one, built
  against the installed package in CI. `tools/reference_artifact.py --backend
  native --check` is the parity gate: the native core must match the Python
  artifact's semantic view on every case it supports (#119).
- `PlanningProblem.sampler`: a replaceable free-space sampler
  (`SpaceSampler`, `sample(rng) -> q`) that proposes the targets the trees
  grow toward. None means the space itself, uniform, so defaults are
  unchanged. Start and goal bias stay in the planner (#110).

### Changed

- `abort_fn` is also polled before each sampling draw during root
  collection and before each smoothing attempt, not only once per search
  iteration. Firing during roots returns an Aborted result with the roots
  gathered so far; firing during smoothing returns the path found, as
  smoothed so far, as a success (#109).
- `CBiRRTConfig` validates its ranges at construction and raises
  `ValueError` naming the field and value: positive step size, timeout, and
  progress tolerances; nonnegative membership and connection tolerances;
  counts at least 1; biases within [0, 1]; `edge_resolution`,
  `extend_steps`, and `connect_steps` None or positive. Previously only
  `edge_resolution` was checked (#108).
- **Joint topology is the caller's declaration.** `JointSpace` rejects a
  non-finite limit on a joint not marked angular ("give finite planning
  limits or mark it angular"), and an angular joint ignores its stored
  limits and samples over one full turn. `MuJoCoRobotModel.joint_limits`
  reports unlimited joints as ±∞ instead of MuJoCo's stored `(0, 0)`, so a
  model with a continuous joint now fails at `CBiRRT(...)` construction
  unless `angular_joints` declares it, where before the joint was silently
  frozen at zero (#107).

## [1.4.0] - 2026-09-28

The Python reference implementation is frozen. This release adds TSR chains,
moves to sstsr 3.1, fixes the returned path on angular joints, adds the
constrained-transport example, and records the deterministic reference
artifact that the native backends in v1.5 through v2.0 must reproduce.
`plan(...)` and the set model are unchanged; the minor bump is for the
additive `PoseRegion` protocol, `pycbirrt.testing`, and the sstsr
requirement.

### Added

- **TSR chains.** `TSRConfigurationSet` accepts a `tsr.TSRChain` as well as a
  `tsr.TSR`: both satisfy the new `PoseRegion` protocol (distance, closest
  world-frame transform, seeded sampling), so a chain is one configuration
  set, one alternative in `AnyOf`, one factor in `AllOf`, and one entry in
  the legacy `goal_tsrs` / `start_tsrs` / `constraint_tsrs` lists.
  `region_volume` and `tsr_weights` handle chains. For a multi-TSR chain,
  membership and projection use sstsr's bounded numerical inverse, so they
  cost a few milliseconds and membership can be a false negative on a hard
  chain (#7).

- `pycbirrt.testing`: the two-link planar reference arm (`PlanarArm`,
  `PlanarIK`, `NoCollision`, `Wall`) as an importable module.
- `tools/reference_artifact.py` and `tests/reference/python_reference.json`:
  the deterministic Python reference artifact, a fixed twelve-case matrix
  with fixed seeds, implementation versions, results, and an independent
  validation report. `tests/test_reference_artifact.py` regenerates it and
  fails on any semantic change. It is the definition of "compatible with
  the Python reference" for the native backends (#94).
- `examples/ur5e_transport.py`: constrained transport on the UR5e, the
  gripper kept pointing down from one side of the base to the other, planned
  with and without the constraint and reporting each path's largest tilt.
  Same backend selection as the other UR5e examples (#78).

### Fixed

- Paths on angular (limit-free) joints are unwrapped forward from the first
  waypoint before being returned, so an executor interpolating raw joint
  values no longer sees a full-turn jump where the two trees met or a
  shortcut ended. The start is returned as given; the goal may be
  re-expressed by a multiple of 2π on an angular joint. Paths without
  angular joints are unchanged (#77).

### Changed

- Depends on `sstsr>=3.1,<4` from PyPI (imported as `tsr`). The TSR adapter
  now samples poses with `TSR.sample(rng=...)` and projects with
  `TSR.closest_transform`, both added upstream in response to
  personalrobotics/tsr#52 and #53, instead of reading the TSR's private
  continuous bounds and composing the frames by hand.

## [1.3.0] - 2026-09-19

The analytical IK backend moves from EAIK to SSIK. `plan(...)` and the
planner core are unchanged; the minor bump is for the new `pycbirrt[ssik]`
extra, the reduced `IKSolver` protocol, and the EAIK deprecation. Users of
the MuJoCo differential fallback should read the Fixed section: it was not
runnable from the examples in 1.2.0 and its unseeded behavior has changed.

### Added

- **SSIK backend** (`pycbirrt.backends.ssik.SSIKSolver`, extra
  `pycbirrt[ssik]`, requires `ssik>=6.0.1,<7`): enumerative analytical IK
  for 6R and 7R arms. The adapter wraps an `ssik.Manipulator` or a prebuilt
  artifact, forwards `q_init` as SSIK's seed, requests limit-respecting
  solutions with in-limit winding enumeration, applies no solution cap, does
  no collision checking, and accepts optional fixed `T_base` / `T_ee`
  transforms for frame conformance. Its `fk` lets you assert the frame
  contract against your `RobotModel` before planning (#63).
- `pycbirrt.backends.mujoco.site_offset_in_body(model, site)`: the fixed
  transform to pass as `T_ee` when SSIK is built from the same MJCF with the
  end-effector body as `ee`; the UR5e examples and integration test use it and
  match MuJoCo's forward kinematics to machine precision.

### Fixed

- The UR5e examples' MuJoCo differential-IK fallback crashed on the first
  IK update because the collision checker was passed in the `joint_limits`
  positional slot. Both examples now select the backend through
  `build_ik_solver(..., backend)` with a `--ik {auto,ssik,mujoco}` flag, and
  the fallback is exercised by tests even when SSIK is installed (#65).
- `MuJoCoIKSolver.solve` without a seed tries the current state and then
  `restarts` (default 3) random initial configurations within limits and
  returns every distinct converged solution. Previously it depended on
  whatever the shared MuJoCo state was last left in and converged on only
  about half of reachable poses, which made TSR goal sampling through the
  fallback unreliable.
- `MuJoCoIKSolver` restart windows are anchored at the current configuration
  and derived from each joint's MuJoCo type and limits: hinges sample one
  turn around the anchor intersected with their limits, limited slides their
  whole interval, unlimited slides ±1 around the anchor. The previous blanket
  intersection with [-π, π] raised for a valid interval such as [4, 5] and
  applied a revolute rule to prismatic joints. The examples forward their
  `--seed` to the fallback so runs are reproducible (#69).
- `SSIKSolver` copies `T_base` and `T_ee` at construction and stores them
  read-only, so mutating the caller's array can no longer desynchronize the
  stored transform from its cached inverse (#66).

### Changed

- The `IKSolver` protocol requires only `solve(pose, q_init)`. `solve_valid`
  is no longer part of the interface; the planner never called it, and joint
  limits and collision are the planner's responsibility. Existing backends
  keep it as a convenience.
- The UR5e examples prefer SSIK and fall back to MuJoCo differential IK.
- Because SSIK returns every in-limit winding, the TSR-induced set on
  joints wider than one turn is now complete; #36 is resolved in the IK
  backend, not in the planner.

### Deprecated

- `EAIKSolver` warns on construction. It, the `eaik` extra, and its
  documentation will be removed in pycbirrt 2.0.

## [1.2.0] - 2026-09-17

A hardening release after two adversarial reviews of the 1.1.0 set-based
planner (#42 to #47, #54 to #57). Every item below was reproduced on 1.1.0
and is guarded by tests written against the invariant it restores. The
`plan(...)` signature is unchanged; the minor bump is for the new
motion-validation API and the new set capabilities. Users of custom sets or
validators should read the Fixed section: paths, roots, and projection
results can differ from 1.1.0 where 1.1.0 was wrong.

### Added

- `JointSpace.contains(q)` and `why_invalid(q)`: the authoritative
  membership test for the joint space (shape, finiteness, limits) with a
  reason naming the offending joint (#43).
- `seeds(s)`: the explicit configurations embedded in any set expression,
  distinct from exhaustive `members(s)` (#42, #55).
- `SetViolation` capability: `violation(q)` is zero exactly when the set
  contains `q`; `TSRConfigurationSet` and `FiniteSet` implement it, and
  `AnyOf`/`AllOf` compose it (#44).
- `MotionContractError`, raised when a `MotionValidator` violates the
  `LocalMotion` contract (#54).
- `RestrictedMotionValidator(base, accepts)` and
  `CBiRRT.default_motion_validator(problem)` for explicit composition of a
  motion restriction with the default checks (#56).
- `PlanningProblem.motion_validator`: local-motion validation is an
  explicit, replaceable boundary. `MotionValidator.validate(q_from, q_to)`
  returns a `LocalMotion` (the validated configurations to store, and
  whether the target was reached), and every tree edge, the final
  connection between trees, and every shortcut go through it. The default
  `DiscreteMotionValidator` reproduces the discretized behavior at
  `edge_resolution`. A custom validator *replaces* the default and owns the
  motion's interior; the planner independently checks every configuration
  it returns before storing it. `RestrictedMotionValidator(base, accepts)`
  composes an extra restriction with a base validator (typically
  `planner.default_motion_validator(problem)`) to be stricter while keeping
  the default checks. Goal-tree edges are validated in the reverse of
  execution direction (#46, #56).

### Fixed

- The planner checks a `MotionValidator`'s whole `LocalMotion` before
  touching the tree. `reached=True` with no configurations on a nonzero
  motion, or ending anywhere but the exact target, raises the new
  `MotionContractError` (a validator bug, not a planning failure). A
  claimed success containing an inadmissible configuration is rejected
  whole; a partial result keeps its admissible prefix; a zero-length
  motion succeeds without adding a node whatever the payload (#54).
- `MostViolatedProjection` required the single largest violation to
  decrease after every projection, so an intersection with tied
  violations (two axes from a corner) was reported as stalled after the
  first child was satisfied. Progress is now the lexicographic decrease
  of the sorted violation profile; it stops after a full sweep without
  progress, when a projector makes no change, or at `max_iters` (#57).
- `members` of a finite intersection enumerates a child that is itself
  finite, not the first child that merely has explicit seeds, so an
  `AllOf` whose first seed-bearing child is a mixed union still
  enumerates exhaustively and can seed a search. `seeds` of a non-finite
  intersection now collects from every seed-bearing child, keeps what the
  whole intersection contains, and deduplicates equal configurations by
  first occurrence (#55).
- Membership in the ambient `JointSpace` (shape, finiteness, joint limits)
  is now the first admissibility check for every root, sample, projected
  extension, and edge sample. Out-of-limit or malformed fixed starts and
  goals are rejected before search with the `...Invalid` exceptions and a
  reason that names the joint; samplers and projectors that return
  out-of-space configurations can no longer get them into a tree (#43).
- Root sampling raises the `...InCollision` exception only when every
  rejected candidate was a collision; any other reason makes it `...Invalid`.
- Fixed configurations are always tree roots, including when a start or
  goal role also has sampled TSR alternatives. `plan(start=[q],
  start_tsrs=[...])` silently dropped `q` in 1.1.0 because the mixed union
  was not finite. New `seeds(s)` enumerates the explicit configurations
  embedded in any set expression; mixture weights now govern sampling
  only (#42).
- **Tree connection and shortcuts are validated to the exact target.**
  Growth reported success when merely within `connection_tolerance` of its
  target, leaving the final gap unchecked in the returned path, and shortcut
  smoothing snapped its last waypoint to the target without validating the
  segment (and could drop the shortcut's start when growth succeeded
  immediately). Now `reached` means the tree contains the exact target
  through a validated edge; the edge routine validates every sample
  including the endpoint; the join between trees carries no duplicate; and
  smoothing accepts a shortcut only when its joint-space length is shorter
  than the segment it replaces. Paths keep their first and last waypoints
  exactly (#47).
- `MostViolatedProjection` ranked children by raw `distance`, so a satisfied
  child with a large tolerance could outrank an unsatisfied one and the
  projection stalled on a feasible intersection. New `SetViolation`
  capability: `violation(q)` is zero exactly when the set contains `q`
  (`max(0, distance - tolerance)` for TSR and finite sets; min over a
  union, max over an intersection). The strategy now selects only among
  unsatisfied children by violation, and requires it of every child (#44).
- Nearest-neighbor selection uses the query's `PlanningProblem.space`, not
  the planner's construction-time space, so a direct `solve(problem)` with
  a different joint topology is internally consistent (#45).

## [1.1.0] - 2026-09-16

Additive and backward compatible. `plan(...)` keeps its signature and
semantics; every 1.0.0 test passes unchanged.

### Added

- **Planning between sets.** `CBiRRT.solve(problem)` takes a
  `PlanningProblem` made of a `JointSpace`, a start set, a goal set, a
  validator, and an optional path-admissible set. `plan(...)` now lowers its
  arguments into this representation. See `docs/design.md`.
- **State sets** (`pycbirrt.sets`): a membership-only `StateSet` protocol
  with optional `SetSampler`, `SetDistance`, and `SetProjector` capabilities;
  `FiniteSet`, `PredicateSet`, `EmptySet`; `AnyOf` (union) and `AllOf`
  (intersection) with arbitrary nesting and explicit capability rules; the
  named strategies `MostViolatedProjection` and `RejectionSampling`;
  `supports`, `is_finite`, `members`.
- **`TSRConfigurationSet`**, the configuration-space set a TSR induces
  through forward kinematics, with membership, distance, sampling, and
  IK-based projection. `tsr_weights` gives the volume-weighted mixture.
- **`JointSpace`**: joint limits, angular-aware metric and direction,
  interpolation, and uniform sampling in one object (`planner.space`).
- **Distinct tolerances** on `CBiRRTConfig`: `membership_tolerance`,
  `connection_tolerance`, `edge_resolution`, `progress_tolerance`,
  `projection_progress_tolerance`. Defaults reproduce 1.0.0 behavior.
- `PlanResult` gains `start_source` and `goal_source` (which alternative a
  path used, as a path through nested sets), `failure_reason`,
  `planning_time`, `tree_sizes`, and the search trees `tree_start` and
  `tree_goal` for inspection.
- `abort_fn` on `CBiRRTConfig` to stop planning early (#12).
- Failed root sampling reports why: IK unreachable, in collision, or
  constraint violated.
- Integration test on the UR5e in MuJoCo (skipped without the menagerie),
  hypothesis property tests for the set laws, and headless smoke tests for
  the examples.
- MIT license, SPDX headers, issue and PR templates.

### Changed

- **Sampling from TSRs is reproducible under a seed.** 1.0.0 used numpy's
  global random state for poses, so seeded plans were not repeatable.
- Goal and start bias apply to any sampleable set, including finite sets.
- With several constraint TSRs, projection converges each TSR before
  re-evaluating which is most violated. Single-TSR constraints are unchanged.
- Bias samples are collision-filtered by the validator rather than by
  `IKSolver.solve_valid`; only `solve` is required of a solver for sets.
- TSR-sampled goal configurations are collision-checked before becoming tree
  roots.
- The tsr package is imported only by `pycbirrt.tsr_set` and
  `pycbirrt.legacy`.

### Deprecated

- `CBiRRTConfig.tsr_tolerance`. Passing it sets `membership_tolerance` and
  `connection_tolerance` together and emits a `DeprecationWarning`; reading
  it returns `membership_tolerance`.

### Fixed

- **Constraint projection with non-identity TSR frames.** The closest
  in-bounds point was used as a world pose without composing `T0_w` and
  `Tw_e`, so projection failed whenever a constraint TSR had a translated or
  rotated frame (#28).
- **Do not mark bounded joints as angular.** The UR5e examples marked its
  ±2π joints angular, which let the planner join configurations a full turn
  apart and spin a joint through 360° on execution (#35). `angular_joints`
  is for genuinely unlimited joints only; the config comment now says so.
  If you copied the old example configuration, remove `angular_joints`.
- `examples/planar_arm.py` and `examples/multi_config_demo.py` crashed on a
  result field removed in March (#32).
- Collision checker state no longer goes stale between queries (#11).
- CI: the formatter version is pinned to the 0.15 series and a stale lock
  file that could never be honored was removed (#26).

### Known limitations

- On joints with a range wider than 2π, `TSRConfigurationSet` offers only
  the IK solver's representative of each branch, not its in-limit 2π twins,
  so a short route through the twin can be planned as a long one (#36).
- TSR chains are not yet supported (#7).

## [1.0.0] - 2026-02-02

Initial release: CBiRRT with TSR start, goal, and path constraints; MuJoCo
and EAIK backends; planar arm and UR5e examples.

[1.7.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.6.0...v1.7.0
[1.6.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/personalrobotics/pycbirrt/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/personalrobotics/pycbirrt/releases/tag/v1.0.0
