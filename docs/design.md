# Design: planning between sets

sscbirrt plans between sets. This document defines the objects the planner
works with, what each one means, and what each one is allowed to do. It is
the reference for the `PlanningProblem` API and for anyone adding a new kind
of set. The TSR-specific entry point, `plan(...)`, is one instantiation of
this design and is described at the end.

The native C++20 implementation of this design is specified in
[native-design.md](native-design.md); that document is the boundary the
native backends must honor and records every place it differs from Python.

## The problem

Let $\mathcal{Q}$ be the configuration space. A planning problem names

- a **start set** $\mathcal{S} \subseteq \mathcal{Q}$,
- a **goal set** $\mathcal{G} \subseteq \mathcal{Q}$,
- a **path-admissible set** $\mathcal{C} \subseteq \mathcal{Q}$, and
- a **validator**, a predicate on $\mathcal{Q}$ (collision, in practice).

A solution is a path $\tau : [0, 1] \to \mathcal{Q}$ with $\tau(0) \in \mathcal{S}$,
$\tau(1) \in \mathcal{G}$, $\tau(t) \in \mathcal{C}$ for all $t$, and every
$\tau(t)$ valid. In code this is `PlanningProblem(space, start, goal,
validator, path_constraint)` and `CBiRRT.solve(problem)`.

These are **roles**. A set does not know which role it plays, and the same
set can serve as a goal in one problem and a path constraint in another.

Two further components of a problem are search strategy rather than
specification, and both are replaceable with a default: the **motion
validator** (below, under edges) and the **sampler**
(`PlanningProblem.sampler`, a `SpaceSampler` with `sample(rng) -> q`), which
proposes the free-space targets the trees grow toward. The default sampler
is the space itself, uniform within the limits and over one full turn on
continuous joints. Start and goal bias remain the planner's: they mix the role
sets' own samplers with the free-space sampler, and a custom sampler is not
consulted for roots or bias draws. A target outside the space is handled by
not growing toward it. Replacing the default trades away probabilistic
completeness unless the replacement has full support over the space; that
is the caller's responsibility and the planner does not check it. Samplers
that need the trees (sampling around existing nodes) do not fit this
interface and are deliberately not supported until a concrete strategy
needs them.

## The state space

`JointSpace(lower, upper, continuous_joints)` owns the geometry of
$\mathcal{Q}$: joint limits, the distance metric, the direction between two
configurations, straight-line interpolation, and uniform sampling. The planner
and every set that needs a metric share one instance.

A joint is **continuous** only if the caller says so, and the space never
infers it. A bounded joint must have finite limits, because a sampler needs
a bounded domain; the constructor rejects a non-finite limit on a joint not
marked continuous. A continuous joint has no limits: whatever was stored for it
is ignored, its limit check always passes, it samples over one full turn,
and its distance wraps at $2\pi$. Backends report unlimited joints as
$\pm\infty$ (the MuJoCo model does; MuJoCo itself stores them as $(0, 0)$),
so an undeclared continuous joint fails at construction rather than being
silently frozen or never sampled. Unbounded is not periodic: a rail without
stops needs finite planning limits, not the continuous flag. A joint with
limits is a bounded interval however wide its range. The UR5e's $\pm 2\pi$ joints are bounded: a value and
that value plus one turn are different joint states, and moving between them
is a real full rotation. Marking such a joint continuous lets the planner join
configurations a turn apart and the returned path spins the joint through
360° on execution.

## Sets: membership is the only requirement

A **state set** is anything with

```python
def contains(self, q: np.ndarray) -> bool
```

That is the whole definition. Fixed configurations, finite collections,
predicates, and TSR-induced sets are all state sets. The planner never
branches on a set's concrete type.

Leaves provided by `sscbirrt.sets`:

| Set | Members | Capabilities |
|---|---|---|
| `FiniteSet(configs, tolerance, metric)` | the listed configurations, within `tolerance` under `metric` | membership, distance, sampling |
| `PredicateSet(fn)` | whatever `fn` accepts | membership only |
| `EmptySet()` | nothing | membership only; finite |
| `TSRConfigurationSet(tsr, robot, ik, space, tolerance)` | $\{q : \mathrm{FK}(q) \in \mathrm{TSR}\}$ | membership, distance, sampling, projection |

## Capabilities: what a set can do, separately from what it is

Sampling, distance, and projection are optional. A set declares them by
providing the method; `supports(s, Capability)` asks.

- **`SetSampler.sample(rng) -> list[Sample]`** draws candidate members. One
  draw may yield several candidates, for example every IK branch of one
  sampled pose, or none. The caller validates them. A set never applies
  collision or other external filters to its own samples; that is the
  validator's job. This matters: which IK branch is collision-free is not
  knowable inside the set, and discarding branches before validation was a
  real bug.
- **`SetDistance.distance(q) -> float`** is a nonnegative geometric measure
  of how far `q` is from the set, before any tolerance is applied. For a
  leaf it is within the set's membership tolerance exactly when
  `contains(q)` holds; for composites it is only a summary (min for unions,
  max for intersections) with no membership threshold of its own.
- **`SetViolation.violation(q) -> float`** is nonnegative and exactly zero
  when `contains(q)` holds, on the set's own scale: for a TSR-induced set it
  is the TSR distance beyond the membership tolerance. Unions take the min
  and intersections the max, so `violation(q) == 0` agrees with membership
  at every level. Strategies that compare violations across children require
  them to be on a comparable scale, which the strategy cannot verify.
- **`SetProjector.project(q_previous, q_proposed) -> q | None`** moves a
  configuration onto the set. `q_previous` is where the extension started; a
  projector may use it to seed an iterative solve or to reject results that
  moved too far.

A **`Sample`** carries `q` and a `source` tuple: the sequence of choices that
produced it, outermost first. `AnyOf` prepends the index of the child it
chose; `FiniteSet` contributes the index of the member; other leaves
contribute nothing. This is how `PlanResult.start_source` and `goal_source`
report which alternative a path used.

## Composition

Two constructors, nesting freely:

- **`AnyOf(children, weights=None)`**: $q \in \bigcup_i C_i$.
- **`AllOf(children, projection=None, sampling=None)`**: $q \in \bigcap_i C_i$.

Membership is exact Boolean semantics in both cases, always available, and
grouping is preserved: $(L_1 \cap R_1) \cup (L_2 \cap R_2)$ describes two
matched pairings, while $(L_1 \cup L_2) \cap (R_1 \cup R_2)$ allows all four.

Capabilities of a composite are explicit, never implied:

| | `AnyOf` | `AllOf` |
|---|---|---|
| one child | delegates every capability to it | delegates every capability to it |
| distance | min over children, if all support it | max over children, if all support it (a geometric summary, not membership) |
| violation | min over children, if all support it | max over children, if all support it (zero iff member) |
| sampling | needs `weights`, the mixture policy; all children must sample | needs a named strategy, e.g. `RejectionSampling(source)` |
| projection | nearest successful child projection; all children must project | needs a named strategy, e.g. `MostViolatedProjection()` |

The single-child rule means a one-TSR constraint costs nothing and needs no
strategy. The multi-child `AllOf` rules are deliberate: there is no generic
way to sample or project onto an intersection, and the strategies that exist
are heuristics with names. Asking a composite for a capability it lacks
raises `UnsupportedCapability` with the reason.

Named strategies so far:

- `MostViolatedProjection(max_iters, progress_tolerance)`: repeatedly project
  onto the unsatisfied child with the largest `violation` until every child
  contains the point. A satisfied child is never selected. Progress is the
  lexicographic decrease of the descending-sorted violation profile, so
  clearing one of several equally violated children counts even though the
  maximum is unchanged; it gives up after a full sweep without progress,
  when a projector returns None or leaves the point unchanged, or at
  `max_iters`. Children must support violation and projection, and their
  violations must be comparable (homogeneous TSR sets are).
- `RejectionSampling(source)`: draw from one child and keep the candidates the
  others contain. Exact, but wasteful when the intersection is small.

Finite sets compose: a union of finite sets is finite, and an intersection
with a finite child is finite (enumerate that child, keep what the others
contain). `is_finite` and `members` implement this and the planner uses them
to collect roots.

No `Not`. Exclusion is the validator's job, and complements have no useful
sampler or projector.

## What the planner requires of each role

- **Every accepted configuration is a member of the joint space.** The
  first admissibility check for every root, sample, projected extension,
  and edge sample is `space.contains(q)`: a finite numeric array of shape
  `(dof,)` with every bounded joint inside its limits. A set, sampler, or
  projector that returns anything else is rejected with the reason
  "outside joint space", classified as invalid rather than as a collision.
  Nothing outside `problem.space` is ever stored in a tree. Concrete sets
  may filter limits early as an optimization, but correctness does not
  depend on it.
- **Start and goal** must be finite, sampleable, or both. Every explicit
  configuration embedded in the set is a candidate root: `explicit_samples(s)` walks
  the expression and collects the members of finite sets, including those
  inside a union with a sampleable region, so mixture weights never decide
  whether a fixed configuration is a root. If the set is not finite and can
  sample, admissible candidates are added until `num_tree_roots` roots
  exist or the draw budget (`sample_draws`) is spent, keeping at most
  `max_per_draw` per draw (default `None`: every admissible candidate,
  since #186) and skipping candidates that repeat a seed. A draw with more
  candidates than can be kept (`max_per_draw` or the roots still wanted,
  whichever is smaller) is visited in a
  random order from the planner's RNG, so the kept ones are a uniform
  subset; an IK solver that enumerates branches and joint windings lists
  hundreds in a fixed order, and the first few are one corner of that set
  (#168). Bias sampling draws from any sampleable set.
- **Roots are taken from $\mathcal{S} \cap \mathcal{C}$ (and $\mathcal{G} \cap \mathcal{C}$) by
  rejection**: a candidate outside the path constraint or rejected by the
  validator is not a root. If no root survives, the planner raises the
  `All…Invalid` or `All…InCollision` exception for that role.
- **The path constraint** needs only membership. If it supports projection,
  every tree extension is projected onto it; otherwise an extension that
  leaves it is rejected. Every intermediate configuration on every edge is
  checked for membership and validity.
- **The validator** (state validity) is applied to every root and, through
  the default motion validator, to every configuration along every edge.
  State validity and motion validity are distinct interfaces:
  `CollisionChecker.is_valid(q)` for one configuration, `MotionValidator`
  for the motion between two.
- **Edges are the single local-motion boundary,** and it is explicit:
  `PlanningProblem.motion_validator` validates the local motion of every
  tree edge, the final connection between trees, and every shortcut, and
  returns the configurations to store (`LocalMotion`). The default,
  `DiscreteMotionValidator`, samples the straight segment every
  `edge_resolution` and requires every sample including the endpoint to be
  admissible, so validity is discrete, not continuous; choose the
  resolution against the thinnest obstacle you must not miss. A backend may
  supply continuous collision checking or a swept-volume check instead.
  A custom validator **replaces** the default and owns the validity of the
  complete motion, interior included; the planner does not run the default
  checks alongside it. Two guarantees hold regardless of validator: every
  configuration a validator returns is checked for admissibility before it
  is stored (which says nothing about the motion between nodes), and a
  `LocalMotion` must satisfy its contract or `MotionContractError` is
  raised. To add a restriction while keeping the default discrete checks,
  wrap `planner.default_motion_validator(problem)` in
  `RestrictedMotionValidator(base, accepts)`. Because the search is
  bidirectional, goal-tree edges are validated with the goal side as
  `q_from`, the reverse of execution; motion validity must not depend on
  direction unless that is acceptable.
- **Cancellation has three polling points.** `abort_fn` is checked once
  per search iteration (before the deadline), before each sampling draw
  during root collection, and before each smoothing attempt. Firing during
  roots returns an Aborted result with zero iterations and trees holding
  the roots gathered so far; during the search it returns Aborted; during
  smoothing it stops smoothing and returns the path found, as smoothed so
  far, as a success, because a valid path exists. Cancellation is
  cooperative: a set, validator, or IK call that runs long is not
  interrupted. The deadline is checked only in the search loop.
- **Reached means connected.** Growth reports success only once the tree
  contains the exact target, added through a validated edge. Coming within
  `connection_tolerance` triggers that final exact edge; it never substitutes
  for it. Where the two trees meet, the connecting tree holds the other
  tree's configuration exactly and the join has no unchecked gap.
- **Shortcuts are validated edges that end at their target.** Smoothing
  accepts a shortcut only if its joint-space path length under the problem's
  space is shorter than the segment it replaces, as in the original CBiRRT;
  fewer waypoints is not the criterion. The first and last waypoints of a
  path are preserved exactly, except as the next rule says.
- **Smoothing for execution is separate and opt-in (#207).** With
  `smooth=...`, a found path's corners are replaced by quintic blends with
  continuous curvature, reported as `PlanResult.smooth_path`; `path` stays
  the polyline. Each blend is accepted only if samples along it, at most
  `edge_resolution` apart, are admissible and the problem's motion validator
  accepts every chord between consecutive samples: the edge guarantee above,
  no stronger. Failed blends shrink; a corner with no admissible blend stays
  a stop, between two segments that each start and end at rest. Equality path
  constraints get no projection of blend samples (it would break
  smoothness), so their corners generally stay stops. Smoothing is
  deterministic and changes nothing unless asked; timing is left to a
  retimer.
- **Output representation on continuous joints.** A returned path is unwrapped
  forward from its first waypoint, so on a continuous joint each consecutive
  raw difference is the short way around and never exceeds one step. The
  first waypoint is the start as given; the last waypoint is the goal as a
  configuration but may differ from the value given by a multiple of 2π.
  Paths without continuous joints are unaffected. This is what lets an
  executor that interpolates raw joint values follow the path (#77).

## Tolerances

Each has one meaning:

| `CBiRRTConfig` field | meaning |
|---|---|
| `membership_tolerance` | a configuration is in a TSR-induced set if its TSR distance is within this; projection is done when within it |
| `connection_tolerance` | tree growth counts as reaching its target within this joint-space distance |
| `edge_resolution` | spacing of validity checks along an edge, a joint-space distance; default 0.05 rad, independent of `step_size` since 3.4 (#204); `None` means `step_size` |
| `progress_tolerance` | tree growth stops when the distance to target shrinks by less than this |
| `projection_progress_tolerance` | projection gives up when the violation shrinks by less than this |

Membership tolerance belongs to the concrete set; the legacy lowering copies
the config value into the sets it builds. `tsr_tolerance` is a deprecated
alias that sets membership and connection tolerances together.

`CBiRRTConfig` validates its ranges at construction and raises `ValueError`
naming the field: `timeout`, `step_size`, `progress_tolerance`, and
`projection_progress_tolerance` positive (a zero progress tolerance can loop
forever under a projector that stalls); `membership_tolerance` and
`connection_tolerance` nonnegative (zero means exact); `edge_resolution`
None or positive; `max_iterations`, `sample_draws`, `num_tree_roots`,
and `max_projection_iters` at least 1; `max_per_draw` None or at least 1;
`smoothing_iterations` and `smoothing_patience` nonnegative; `extend_steps`
and `connect_steps` None or at least 1; `start_sample_probability` and
`goal_sample_probability` within $[0, 1]$. The native `PlannerConfig` applies the same ranges.

## The reference artifact

`tools/reference_artifact.py` runs a fixed matrix of problems on the planar
reference arm (`sscbirrt.testing`) with fixed seeds and writes
`tests/reference/python_reference.json`: per case, the problem as data, the
seed, the implementation versions, the result (status, failure category,
provenance, iterations, path), and an **independent** validation report
computed from the problem alone: every waypoint in the space, the first in
the declared start set, the last in the declared goal set, every waypoint
admissible, every consecutive pair validated at `edge_resolution` along
the space's direction, and raw steps within one step size (the continuous-joint
output rule above). The matrix covers fixed endpoints, multiple roots, a
nested finite goal, a wrapped joint across the seam, rejection-only and
projected path constraints, a TSR union goal, a TSR chain goal, an `AllOf`
constraint with the named strategy, timeout, cancellation, and an
unreachable problem.

The semantic oracle is the status, category, provenance, and validation
report; endpoints are compared as configurations under the space metric,
never as arrays. Paths and iteration counts are recorded for inspection and
not compared, so the artifact is stable across platforms while behavior is
pinned. `tests/test_reference_artifact.py` regenerates it and fails on any
semantic change; `--check` also reports whether the regeneration is
bit-for-bit on the same versions. This artifact is the definition of
"compatible with the Python reference" for the native backends, and the tool
runs it under each selection: `--backend python` regenerates it,
`--backend native` checks parity on every case the core supports, and
`--backend auto` checks the default selection, which must pick the native
core wherever it can and Python only with a stated reason.

## The TSR instantiation

`plan(start, goal, goal_tsrs, start_tsrs, constraint_tsrs)` lowers its
arguments into a `PlanningProblem` (`sscbirrt.legacy`) and reproduces the
semantics the arguments always had:

- fixed configurations become a `FiniteSet` whose members are always roots
  and are never sampled;
- a list of start or goal TSRs becomes an `AnyOf` of `TSRConfigurationSet`s
  with weights proportional to TSR volume;
- a list of constraint TSRs becomes an `AllOf` with `MostViolatedProjection`,
  or the bare set when there is one TSR;
- `start_index` and `goal_index` are the last element of the root's
  provenance, which the lowering arranges to be the index into the config
  list or the TSR list.

`TSRConfigurationSet` samples poses from the TSR bounds with the planner's
random generator, solves IK, and returns every solution inside the joint
limits as candidates. Completeness of that candidate set is the IK
backend's job: on a joint whose range exceeds one turn, the same geometric
branch has several in-limit windings, and an enumerative backend such as
SSIK returns all of them (#36, #63). The set never filters by collision. Its projection moves the pose to the closest point of
the TSR, composed with the TSR's `T0_w` and `Tw_e` frames, and takes the IK
solution nearest the current configuration under the space metric.

## TSR chains

A TSR chain couples several TSRs through serial composition: each link's
pose is sampled relative to the previous one, and the chain as a whole
defines **one** set of end-effector poses (a door handle on a swinging
door). `TSRConfigurationSet` accepts a `tsr.TSRChain` exactly as it accepts
a `tsr.TSR`: both are pose regions with a distance, a closest world-frame
transform, and a seeded sampler (`PoseRegion`), so a chain is one state
set, one alternative in a union, one factor in an intersection, and one
entry in the legacy `goal_tsrs` / `start_tsrs` / `constraint_tsrs` lists.
It is **not** an `AllOf` of its constituent TSRs: those constrain different
frames, and their world-frame intersection is not the chain's set; a test
shows a chain member lying in neither component.

Two properties follow from the chain's numerical inverse: for two or more
links, distance is the best residual a bounded solve found (an upper
bound), so membership can be a false negative on a hard chain, and each
membership test or projection step costs a solve. In Python that is a few
milliseconds, which a path constraint pays on every edge sample. The native
backend runs sstsr's C++ chain (sstsr 3.3, #184), whose inverse is a
different solver held to the same properties, so a chain plans natively
like a single TSR.

Independent TSRs on different robot links (an end-effector region and an
elbow clearance region) are a different object: they compose as `AllOf` of
per-link sets with a joint projection strategy and need forward kinematics
per link. That is not implemented.
