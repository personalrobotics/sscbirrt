# Native design: the SSCBiRRT C++20 contract

> **Reading this for what is supported today?** This is a design record built
> up release by release: each section states the scope *of its release*, and
> later sections extend earlier ones. Today the native backend also plans
> TSRs and TSR chains (sstsr's C++, the last section), runs SSIK for the UR
> family and checks collisions in MuJoCo. The README's
> [Backends](../README.md#backends-native-and-python) section is the current
> summary.

This document is the normative boundary for the native implementation of
sscbirrt (#82). It translates the Python design in [design.md](design.md)
into C++20 types without changing its semantics. The two backends are one
contract with two implementations; the only recorded difference is the
random-number engine, see [Concept map](#concept-map). Where the Python
reference needed a rule stated more sharply to make that true (finite
limits, config ranges, cancellation points, the free-space sampler), the
reference was changed first (#107, #108, #109, #110) and this document
describes it as it now behaves. Implementation begins only after this document
is reviewed and merged; #85 implements it, #92 gates it against the
reference artifact.

The reference for "what the planner does" is the Python implementation at
v1.4.0 and the artifact `tests/reference/python_reference.json`. This
document does not restate every rule in design.md; it says how each rule is
carried into C++ and what the C++ types promise.

## Scope of v1.5.0

The native core plans non-TSR problems: joint spaces with bounded and
angular joints, finite sets, `AnyOf` and `AllOf` of finite and native sets,
native state validators, native path constraints, a replaceable free-space
sampler, deadlines, cancellation, provenance, and the full search (roots, bidirectional growth, exact
connection, complete-edge validation, extraction, shortcutting, unwrapping).
The Python planner is unchanged and remains the reference and the fallback.

Not in this release: TSR-induced sets, IK, MuJoCo, any Python callback
inside the native solve, parallel search, a dynamic plugin ABI. Extensibility
is a source-level C++ API: users subclass the interfaces below and link
against `sscbirrt::core`.

## Dependency boundary

`sscbirrt::core` depends on the C++20 standard library and nothing else. It
does not depend on Python, pybind11, numpy, Eigen, sstsr, SSIK, MuJoCo,
OMPL, RoboPlan, or `mj_manipulator`. Configurations are `std::vector<double>`
and views are `std::span<const double>`; a consumer that uses Eigen converts
at the boundary. The reason is the standalone consumer criterion in #85: a
C++ project must be able to `find_package(sscbirrt)` and build with no
transitive dependency to resolve. Eigen may be adopted inside later targets
(pose regions in v1.6.0) without touching the core.

```
sscbirrt._native  (pybind11)   ──►  sscbirrt::core  ◄──  sscbirrt_tests, sscbirrt_consumer
                                          ▲
                        (v1.6.0) sscbirrt::tsr ─┘   (v1.7.0) sscbirrt::mujoco ─┘
```

Arrows point from dependent to dependency. Nothing points out of `core`.

## Vocabulary

| Name | Definition |
|---|---|
| `Config` | `std::vector<double>` of length `dof`; owned. |
| `ConfigView` | `std::span<const double>`; borrowed, valid for the call. |
| `Provenance` | `std::vector<int>`; the sequence of choices that produced a sample, outermost first. Empty for a leaf. |
| `Rng` | `std::mt19937_64`; owned by the solver, passed by reference to sets. |
| `Metric` | `std::function<double(ConfigView, ConfigView)>`; a set's notion of distance between configurations. |

## Joint space

```cpp
namespace sscbirrt {

class SpaceSampler {
public:
  virtual ~SpaceSampler() = default;
  virtual Config sample(Rng& rng) const = 0;            // a free-space target of length dof
};

class JointSpace final : public SpaceSampler {
public:
  JointSpace(std::vector<double> lower, std::vector<double> upper,
             std::vector<bool> angular = {});           // throws std::invalid_argument

  int dof() const;
  const std::vector<double>& lower() const;
  const std::vector<double>& upper() const;
  const std::vector<bool>& angular() const;             // all false if none

  bool contains(ConfigView q) const;                    // shape, finiteness, limits
  std::optional<std::string> why_invalid(ConfigView q) const;
  Config direction(ConfigView from, ConfigView to) const;   // short way around angular joints
  double distance(ConfigView a, ConfigView b) const;        // Euclidean norm of direction
  Config interpolate(ConfigView from, ConfigView to, double t) const;
  Config sample(Rng& rng) const override;                   // uniform; one full turn on angular joints
  std::vector<Config> unwrap_path(const std::vector<Config>& path) const;
};

}  // namespace sscbirrt
```

Contracts, identical to Python:

- Construction throws `std::invalid_argument` if `lower` and `upper` differ
  in length, any `lower[i] > upper[i]`, `angular` is nonempty with a length
  other than `dof`, or a joint **not** marked angular has a non-finite
  limit. The message for the last case is Python's: "joint i has non-finite
  limits [lo, hi]; give finite planning limits or mark it angular". Topology
  is the caller's declaration and is never inferred from the limits (#107).
- A joint is angular only if the caller says so. Its stored limits are
  ignored: it passes the limit check for any finite value, and its
  `direction` component is wrapped into $(-\pi, \pi]$ by `atan2(sin, cos)`.
  A bounded joint's component is the plain difference. Bounded joints wider
  than one turn stay bounded.
- `contains` is `why_invalid(q) == nullopt`. `why_invalid` reports, in
  order: wrong length, non-finite entry, out-of-limit joints (bounded only).
- `sample` draws each bounded joint uniformly in its limits and each angular
  joint uniformly in $[-\pi, \pi)$, whatever limits were stored for it.
  `JointSpace` is the default `SpaceSampler` of a problem.
- Uniform draws are `(rng() >> 11) * 2^-53`, not `std::uniform_real_distribution`,
  so a seeded solve is repeatable across standard libraries as well as builds.
- `unwrap_path` reproduces design.md's output rule: the first waypoint is
  returned as given, each later waypoint is the previous one plus
  `direction(prev, next)` on angular joints and the given value on bounded
  joints.

## Sets and capabilities

Membership is the only requirement. Capabilities are separate interfaces
and a set reports the ones it has through accessors that return a pointer
or `nullptr`; this is the C++ form of `supports(s, Capability)`.

```cpp
namespace sscbirrt {

struct Sample {
  Config q;
  Provenance source;
};

class SetSampler {
public:
  virtual ~SetSampler() = default;
  virtual std::vector<Sample> sample(Rng& rng) const = 0;  // candidates of one draw, unfiltered
};

class SetDistance {
public:
  virtual ~SetDistance() = default;
  virtual double distance(ConfigView q) const = 0;         // nonnegative, geometric, pre-tolerance
};

class SetViolation {
public:
  virtual ~SetViolation() = default;
  virtual double violation(ConfigView q) const = 0;        // nonnegative; zero iff contains(q)
};

class SetProjector {
public:
  virtual ~SetProjector() = default;
  virtual std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const = 0;
};

class StateSet {
public:
  virtual ~StateSet() = default;
  virtual bool contains(ConfigView q) const = 0;

  virtual const SetSampler*   sampler()   const { return nullptr; }
  virtual const SetDistance*  distancer() const { return nullptr; }
  virtual const SetViolation* violator()  const { return nullptr; }
  virtual const SetProjector* projector() const { return nullptr; }

  // Why a capability is absent, for diagnostics. Empty if present.
  virtual std::string why_unsupported(Capability c) const;

  // Enumeration, mirroring is_finite / members / seeds.
  virtual bool is_finite() const { return false; }
  virtual std::vector<Sample> members() const { return {}; }   // exhaustive iff is_finite()
  virtual std::vector<Sample> seeds() const { return {}; }     // explicit configs anywhere inside

  virtual std::string describe() const = 0;                    // for messages; Python's repr
};

enum class Capability { Sampler, Distance, Violation, Projector };

}  // namespace sscbirrt
```

A concrete set that has a capability inherits the interface and returns
`this` from the accessor. The contracts on each method are those of
design.md's Capabilities section, restated as obligations on implementers:

- `sample` returns every candidate a draw produces and applies no external
  filter; the planner validates. It may return an empty vector.
- `distance` and `violation` are nonnegative. `violation(q) == 0.0` exactly
  when `contains(q)`.
- `project` may return `nullopt` to give up. It may use `q_previous` to seed
  or to reject a result that moved too far. The planner re-checks any
  returned configuration against the space and admissibility before use.
- All methods are `const` and must be safe to call repeatedly from one
  thread. v1.5.0 never calls a set from two threads.

Lifetime and ownership: sets are held through `std::shared_ptr<const
StateSet>`. Composites own their children the same way. The planner borrows
the problem for the duration of `solve` and stores nothing that outlives it
except the returned result, which copies configurations. A set that
captures external state (a scene, a model) is responsible for keeping that
state alive; the planner never takes ownership of it.

### Leaves

```cpp
class FiniteSet final : public StateSet, public SetSampler, public SetDistance, public SetViolation {
public:
  FiniteSet(std::vector<Config> members, double tolerance = 1e-6, Metric metric = euclidean);
  // contains: metric(q, m) <= tolerance for some member m
  // distance: min over members; violation: max(0, distance - tolerance)
  // sample: one member drawn uniformly, source {index}  (as Python's FiniteSet.sample)
  // is_finite: true; members(), seeds(): every member with source {index}
};

class EmptySet final : public StateSet {   // contains: false; is_finite: true; members: {}
};

class PredicateSet final : public StateSet {
public:
  PredicateSet(std::function<bool(ConfigView)> predicate, std::string name = "");
};
```

`FiniteSet` throws `std::invalid_argument` on an empty member list, as
Python's does ("FiniteSet requires at least one configuration"), or members
of unequal length. Members are validated against the problem's space at
`solve`, not at construction, because a set does not know its space.

### Composites

```cpp
class AnyOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
public:
  AnyOf(std::vector<std::shared_ptr<const StateSet>> children,
        std::optional<std::vector<double>> weights = std::nullopt,
        Metric metric = euclidean);
};

class IntersectionProjection {  // strategy interface
public:
  virtual ~IntersectionProjection() = default;
  virtual std::optional<Config> project(const std::vector<std::shared_ptr<const StateSet>>& children,
                                        ConfigView q_previous, ConfigView q_proposed) const = 0;
};

class IntersectionSampling {
public:
  virtual ~IntersectionSampling() = default;
  virtual std::vector<Sample> sample(const std::vector<std::shared_ptr<const StateSet>>& children, Rng& rng) const = 0;
};

class AllOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
public:
  AllOf(std::vector<std::shared_ptr<const StateSet>> children,
        std::shared_ptr<const IntersectionProjection> projection = nullptr,
        std::shared_ptr<const IntersectionSampling> sampling = nullptr);
};

class MostViolatedProjection final : public IntersectionProjection {
public:
  MostViolatedProjection(int max_iters = 50, double progress_tolerance = 1e-6);
};

class RejectionSampling final : public IntersectionSampling {
public:
  explicit RejectionSampling(int source_child);
};
```

Capabilities of a composite are computed at construction and are exactly
design.md's table. The accessors return `nullptr` for an absent capability
even though the class inherits the interface, so a caller must query the
accessor and never `dynamic_cast`. Specifically:

- One child: every accessor delegates to the child.
- `AnyOf`, several children: `distancer` and `violator` present iff every
  child has them (min over children); `sampler` present iff `weights` is
  given and every child samples (draw a child by weight, prepend its index
  to each candidate's `source`); `projector` present iff every child
  projects (project onto each, return the successful result nearest
  `q_proposed` under `metric`).
- `AllOf`, several children: `distancer` and `violator` present iff every
  child has them (max over children); `sampler` present iff a `sampling`
  strategy is given; `projector` present iff a `projection` strategy is
  given. The strategy is responsible for its own requirements: `AllOf`
  calls `projection->requires(children)` at construction, and
  `MostViolatedProjection::check_requirements` throws `UnsupportedCapability` naming
  the first child that lacks `violator` or `projector`. This is Python's
  `requires` hook at the same point.
- `is_finite`: `AnyOf` iff every child is; `AllOf` iff some child is.
  `members`: union of children's members with the child index prepended;
  intersection enumerates one finite child and keeps what the others
  contain. `seeds`: the explicit configurations anywhere in the expression,
  with full provenance, regardless of finiteness.
- Construction throws `std::invalid_argument` for an empty child list,
  weights of the wrong length, negative weights, or weights summing to zero.

`MostViolatedProjection` reproduces the Python rule exactly: repeatedly
project onto the unsatisfied child with the largest violation; progress is
the lexicographic decrease of the descending-sorted violation profile by
more than `progress_tolerance`; give up after a sweep without progress, when
a projector returns `nullopt` or leaves the point unchanged, or at
`max_iters`; succeed when every child contains the point. A satisfied child
is never selected.

### Provenance

`Provenance` semantics are Python's: `AnyOf` prepends the index of the child
it chose, `FiniteSet` contributes the index of the member, other leaves
contribute nothing. `PlanResult::start_source` and `goal_source` are the
provenance of the roots the path connects. The legacy `start_index` and
`goal_index` are the last component, or 0 if empty, computed by the binding.

## Validity and motion

```cpp
class StateValidator {
public:
  virtual ~StateValidator() = default;
  virtual bool is_valid(ConfigView q) const = 0;
};

class AcceptAll final : public StateValidator { /* true */ };

class JointBoxObstacles final : public StateValidator {
  // Invalid inside any listed axis-aligned box in joint space. The native
  // counterpart of sscbirrt.testing.Wall, so the artifact matrix runs natively.
public:
  struct Box { std::vector<double> lo, hi; };   // open on both sides, as Wall is
  explicit JointBoxObstacles(std::vector<Box> boxes);
};

struct LocalMotion {
  std::vector<Config> configs;   // validated configurations after q_from, in order
  bool reached = false;          // whole motion valid and configs.back() == q_to exactly
};

class MotionValidator {
public:
  virtual ~MotionValidator() = default;
  virtual LocalMotion validate(ConfigView q_from, ConfigView q_to) const = 0;
};

class DiscreteMotionValidator final : public MotionValidator {
public:
  DiscreteMotionValidator(const JointSpace& space, std::function<bool(ConfigView)> is_admissible, double resolution);
  // n = max(1, ceil(distance / resolution)); samples q_from + (i/n) * direction for i = 1..n;
  // sample n is q_to exactly; stops at the first inadmissible sample with reached = false.
};

class RestrictedMotionValidator final : public MotionValidator {
public:
  RestrictedMotionValidator(std::shared_ptr<const MotionValidator> base,
                            std::function<bool(ConfigView, ConfigView)> accepts);
};
```

`LocalMotion` carries Python's contract: with `reached == true` on a nonzero
motion, `configs` is nonempty and `configs.back()` equals `q_to` exactly
(bitwise, as `np.array_equal`). The planner checks this before touching a
tree and throws `ContractError` on violation. A custom validator replaces
the default and owns the interior of the motion; the planner still checks
every returned configuration for admissibility, rejects a `reached` motion
whole if any is inadmissible, and keeps the admissible prefix of an
unreached one.

## The problem

```cpp
struct PlanningProblem {
  std::shared_ptr<const JointSpace>      space;
  std::shared_ptr<const StateSet>        start;
  std::shared_ptr<const StateSet>        goal;
  std::shared_ptr<const StateValidator>  validator;
  std::shared_ptr<const StateSet>        path_constraint;   // may be null
  std::shared_ptr<const MotionValidator> motion_validator;  // null means the default
  std::shared_ptr<const SpaceSampler>    sampler;           // null means *space
};
```

Roles are Python's, and so are the two replaceable strategy components:
the motion validator and the free-space sampler (#110). The sampler
proposes the targets the trees grow toward; start and goal bias remain the
planner's and mix the role sets' own samplers with it, and the sampler is
not consulted for roots or bias draws. A target outside the space is
handled by not growing toward it. Replacing the default trades away
probabilistic completeness unless the replacement has full support over
the space; that is the caller's responsibility. The problem is a value type; copying it shares the
components.

## Planner configuration

```cpp
struct PlannerConfig {
  // Termination
  double timeout_seconds = 30.0;
  int    max_iterations  = 100000;

  // Tolerances
  double connection_tolerance          = 1e-3;
  std::optional<double> edge_resolution = 0.05;  // nullopt means step_size (the default until #204)
  double progress_tolerance            = 1e-6;

  // Growth
  double step_size  = 0.1;
  double start_sample_probability = 0.1;  // goal_bias/start_bias until #196
  double goal_sample_probability  = 0.1;
  std::optional<int> extend_steps;            // nullopt: connect until blocked
  std::optional<int> connect_steps;

  // Roots
  int sample_draws     = 100;   // Python: tsr_samples, the draw budget per role
  int num_tree_roots   = 100;
  int max_per_draw     = 3;     // Python: max_ik_per_pose

  // Smoothing
  bool smooth_path         = true;
  int  smoothing_iterations = 50;
  int  smoothing_patience   = 15;
};
```

Three Python fields are absent on purpose. `membership_tolerance`,
`max_projection_iters`, and `projection_progress_tolerance` belong to the
concrete sets and strategies (design.md: "membership tolerance belongs to
the concrete set"); the legacy lowering copies them into the sets it builds,
and the binding does the same. `angular_joints` lives on `JointSpace` only.
`abort_fn` is replaced by a cancellation token. `tsr_tolerance` is a
deprecated alias with no native form.

Ingress validation of the config, at `solve`, throws `std::invalid_argument`
with Python's message shape ("<field> must be <requirement>, got <value>")
and Python's ranges (#108): positive `timeout_seconds`, `step_size`,
`progress_tolerance`; nonnegative `connection_tolerance`,
`smoothing_iterations`, `smoothing_patience`; at least 1 for
`max_iterations`, `sample_draws`, `num_tree_roots`, `max_per_draw`;
`edge_resolution` empty or positive; `extend_steps` and `connect_steps`
empty or at least 1; `start_sample_probability` and
`goal_sample_probability` (`goal_bias`/`start_bias` until #196) within $[0, 1]$. The
ranges Python checks on its set-owned fields (`membership_tolerance`
nonnegative, `max_projection_iters` at least 1,
`projection_progress_tolerance` positive) are enforced natively by the
constructors that own those values, `FiniteSet`, the TSR set in v1.6.0, and
`MostViolatedProjection`.

## Deadlines, cancellation, randomness

```cpp
class CancellationToken {
public:
  void cancel() noexcept;            // any thread
  bool cancelled() const noexcept;   // std::atomic<bool>, relaxed
};

struct SolveOptions {
  std::optional<std::uint64_t> seed;                    // nullopt: nondeterministic seed
  std::shared_ptr<const CancellationToken> cancel;      // may be null
};
```

- The solver owns one `std::mt19937_64` per `solve`, seeded from `seed`.
  Every random decision in the solve draws from it, in a fixed order, and
  every set receives it by reference. With the same seed, the same problem,
  and the same build, a single-threaded solve is repeatable. Python uses
  numpy's PCG64, so a native path is never expected to equal a Python path;
  the artifact compares semantics (#92), not waypoints.
- The deadline is `std::chrono::steady_clock::now() + timeout_seconds`,
  taken when the search loop starts (after roots, as Python does). It is
  checked once per iteration.
- The token is checked at Python's three points (#109): once per search
  iteration before the deadline, before each sampling draw during root
  collection, and before each smoothing attempt. The outcomes are Python's:
  during roots, `Status::Aborted` with zero iterations, `reason` "Aborted by
  user during <role> root collection", and trees holding the roots gathered
  so far; during the search, `Status::Aborted`; during smoothing, smoothing
  stops and the result is `Status::Success` with the path as smoothed so
  far, because a valid path exists. Finite sets involve no draws and are
  not polled during roots. Cancellation is cooperative: a set or validator
  that runs long is not interrupted.
- The solver does not spawn threads and calls no set from more than one
  thread.

## Result and statuses

```cpp
enum class Status {
  Success,
  Timeout,
  Aborted,
  MaxIterations,
};

struct RootReport {                 // per role; carried by the result and by NoRoots
  int explicit_candidates = 0;      // from seeds()
  int explicit_rejected   = 0;
  int draws               = 0;
  int draws_empty         = 0;      // "IK unreachable" in Python's summary
  int outside_space       = 0;
  int in_collision        = 0;
  int constraint_violated = 0;
  int roots               = 0;
  bool only_collisions() const;     // every rejection was the validator's
  std::vector<std::string> details; // per explicit rejection, as Python logs them
};

struct PlanResult {
  Status status;
  std::string reason;                  // human-readable; empty on success
  std::vector<Config> path;            // empty unless Success; unwrapped per JointSpace rule
  Provenance start_source, goal_source;
  int iterations = 0;
  double planning_seconds = 0.0;
  std::pair<int, int> tree_sizes{0, 0};
  RootReport start_roots, goal_roots;
  std::shared_ptr<const Tree> tree_start, tree_goal;   // for inspection; may be null if not requested
};
```

Ordinary search outcomes are statuses. Exceptions are reserved for:

| Exception | When |
|---|---|
| `std::invalid_argument` | malformed input at ingress: dimensions, non-finite values, bounds, tolerances, option combinations, a set whose members do not match the space's `dof` |
| `sscbirrt::UnsupportedCapability` | a start or goal set that is neither finite nor sampleable; a composite asked for a capability it lacks; a strategy whose children lack what it needs |
| `sscbirrt::NoRoots` | no admissible root for a role; carries the role and its `RootReport` |
| `sscbirrt::ContractError` | a `MotionValidator` violated the `LocalMotion` contract; a set or projector returned a configuration of the wrong length |

No admissible root is an exception in both backends, as Python has it:
whether a role has any root is a property of the problem the planner
discovers before the search starts, and callers distinguish it from a
search that ran and failed. The binding maps `NoRoots` to Python's four
exceptions: `AllStartConfigurationsInCollision` when
`report.only_collisions()`, otherwise `AllStartConfigurationsInvalid`, and
likewise for the goal, with the report's details as the message. A role
with no explicit members and nothing to draw from (an `EmptySet`) is
`std::invalid_argument`, Python's `ValueError` "No valid <role>
configurations available". Whether these become statuses is a 2.0 decision
to be made for both backends at once.

`reason` strings begin with the same prefixes the artifact's
`failure_category` recognizes: `Timeout`, `Aborted`, `Max iterations`.

## The search, as obligations

The native search is Python's, function for function. This list is the
contract; `planner.py` is the reference for anything it leaves open.

1. **Ingress.** Validate the config. For each of start and goal, require
   `is_finite() || sampler() != nullptr`, else throw `UnsupportedCapability`.
   Check every explicit member (`seeds`) has length `dof`.
2. **Admissibility** of a configuration is, in order: `space.contains`,
   `validator.is_valid`, `path_constraint.contains` (if any). The order is
   observable through `RootReport` and must be kept.
3. **Roots** for a role: every `seeds()` candidate that is admissible, in
   order, with its provenance. Then, if the set is not finite and samples,
   draw up to `sample_draws` times or until `num_tree_roots` roots exist,
   keeping at most `max_per_draw` admissible candidates per draw, skipping
   a candidate whose provenance equals an explicit seed's. A draw with more
   candidates than `max_per_draw` is first put in a random order
   (Fisher-Yates on `index()`), as the reference does (#168). Rejections are
   counted in the `RootReport`. No roots for a role throws `NoRoots`.
   Before each draw the cancellation token is checked; if set, the solve
   returns `Status::Aborted` with the roots gathered so far.
4. **Iteration** `i` extends the start tree if `i` is even, else the goal
   tree. Cancellation is checked, then the deadline. *(Superseded by #196,
   see the last section: a turn now either adds roots to its own tree or
   extends toward a free-space sample.)* The target is a sample
   from the opposite role's set with probability `goal_bias` or
   `start_bias` (only if that set samples; up to `sample_draws` attempts to
   find an admissible one), otherwise one draw from `problem.sampler`, or
   from `space` when the sampler is null.
5. **Growth** toward a target from the nearest node under `space.distance`
   (ties to the lowest index): if the target is outside the space, no
   growth. Otherwise repeat: if the remaining distance is exactly zero,
   reached without adding a node; if it is below `connection_tolerance`,
   extend along the exact final edge to the target and return its outcome;
   if the distance shrank by less than `progress_tolerance`, stop; if the
   step budget (`extend_steps` or `connect_steps`) is spent, stop; take a
   step of at most `step_size` along `direction`; if the step leaves the
   space, stop; if the constraint has a projector, project from the current
   node and stop on `nullopt` or a result outside the space; extend along
   the edge to the new configuration and stop if not reached.
6. **Edges** are the single local-motion boundary: growth, the final
   connection, and shortcuts all go through `motion_validator.validate`
   (default: `DiscreteMotionValidator` at `edge_resolution` or `step_size`
   over the admissibility predicate). Zero-length motions succeed without a
   node. Returned configurations are stored as consecutive nodes after the
   contract and admissibility checks above.
7. **Connection.** After extending tree A to a node, grow tree B toward that
   node's configuration with the connect budget. Connected means tree B now
   holds that exact configuration through a validated edge.
8. **Extraction** joins the start-tree path to the meeting node with the
   reversed goal-tree path, dropping the duplicated join configuration.
9. **Smoothing**, if enabled: up to `smoothing_iterations` attempts, each
   choosing `i` uniformly in `[0, n-3]` and `j` uniformly in `[i+2, n-1]`,
   growing a fresh single-root tree from `path[i]` toward `path[j]` with no
   step budget; replace the segment only if the shortcut's path length under
   `space.distance` is smaller by more than `1e-9`. Stop after
   `smoothing_patience` attempts without improvement, when the path has two
   waypoints, or when the cancellation token is set before an attempt; in
   the last case the result is still `Success` with the path so far.
10. **Output.** `space.unwrap_path` on the final path. `start_source` and
    `goal_source` are the provenance of the roots the meeting node descends
    from in each tree.

The `Tree` type is `struct Node { Config q; int parent; Provenance source; }`
in a vector, with `nearest` a linear scan. A spatial index is permitted
later only if it returns the same node as the linear scan, ties included.

## Python binding

The extension module is `sscbirrt._native`, built with pybind11 as ssik's
is. It exposes the types above under the same names and one function:

```python
sscbirrt._native.Planner(config: _native.PlannerConfig).solve(
    problem: _native.PlanningProblem, seed: int | None, cancel: _native.CancellationToken | None,
    keep_trees: bool = True) -> _native.PlanResult
```

The GIL is released for the duration of `solve`. After entry, the native
solve touches no Python object; that is why every component must be a
native object. The public Python surface in v1.5.0 is:

```python
from sscbirrt.backends.native import lower, NativeUnsupported

lowered = lower(problem, config)       # PlanningProblem + CBiRRTConfig -> native problem, or raises NativeUnsupported
planner = CBiRRT(robot, ik, collision, config, backend="native")   # "python" | "native" | "auto" (default since 2.0; "python" in 1.x)
```

A note for the parity gate (#92): provenance is a property of the draws
when a role has more than one admissible root, so the two backends compare
it only where the reached root is unique; otherwise the check is that the
reported provenance names one of the admissible roots and matches the
path's endpoint. The artifact's `nested_finite_goal` case is the example:
Python's seed reaches the near member, a native seed may reach the far one,
and both are correct.

`lower` walks the Python problem and maps each component:

| Python | Native | Otherwise |
|---|---|---|
| `JointSpace` | `JointSpace` | |
| `FiniteSet`, `EmptySet` | same | |
| `AnyOf`, `AllOf` with `MostViolatedProjection` / `RejectionSampling` | same, children lowered recursively | |
| `PredicateSet`, `TSRConfigurationSet`, any other set | | `NativeUnsupported("goal: TSRConfigurationSet has no native form")` (v1.5.0; v1.6.0 lowers a plain TSR with SSIK) |
| validator: `sscbirrt.testing.NoCollision`, `Wall` | `AcceptAll`, `JointBoxObstacles` | any other validator: `NativeUnsupported("validator: <type> is a Python object; native needs a sscbirrt.StateValidator")` |
| `motion_validator` None | default | any custom validator: `NativeUnsupported` |
| `sampler` None | default (`space`) | any custom sampler: `NativeUnsupported("sampler: <type> is a Python object")` |
| `CBiRRTConfig` | `PlannerConfig` (field map above), `abort_fn` wrapped in a token polled from a Python thread | |

`NativeUnsupported` carries the list of every component that blocked
lowering, not only the first. With `backend="auto"`, the planner catches it,
selects the Python backend, records the reasons on
`PlanResult.backend_reasons`, and sets `PlanResult.backend = "python"`; with
`backend="native"` it propagates; with `"python"` no lowering is attempted.
The default was `"python"` through 1.x, so `plan(...)` was unchanged by the
native core's arrival; 2.0 made `"auto"` the default (#86), and the planner
also logs the reasons at INFO when it selects Python, so a caller of the
legacy `plan(...)` that never sees a `PlanResult` still has a diagnostic.
`tools/reference_artifact.py --backend auto --check` runs the artifact under
the default and requires Python to have been chosen only with a stated reason.

Native results are converted to Python `PlanResult`s: `Status::Success` and
the three search failures map to `success` and `failure_reason` with the
same reason prefixes; `NoRoots` becomes the Python exception described
above; `std::invalid_argument` becomes `ValueError`, `UnsupportedCapability`
and `ContractError` their Python namesakes; trees are wrapped read-only. `sscbirrt` imports and works without
the extension present; `backend="native"` then raises `NativeUnsupported`
naming the missing module.

## CMake targets and packaging

```
cpp/
  CMakeLists.txt                 project(sscbirrt LANGUAGES CXX), C++20
  include/sscbirrt/*.hpp         public headers
  src/*.cpp                      core implementation
  bindings/sscbirrt_native.cpp   pybind11 module (only target that sees Python)
  tests/*.cpp                    ctest targets
  examples/consumer/             standalone find_package consumer, built against the installed package
```

| Target | Kind | Depends on | Installed |
|---|---|---|---|
| `sscbirrt::core` | static library (`BUILD_SHARED_LIBS` respected) | C++20 standard library | yes, with headers and `sscbirrtConfig.cmake` |
| `sscbirrt_tests` | executables under ctest | `sscbirrt::core` | no |
| `sscbirrt_consumer` | executable, `examples/consumer/` | `find_package(sscbirrt)` | no |
| `sscbirrt_native` | pybind11 module `sscbirrt._native` | `sscbirrt::core`, pybind11, Python | into the wheel |

Options: `SSCBIRRT_BUILD_TESTS` (default ON in-tree), `SSCBIRRT_BUILD_PYTHON`
(default OFF; the wheel build turns it on), `SSCBIRRT_SANITIZE` (adds
`-fsanitize=address,undefined` to tests; CI runs the test target with it,
as #85 requires). Warnings are `-Wall -Wextra -Wpedantic -Werror` on the
core and tests.

The wheel is built by scikit-build-core driving this CMake project with
`SSCBIRRT_BUILD_PYTHON=ON`, which replaced the setuptools backend in v1.5.0. ssik
builds its extension from a hatchling hook instead; the difference here is
that the C++ package must be installable and consumable on its own (#85's
consumer criterion), so CMake is the primary build and the wheel reuses it
rather than the other way around. This is a decision for review.

The consumer example is the acceptance test of the export: it is built
against `cmake --install`'s output, not the source tree, exactly as ssik's
`cpp/examples/consumer` is.

### Release wheels (#153, v3.0.0)

`release.yml` builds the wheels with cibuildwheel from the settings in
`pyproject.toml`: CPython 3.10 through 3.14 on manylinux_2_28 x86_64 and
macOS arm64, the platforms for which the mujoco wheel we build against
exists. Two rules keep the dependency boundary intact in the published
artifacts. First, **libmujoco is excluded from wheel repair** (auditwheel and
delocate both take `--exclude`): `sscbirrt._native_mujoco` links the mujoco
wheel's own library through its rpath and refuses any other version, and a
bundled copy would silently defeat that check while doubling the wheel.
Second, **Eigen is fetched when absent**: the manylinux image has no Eigen,
so the CMake build downloads the pinned 3.4.0 release with a checked hash
and generates an `Eigen3Config.cmake` for it, which is also what makes the
sdist compile on a bare machine. Every wheel is installed with
`[mujoco,ssik]` in cibuildwheel's test environment and runs
`tools/reference_artifact.py --backend native --check` and `--backend auto
--check` before it is kept, so the published artifact is the one that passed
the parity gate.

## Minimal standalone consumer

```cpp
#include <sscbirrt/sscbirrt.hpp>
#include <cstdio>

int main() {
  using namespace sscbirrt;
  auto space = std::make_shared<JointSpace>(std::vector<double>{-3.14159, -3.14159},
                                            std::vector<double>{ 3.14159,  3.14159});
  auto start = std::make_shared<FiniteSet>(std::vector<Config>{{-2.0, 0.5}});
  auto goal  = std::make_shared<FiniteSet>(std::vector<Config>{{2.0, -0.5}, {2.0, 0.5}});
  auto walls = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{
      {{-0.2, -3.2}, {0.2, 1.0}}});   // a slab at q0 in (-0.2, 0.2) for q1 below 1.0

  PlanningProblem problem{space, start, goal, walls, nullptr, nullptr};
  PlannerConfig config;
  config.step_size = 0.2;
  config.timeout_seconds = 5.0;

  Planner planner(config);
  PlanResult r = planner.solve(problem, SolveOptions{.seed = 0});
  if (r.status != Status::Success) {
    std::printf("no path: %s\n", r.reason.c_str());
    return 1;
  }
  std::printf("%zu waypoints; goal member %d\n", r.path.size(), r.goal_source.back());
  return 0;
}
```

```cmake
cmake_minimum_required(VERSION 3.16)
project(consumer LANGUAGES CXX)
find_package(sscbirrt REQUIRED)
add_executable(plan main.cpp)
target_link_libraries(plan PRIVATE sscbirrt::core)
```

## Concept map

Every native concept maps to a Python concept. "same" means the contract is
identical; otherwise the difference and its reason are stated.

| Python | Native | Relation |
|---|---|---|
| `JointSpace(lower, upper, angular_joints)` | `JointSpace` | same: finite limits required on bounded joints, angular joints ignore theirs and sample one full turn (#107) |
| `CBiRRTConfig.angular_joints` | on `JointSpace` only | same: in Python too the space owns the topology and the config field only feeds the legacy constructor; the binding does what `CBiRRT.__init__` does |
| `PlanningProblem.sampler`, `SpaceSampler` | same | same (#110) |
| `StateSet.contains` | `StateSet::contains` | same |
| `supports(s, Cap)` | `s.sampler() != nullptr` etc. | same meaning; accessor instead of protocol check |
| `Sample(q, source)` | `Sample{q, source}` | same |
| `SetSampler/Distance/Violation/Projector` | same names | same contracts |
| `FiniteSet(configs, tolerance, metric)` | `FiniteSet` | same, including rejecting an empty member list at construction |
| `PredicateSet(fn)` | `PredicateSet(std::function)` | same in C++; not lowerable from Python in v1.5.0 (would need a callback) |
| `EmptySet` | `EmptySet` | same |
| `AnyOf`, `AllOf`, strategies | same | same rules; strategy requirements checked at `AllOf` construction through `requires`, as in Python |
| `is_finite`, `members`, `seeds` | virtual methods | same |
| `CollisionChecker.is_valid` | `StateValidator::is_valid` | same; renamed because validity is not only collision |
| `sscbirrt.testing.NoCollision`, `Wall` | `AcceptAll`, `JointBoxObstacles` | same predicates; `Wall`'s `extent` becomes a second box dimension |
| `LocalMotion`, `MotionValidator`, `DiscreteMotionValidator`, `RestrictedMotionValidator` | same | same contracts |
| `PlanningProblem` | `PlanningProblem` | same roles; `shared_ptr<const>` ownership |
| `CBiRRTConfig` | `PlannerConfig` | same ranges and messages (#108); two fields renamed where the Python name was TSR-specific (`tsr_samples` → `sample_draws`, `max_ik_per_pose` → `max_per_draw`); set-owned tolerances live on the sets, which is where Python's lowering puts them |
| `abort_fn` | `CancellationToken` | same three polling points and outcomes (#109) |
| `seed` | `SolveOptions::seed`, `std::mt19937_64` | **the one difference**: Python uses numpy's PCG64. Same seed gives a repeatable solve within each backend, never the same path across them; the artifact compares semantics. A PCG64 port would remove this at the cost of coupling every set's draw order to numpy internals, and #85 lists waypoint equality as a non-goal. |
| `PlanResult.success/failure_reason` | `Status` + `reason` | same information; `reason` prefixes preserved |
| `AllStart/Goal...Invalid/InCollision`, `ValueError` for an empty role | `NoRoots` + `RootReport`, `std::invalid_argument` | same: exceptions in both; the binding maps them by name and by `only_collisions()` |
| `UnsupportedCapability`, `MotionContractError` | `UnsupportedCapability`, `ContractError` | same triggers |
| `RRTree`, `Node` | `Tree`, `Node` | same; nearest with lowest-index tie-break |
| `unwrap_path` output rule | same | same |

## Open questions for review

1. **Core without Eigen.** Plain `std::vector<double>` keeps the consumer
   story dependency-free. If v1.6.0 pose regions want Eigen in the core
   rather than in `sscbirrt::tsr`, that is a later, separate decision.
2. **scikit-build-core versus a hatchling hook** for the wheel. Argued above;
   the alternative keeps sscbirrt on one build tool with ssik.
3. **Exposing trees.** Kept because Python exposes them and the artifact
   tooling inspects them. They cost a copy per solve; a flag on
   `SolveOptions` can suppress it.
4. **`PredicateSet` in the binding.** Excluded in v1.5.0 to honor the
   no-callback rule. A later release could allow it with the GIL held and a
   documented cost.

---

# v1.6.0 addendum: native pose regions and IK-lifted sets

This section extends the contract above to TSR planning without Python in
the loop (#87, #90, #91). The rules of the first part hold unchanged; the
Python `TSRConfigurationSet` (`tsr_set.py`) and sstsr 3.1 are the reference,
and the same principle applies: one contract, two implementations, with any
rule the reference needed sharpened changed in Python first.

## Scope

- A native **pose region** for a single `tsr.TSR`: frames `T0_w` and `Tw_e`,
  six bounded local coordinates, containment, closed-form distance, closest
  world-frame transform, and seeded sampling, each conforming to sstsr.
- Native **kinematics interfaces**: forward kinematics and inverse
  kinematics as abstract types in the core, so any implementation can lift
  a pose region.
- A native **`TSRConfigurationSet`** over those interfaces with all four
  capabilities, copied rule for rule from the Python set.
- An **SSIK adapter** implementing the interfaces through ssik's header-only
  C++ family solvers, for the families sscbirrt has verified. v1.6.0 verifies
  one: `ikgeo.three_parallel`, the UR family.
- **Lowering** of a Python `TSRConfigurationSet` whose region is a `TSR` and
  whose IK is an `SSIKSolver` wrapping an `ssik.Manipulator` of a verified
  family, with every other combination reported as unsupported.

Not in v1.6.0: TSR chains (they used sstsr's bounded numerical inverse and
stayed Python until #184, below), other IK backends, collision checkers (MuJoCo is v1.7.0, so the
v1.6.0 corpus uses the native validators of v1.5.0), and treating a list of
TSRs as anything but what the Python expression says.

## Targets and dependency boundary

```
sscbirrt._native ──► sscbirrt::ssik ──► sscbirrt::tsr ──► sscbirrt::core
                          │
                          └──► ssik::ssik_cpp (header-only) ──► Eigen3
```

| Target | Depends on | Contents |
|---|---|---|
| `sscbirrt::core` | standard library | as before, plus the `ForwardKinematics` and `IKSolver` interfaces and a `Transform` type |
| `sscbirrt::tsr` | `sscbirrt::core`, `sstsr::sstsr_cpp` (since #184) | `TSR` and `TSRChain` regions (sstsr's), `TSRConfigurationSet`, volume weighting |
| `sscbirrt::ssik` | `sscbirrt::tsr`, `ssik::ssik_cpp`, `Eigen3::Eigen` | `SSIKArm`: kinematics and IK through ssik's family solvers |

`core` and `tsr` stay standard-library only (since #184, `tsr` also uses
sstsr's C++, itself standard-library only; see the end of this document). The decision recorded in the
first part, core without Eigen, therefore extends to the pose-region
runtime: a 4x4 transform is `std::array<double, 16>` (row-major) with the
handful of operations the TSR math needs (multiply, inverse of a rigid
transform, RPY conversions). Eigen enters only in `sscbirrt::ssik`, where
ssik's headers require it, and is converted at that boundary. The reason is
the same as before: a C++ consumer of the planner and of TSRs should not
inherit Eigen unless it uses SSIK.

`sscbirrt::ssik` is optional at build time (`SSCBIRRT_WITH_SSIK`, default
ON when `ssik_cpp` and Eigen are found, else OFF with a status message). A
build without it still plans natively for everything v1.5.0 supports and
reports TSR problems as unsupported with the reason "built without SSIK
support". ssik 7.0 ships its headers in the wheel and exposes them
(personalrobotics/ssik#641): `ssik.get_cmake_dir()` holds
`ssik_cppConfig.cmake`, which exports `ssik::ssik_cpp` and finds Eigen3 as
a dependency. ssik is a build requirement of the wheel, so the isolated
build asks its interpreter for that directory (`SSCBIRRT_WITH_SSIK=AUTO`);
a pure CMake build passes `SSCBIRRT_SSIK_CMAKE_DIR` or puts it on
`CMAKE_PREFIX_PATH`. A build without ssik or Eigen disables the adapter
and `sscbirrt._native.has_ssik()` is false with a reason. sscbirrt pins
`ssik>=7.0,<8` for the native SSIK support. The C++ interface used here,
`three_parallel_artifact_solve(consts, limits, T, params)` with
`JointConsts<6>`, `JointLimits<6>`, and `ArtifactParams<6>`, is fixed for
7.0.

## Transform and kinematics interfaces (core)

```cpp
namespace sscbirrt {

struct Transform {                      // 4x4 homogeneous, row-major
  std::array<double, 16> m;
  static Transform identity();
  Transform operator*(const Transform&) const;
  Transform inverse_rigid() const;      // transpose the rotation, negate-rotate the translation
  double at(int r, int c) const;
};

class ForwardKinematics {
 public:
  virtual ~ForwardKinematics() = default;
  virtual int dof() const = 0;
  virtual Transform fk(ConfigView q) const = 0;   // end-effector pose in the planner's frames
};

class IKSolver {
 public:
  virtual ~IKSolver() = default;
  virtual int dof() const = 0;
  // Every solution, unfiltered, as the Python IKSolver protocol: no collision, no cap. seed may be empty.
  virtual std::vector<Config> solve(const Transform& pose, ConfigView seed) const = 0;
};

}  // namespace sscbirrt
```

These mirror `RobotModel.forward_kinematics` and `IKSolver.solve`. Joint
limits are not part of `ForwardKinematics`; the space owns them, as in
Python. An implementation must be `const` and safe to call repeatedly from
one thread.

## Pose region (tsr)

```cpp
namespace sscbirrt::tsr {

struct Bounds6 { std::array<std::array<double, 2>, 6> rows; };   // [lo, hi] for x y z roll pitch yaw

class TSR {
 public:
  TSR(Transform T0_w, Transform Tw_e, Bounds6 Bw);     // throws std::invalid_argument
  bool contains(const Transform& T) const;
  double distance(const Transform& T) const;            // 0 if contained; else Berenson 2011 Sec. 4.2, rotation weight 1
  std::pair<double, Transform> closest_transform(const Transform& T) const;
  Transform sample(Rng& rng) const;                    // six unit draws, in order x y z roll pitch yaw
  double volume() const;                                // sum of widths, rotational widths clamped to one turn
  const Bounds6& continuous_bounds() const;             // sstsr's _Bw_cont
};

}  // namespace sscbirrt::tsr
```

Every rule is sstsr 3.1's, stated here so the C++ can be checked against
the text as well as against the differential corpus:

- **Construction.** `T0_w` and `Tw_e` must be finite 4x4 with last row
  `0 0 0 1` and a rotation block with $R R^T = I$ and $\det R = +1$ within
  `tsr.FRAME_ATOL` (1e-6, absolute); `Bw` must be a finite 6x2 with
  `lo <= hi` on the three translations. These are sstsr 3.2.0's rules
  (personalrobotics/tsr#162), stated in its `docs/ARCHITECTURE.md` as
  conditions on the inputs so that a second implementation reproduces them;
  sscbirrt pins `sstsr>=3.2,<4`. Rotational rows may have `hi < lo`: that
  is an outer interval wrapping through $\pm\pi$, and its width is
  $2\pi + (hi - lo)$. Widths are clamped to $2\pi$. The continuous bounds
  wrap each rotational `lo` into $[-\pi, \pi)$ and set `hi = lo + width`.
- **Frames.** For a world-frame end-effector pose `T`, the local pose is
  `Tw_s' = inv(T0_w) · T · inv(Tw_e)`; its translation and rotation are
  tested against the continuous bounds.
- **RPY convention.** `rpy_to_rot` is Z-Y-X (yaw about z, then pitch about
  y, then roll about x), exactly the nine-entry formula in sstsr.
  `rot_to_rpy` takes `pitch = -asin(R[2,0])` off the gimbal lock and the
  two coupled branches at `|R[2,0]| = 1` with yaw fixed to 0, with the
  gimbal test `| |R[2,0]| - 1 | < 1e-9` and `R[2,0]` clipped to $[-1, 1]$.
- **Containment.** Translation within bounds with sstsr's `EPSILON`
  slack on both sides. Rotation: off the singularity, both pitch solutions
  `p` and `pi - p` are tried and the first whose RPY passes is accepted;
  at the singularity the four corner combinations of the coupled roll and
  yaw bounds are tried, and pitch outside the bounds fails immediately.
  Each RPY is wrapped into the continuous interval starting `EPSILON` below
  the lower bound before the test, and an outer interval (`lo > hi +
  EPSILON`) accepts either side.
- **Distance.** If contained, zero. Otherwise the displacement of Section
  4.2: from the local pose's RPY, the nine candidate representations
  `(r, p, y)` and `(r ± π, ±π - p, y ± π)`, each wrapped into the
  continuous interval starting at the lower bounds, each producing a
  per-coordinate displacement to the nearer violated bound (zero inside);
  the candidate with the smallest Euclidean norm wins (strict `<`, so the
  first minimum is kept). The distance is that norm with rotation weight 1.
- **Closest transform.** The winning candidate clipped into the continuous
  bounds, rotational entries wrapped back into $[-\pi, \pi)$, composed as
  `T0_w · xyzrpy_to_trans(bwopt) · Tw_e`. When contained, the "closest"
  pose is the input's own coordinates, as sstsr's `to_xyzrpy` computes
  them.
- **Sampling.** Six uniform draws in the continuous bounds, in coordinate
  order, from the caller's RNG and nothing else; the rotational draws are
  wrapped into $[-\pi, \pi)$; the result is `T0_w · xyzrpy_to_trans(s) ·
  Tw_e`. The draw is `unit(rng)` from the core, so a seeded native solve
  stays repeatable across standard libraries.
- **Volume** is `_interval_sum`: the sum of the six widths with rotational
  widths clamped to $2\pi$. Mixture weights for an `AnyOf` of TSR sets are
  proportional to volume, uniform when every volume is zero
  (`weights_from_tsrs`); the binding lowers the Python-computed weights as
  numbers, so this only matters for a pure C++ consumer.

`EPSILON` (0.001, the containment slack), `FRAME_ATOL` (1e-6), and the
gimbal test's 1e-9 are sstsr's constants and are copied, not redefined.

## The lifted set (tsr)

```cpp
namespace sscbirrt::tsr {

class TSRConfigurationSet final : public StateSet, public SetSampler, public SetDistance,
                                  public SetViolation, public SetProjector {
 public:
  TSRConfigurationSet(TSR region, std::shared_ptr<const ForwardKinematics> fk, std::shared_ptr<const IKSolver> ik,
                      std::shared_ptr<const JointSpace> space, double tolerance = 1e-3,
                      int max_projection_iters = 50, double progress_tolerance = 1e-6);
  // contains: distance(q) <= tolerance;  distance: region.distance(fk(q));  violation: max(0, distance - tolerance)
  // sample: one pose from the region, every IK solution within the space's limits, provenance empty
  // project: Python's loop, below
};

}  // namespace sscbirrt::tsr
```

`project(q_previous, q_proposed)` is Python's loop exactly: start at
`q_proposed`; up to `max_projection_iters` times, take `(dist, target) =
region.closest_transform(fk(q))`; if `dist <= tolerance` return `q`; if the
distance did not shrink by `progress_tolerance` return nothing; solve IK
for `target` seeded with `q`; among solutions within the space's limits take
the one nearest `q` under the space metric; none means failure. `q_previous`
is accepted and unused, as in Python. Membership tolerance and projection
parameters are the set's own, as the tolerances section says; the binding
copies them from the Python set instance.

Sampling filters by `space.within_limits`, not by admissibility: collision
is the planner's validator, as in Python, and every candidate reaches the
planner.

## The SSIK adapter (ssik)

```cpp
namespace sscbirrt::ssik {

enum class Family { ThreeParallel };                   // the verified allowlist; grows per release

struct ArmSpec {                                      // rendered from an ssik.Manipulator's KinBody
  Family family;
  std::array<Eigen::Vector3d, 6> axis;                // JointConsts<6>, as cpp_emit renders them
  std::array<Eigen::Matrix4d, 6> t_left, t_right;
  std::array<bool, 6> revolute;
  std::array<std::optional<std::pair<double, double>>, 6> limits;   // JointLimits<6>; nullopt = unlimited
  Transform T_base = Transform::identity();          // SSIKSolver.T_base
  Transform T_ee = Transform::identity();            // SSIKSolver.T_ee
};

class SSIKArm final : public ForwardKinematics, public IKSolver {
 public:
  explicit SSIKArm(ArmSpec spec);                     // throws std::invalid_argument for an unverified family
  int dof() const override { return 6; }
  Transform fk(ConfigView q) const override;         // T_base · ssik::fk(consts, q) · T_ee
  std::vector<Config> solve(const Transform& pose, ConfigView seed) const override;
};

}  // namespace sscbirrt::ssik
```

`solve` is `SSIKSolver.solve`: the target is mapped into SSIK's frames
(`inv(T_base) · pose · inv(T_ee)`), `ArtifactParams<6>` is the Python
adapter's call (`respect_limits = true`, `enumerate_windings = true`, no
cap, seed present iff given, ssik's default seed metric and rescue), and
every returned `q` is copied out. Windings and branches are preserved
because the same ssik code produces them on both sides; the Python adapter
calls the same family solver through ssik's own extension.

Verification per family (#90): before a family enters the allowlist, the
C++ adapter is checked against Python's `SSIKSolver` on an integration
corpus of the family's robot (UR5e from the Menagerie MJCF): identical
solution sets up to ordering within 1e-9, and FK closure of every solution
against the independent MuJoCo FK. Singular and near-singular poses are in
the corpus and compared for agreement, not completeness; SSIK's contract
governs what is found there.

## Lowering

`lower` extends its table:

| Python | Native | Otherwise |
|---|---|---|
| `TSRConfigurationSet` with `tsr: TSR`, `ik` a `KinematicsIntegration` (since #147; `SSIKSolver` around an `ssik.Manipulator` whose `solver_name` is verified, `dof == 6`, is the shipped one) | `TSRConfigurationSet(TSR, arm, arm, space, tolerance, max_projection_iters, progress_tolerance)` with `arm = ik.native_kinematics()` | `tsr` neither a `TSR` nor a `TSRChain` (since #184): "<type> has no native form"; `ik` not a `KinematicsIntegration`: "IK <type> is a Python object"; `native_kinematics()` raised `NativeUnsupported`: its reasons (SSIK: "SSIK family <name> is not in the native allowlist", "built without SSIK support"); returned something else: "not a sscbirrt ForwardKinematics and IKSolver" |

Two consistency checks run at lowering, each a blocker with a reason if it
fails. First, the Python set's `robot.forward_kinematics` and the SSIK
adapter's `fk` must agree within 1e-6 on every explicit configuration in the
problem (the seeds), because the native set uses SSIK's FK where the Python
set used the robot model's; the two are equal by construction for a
manipulator built from the same MJCF, and the check makes the assumption
visible. Second, the Python `TSR` must pass the native constructor's
validation, which the Python constructor now also performs.

The `ArmSpec` is rendered in Python from the manipulator through
`ssik.cpp.joint_data` (7.0), the public form of what `cpp_emit.py` renders:
per joint the axis, the fixed transforms on either side of the joint, the
joint type, and the limits. It is rendered once per lowering and shared by
every set in the problem that wraps the same `SSIKSolver`.

## Conformance corpus (#87)

`tools/tsr_conformance.py` generates `tests/reference/tsr_conformance.json`
from sstsr with a fixed seed: a list of TSRs (identity and non-identity
frames, zero-width rows, full rotations, outer rotational intervals, near
singular pitch) and, for each, a list of probe transforms (samples from
the region, samples perturbed off it, poses with pitch within 1e-7 of
$\pm\pi/2$) with sstsr's `contains`, `distance`, and `closest_transform`
results. `tests/test_tsr_conformance.py` runs the native `TSR` on every
probe and asserts: identical containment; distances equal within 1e-9;
closest transforms equal within 1e-9 elementwise, and contained by both
implementations within the declared tolerance; sampling with a fixed
seed produces poses the region contains, and consumes exactly six draws
per sample. The corpus is the inspectable artifact the issue asks for; it
is regenerated by one command and compared on regeneration like the
planning artifact.

## The v1.6.0 artifact (#91)

A UR5e problem: finite start, an `AnyOf` of two goal TSRs weighted by
volume, and one path-constraint TSR, with `AcceptAll` as the validator
(collision is v1.7.0). Recorded: seed, the ssik version and `solver_name`,
the MJCF's path and hash, selected endpoint provenance, path, the
independent validation report (every waypoint and interpolated edge sample
checked with the Python set's `contains` and the space), and the
profile-hook proof of zero Python calls during the solve. The same problem
runs on the Python backend and the two are compared as the planning
artifact compares them: outcome, validation, provenance where unique.

## Concept map, addendum

| Python | Native | Relation |
|---|---|---|
| `tsr.TSR` | `sscbirrt::tsr::TSR` | same math and the same construction rules and tolerance (sstsr 3.2.0) |
| `PoseRegion` protocol | `std::variant<sstsr::TSR, sstsr::TSRChain>` | since #184 both are sstsr's C++ |
| `TSRConfigurationSet` | `sscbirrt::tsr::TSRConfigurationSet` | same rules; FK comes from the IK adapter rather than a separate robot model, checked equal at lowering |
| `RobotModel.forward_kinematics`, `IKSolver.solve` | `ForwardKinematics`, `IKSolver` | same contracts |
| `SSIKSolver` | `sscbirrt::ssik::SSIKArm` | same call into the same ssik family solver; families outside the allowlist are unsupported rather than silently different |
| `tsr_weights`, `region_volume` | `TSR::volume` | same |
| numpy RNG in `TSR.sample` | `unit(rng)`, six draws | the one difference, as before: engine only |

## Decisions

1. **Header source for ssik_cpp**: resolved. ssik 7.0 ships the headers and
   `ssik.get_cmake_dir()` (personalrobotics/ssik#641); sscbirrt consumes
   them through `find_package(ssik_cpp)` and does not vendor.
2. **Eigen only in `sscbirrt::ssik`**, with a sixteen-double `Transform` in
   the core: adopted. The alternative, Eigen in the core, would make every
   consumer inherit it.
3. **FK-agreement check at lowering** (1e-6 on the seeds): adopted. It
   catches a robot model that disagrees with the SSIK model, the one way the
   native set could differ from the Python set, and costs a few FK
   evaluations.
4. **Python `TSR` validation**: resolved upstream in sstsr 3.2.0
   (personalrobotics/tsr#162) with the tolerance exported as
   `tsr.FRAME_ATOL`; sscbirrt pins `sstsr>=3.2,<4`.

---

# v1.7.0 addendum: the owned MuJoCo scene

This section extends the contract to collision checking in C++ against a
MuJoCo world the planner owns (#93, #84, #89, #88). The reference for the
semantics is `mj_manipulator`'s `CollisionChecker` in snapshot mode
(`mj_manipulator/collision.py`) and its `GraspManager`'s attachment update,
read at mj_manipulator commit `f3c1` of 2026-09; sscbirrt's own
`MuJoCoCollisionChecker` is the degenerate case with no attachments. The
principle is unchanged: one contract, two implementations, and the
comparison corpus is the oracle.

## Scope

- A native **scene**: an `mjModel` the planner owns, loaded from the MJB
  bytes of a compiled Python model, with the controlled joints, bodies,
  sites, and geoms resolved and validated at construction.
- An immutable **snapshot**: everything a geometric query needs from the
  live world at one instant: the full `qpos`, mocap poses, attachments,
  and allowed-contact sets. Nothing else.
- A native **state validator** over a scene and a snapshot implementing
  `StateValidator`, with mj_manipulator's attachment-aware contact policy.
- **Packaging** that keeps `import sscbirrt` free of MuJoCo: the scene
  lives in its own extension module, built when a `mujoco` wheel is
  present at build time and imported only on demand.
- **Lowering** of the Python `NativeCollisionChecker` (the Python face of
  the scene) into the native validator, and a one-call path from a Python
  `MjModel` and `MjData` to a native solve.
- Two release artifacts: a UR5e TSR-goal query among obstacles and a
  held-object query, both from snapshots, native SSIK plus native
  collision, zero Python callbacks; and a decision-parity corpus between
  the native validator and mj_manipulator's checker.

Not in v1.7.0: MuJoCo thread pools, parallel edge validation, continuous
collision checking (edges remain discrete at `edge_resolution`), dynamics
of any kind, and moving execution or simulator ownership into sscbirrt
(mj_manipulator#175 keeps those).

## Targets and dependency boundary

```
sscbirrt._native          ──► sscbirrt::ssik ──► sscbirrt::tsr ──► sscbirrt::core
sscbirrt._native_mujoco   ──► sscbirrt::mujoco ──────────────────► sscbirrt::core
                                     └──► libmujoco (the mujoco wheel's shared library and headers)
```

| Target | Depends on | Contents |
|---|---|---|
| `sscbirrt::mujoco` | `sscbirrt::core`, MuJoCo headers and library | `Scene`, `Snapshot`, `SceneValidator` |
| `sscbirrt._native_mujoco` | `sscbirrt::mujoco`, `sscbirrt._native` (for the `StateValidator` base type) | the binding |

The MuJoCo scene is a **separate extension module**. `sscbirrt._native`
never references MuJoCo, so `import sscbirrt` and every v1.5 and v1.6
feature work with no `mujoco` installed (#89's import criterion), and a
missing or mismatched MuJoCo library fails at `import
sscbirrt._native_mujoco`, which `sscbirrt.backends.native_mujoco` performs
lazily and turns into a reason. The alternative, one module that loads
`libmujoco` at runtime through `dlopen` and a hand-written function table,
avoids a second module but reimplements the dynamic linker for a dozen
symbols; a second module is the ordinary tool. `SceneValidator` derives
from the `StateValidator` registered by `_native`; pybind11 shares
registered base types across modules built with the same pybind11.

Build and link: the wheel lists `mujoco` as a build requirement (as it
lists ssik), takes the headers from `<mujoco package>/include` and the
library `libmujoco.<version>.dylib` or `libmujoco.so.<version>` from the
package directory, and sets the module's rpath to `@loader_path/../mujoco`
and `$ORIGIN/../mujoco`, which is where the wheel that supplied the
headers puts the library at runtime. Without a `mujoco` wheel at build
time, `SSCBIRRT_WITH_MUJOCO=AUTO` disables the target with a status
message and the reason "built without MuJoCo support".

**Version binding.** MJB is a version-specific format and `mjModel` is a
version-specific struct, so the module is bound to the MuJoCo it was built
against: the build records `mjVERSION_HEADER`, and at import the module
compares it with `mj_version()` of the library it loaded and
`sscbirrt.backends.native_mujoco` compares both with `mujoco.__version__`.
Any difference refuses the scene with a reason naming the three versions.
The `mujoco` extra pins the exact version the wheel was built against:
**`mujoco==3.14.0`**, the latest release, which #93 names for the first
verified artifact; the workspace moves to it with this milestone. The
3.14.0 headers carry every entry point and field this contract uses
(`mj_loadModelBuffer`, `mj_saveModel`, `mj_sizeModel`, `mj_kinematics`,
`mj_collision`, `mj_makeData`, `mj_deleteData`, `mj_deleteModel`,
`mj_name2id`, `mj_version`, `mju_mat2Quat`, `mjContact::geom`,
`mjModel::signature`, `geom_bodyid`, `jnt_qposadr`, `body_parentid`,
`mocap_pos`).

## Scene

```cpp
namespace sscbirrt::mujoco {

struct SceneProvenance {
  std::string mujoco_version;       // mj_versionString() of the loaded library
  std::uint64_t model_signature;    // the Python model's signature as passed; mj_saveModel does not serialize it
  std::string mjb_sha256;           // of the bytes the scene was loaded from
};

class Scene {
 public:
  // Loads an independently owned mjModel with mj_loadModelBuffer. Throws std::invalid_argument for
  // an unloadable buffer, a version mismatch, an unknown or duplicate joint, a controlled joint that
  // is not a hinge or slide, an unknown body, site, or geom, or an attachment body without a free joint.
  Scene(std::span<const std::byte> mjb, std::vector<std::string> controlled_joints,
        std::vector<std::string> extra_arm_bodies = {});
  ~Scene();  // mj_deleteModel

  int dof() const;
  const std::vector<int>& qpos_addresses() const;          // one per controlled joint
  const std::vector<double>& lower() const;                 // jnt_range, or ±infinity where jnt_limited is 0
  const std::vector<double>& upper() const;
  const std::vector<int>& arm_bodies() const;               // controlled joints' bodies, their descendants, extra bodies
  int body_id(const std::string& name) const;               // -1 if absent
  std::vector<int> subtree(int body) const;                 // body and all descendants
  int nq() const; int nmocap() const; int ngeom() const;
  const SceneProvenance& provenance() const;
  const ::mjModel* model() const;                           // read-only; shared by every validator and solve
};

}  // namespace sscbirrt::mujoco
```

The `mjModel` is shared read-only by every validator built on the scene and
never written after construction. The Python model that produced the MJB
may be destroyed or changed afterward; the scene does not observe it
(#93's first criterion). Any change to the world, structural (a body
added) or numeric (a geom resized in place), is a new MJB and a new scene;
the Python side keys its scene cache on the SHA-256 of the MJB bytes and
reconstructs. `mjModel.signature` is recorded as provenance but is not the
key: it hashes structure only, and a geom resized in place keeps its
signature (verified on 3.14.0). Limits follow the rule of
#107: an unlimited joint is `±infinity` and the planner's `JointSpace`
demands a declaration.

## Snapshot

```cpp
struct Attachment {
  int object_body;                 // must have a free joint (mjJNT_FREE) as its first joint
  int gripper_body;                // the body the object is attached to
  Transform T_gripper_object;      // recorded at grasp time
  std::vector<int> allowed_bodies; // bodies whose contacts with the object's subtree are allowed
};

struct Snapshot {
  std::vector<double> qpos;                 // nq
  std::vector<double> mocap_pos, mocap_quat; // 3*nmocap, 4*nmocap
  std::vector<Attachment> attachments;
  std::string sha256;                       // of every field above, for provenance
};
```

A snapshot is a value. It holds exactly what a geometric query reads and
nothing else: no `qvel`, `ctrl`, `act`, sensors, warmstarts, forces, or
time (#93's non-goals). Ingress validation (`std::invalid_argument`):
`qpos` has `nq` finite entries, mocap arrays have `3*nmocap` and
`4*nmocap` finite entries with unit quaternions within 1e-6, each
attachment names a valid object body with a free joint and a valid gripper
body, `T_gripper_object` passes the frame check of the TSR addendum, and
every allowed body id is valid. A snapshot captured from a live `MjData`
is copied; later changes to that `MjData` do not reach it (#93's third
criterion).

`allowed_bodies` makes mj_manipulator's gripper rule explicit data.
mj_manipulator allows contact between a grasped object and any body in the
subtree of the gripper **base**, found by name: for an attachment body
`prefix/part` it is `prefix/base` when that body exists, else the
attachment body itself. That name rule stays in Python, in
`Snapshot.capture`, which resolves it to body ids; the native side applies
sets, not names. The rule is therefore in one place and can change without
touching C++.

## Validation semantics

```cpp
class SceneValidator final : public StateValidator {
 public:
  SceneValidator(std::shared_ptr<const Scene> scene, Snapshot snapshot);  // owns a private mjData
  bool is_valid(ConfigView q) const override;
  std::vector<InvalidContact> invalid_contacts(ConfigView q) const;        // for diagnostics and the corpus
};

struct InvalidContact { int body1, body2; int geom1, geom2; double dist; Kind kind; };  // Kind: SelfCollision, RobotEnvironment
```

`is_valid(q)` is mj_manipulator's `is_valid` in snapshot mode, step for
step (#84's pipeline):

1. Restore the snapshot: copy `qpos`, `mocap_pos`, `mocap_quat` into the
   private `mjData`.
2. Write `q` into the controlled joints' `qpos` addresses.
3. `mj_kinematics`: body and geom poses from `qpos` and mocap. This is the
   position stage of `mj_forward`; contacts depend on geom poses only, so
   the two agree, and no dynamics, sensors, or forces run (#84's fifth
   criterion).
4. For each attachment, read the gripper body's `xpos` and `xmat`, form
   `T_world_object = T_world_gripper * T_gripper_object`, and write the
   object's free-joint `qpos` (position, then quaternion by `mju_mat2Quat`).
5. `mj_kinematics` again if there was any attachment.
6. `mj_collision`.
7. Classify every contact by the bodies of its two geoms
   (`geom_bodyid`). Let *arm* be the scene's arm bodies and *grasped* the
   union of the attachments' object subtrees; *robot* is either.
   - Neither body robot: ignored.
   - Both robot: allowed if one is grasped and the other is in that
     attachment's `allowed_bodies`; otherwise a **self-collision**.
   - Exactly one robot: a **robot-environment** collision.
   The configuration is valid iff no contact is invalid.

Every contact MuJoCo generates counts, including margin-inflated ones with
positive `dist`, because mj_manipulator counts `data.ncon` entries without
a distance test. `is_valid` has no threshold parameter; mj_manipulator's
`is_arm_in_collision(min_penetration)` is a control-time query outside
this contract. Contact count and order are never part of a decision or a
test (#84's third criterion).

Edges are the core's business: `SceneValidator` is a `StateValidator`, so
the default `DiscreteMotionValidator` samples every edge at
`edge_resolution` and calls it, and growth, the final connection, and
shortcuts all pass through the same boundary as before.

## Isolation and cancellation (#89)

One `SceneValidator` owns one `mjData`, created by `mj_makeData` at
construction and deleted with it. A solve uses the validator it was given
and nothing else touches that `mjData`; two simultaneous solves need two
validators, which the Python one-call path creates per solve. The
`mjModel` is shared read-only. The GIL is released once for the whole
solve as in v1.5; the deadline and the cancellation token are the core's,
polled at the three points already defined, so cancellation completes
within one state validation plus one edge's remaining samples, which is
bounded by `step_size / edge_resolution` collision queries.

## Python surface

```python
from sscbirrt.backends.native_mujoco import NativeScene, Snapshot, NativeCollisionChecker, plan_native

scene = NativeScene.from_model(model, joint_names, extra_arm_bodies=[...])     # exports MJB once; caches by signature
snap = Snapshot.capture(scene, data, attachments={obj: (gripper_body, T)})    # on the MuJoCo owner thread
checker = NativeCollisionChecker(scene, snap)                                 # a sscbirrt CollisionChecker
```

`NativeCollisionChecker.is_valid(q)` calls the native validator, so the
same object serves the Python backend and lowers to itself for the native
one; there is one implementation of the policy and the Python backend
exercises it too. `lower` gains the row: validator is a
`ValidatorIntegration` (#147; `NativeCollisionChecker` is the shipped one) →
`fresh()`, a native validator owned by this solve; the row for other
validators is unchanged (they remain Python objects). `NativeScene.from_model`
refuses a model whose `mujoco.__version__` differs from the module's
build version.

`plan_native(model, data, joint_names, *, ik, start, goal_tsrs, ...)` (deprecated since 3.1.0 for `sscbirrt.mujoco.plan` with an `Arm`) is
the one-call path #88 asks for: build or reuse the scene, capture the
snapshot, build the `SSIKRobotModel` and lowering, solve with
`backend="native"`, and return a `PlanResult` whose new `provenance` field
records `{"scene": signature, "mjb_sha256", "snapshot": sha256,
"mujoco", "ssik", "sstsr", "sscbirrt", "solver_name"}`. The Python backend
fills `provenance` with the versions only.

## Integrations (#147)

MuJoCo is one implementation of `StateValidator`, SSIK one implementation
of `ForwardKinematics` and `IKSolver`, TSRs one implementation of
`StateSet`. The core knows none of them, and the target graph above is the
shape every further integration takes. What made it reachable from Python
without a patch to sscbirrt is that `lower` recognizes validators and IK
solvers by two runtime-checkable protocols in `sscbirrt.backends.native`,
never by type:

| Protocol | Members | Shipped implementation |
|---|---|---|
| `ValidatorIntegration` | `fresh()` → a native `StateValidator` for one solve; `provenance` → `dict` | `NativeCollisionChecker` |
| `KinematicsIntegration` | `native_kinematics()` → one object that is a native `ForwardKinematics` and `IKSolver`, or raises `NativeUnsupported([reasons])`; `provenance` → `dict` | `SSIKSolver` |

`lower` merges every integration's `provenance` into `Lowered.provenance`
and so into `PlanResult.provenance`; the SSIK-specific checks (family
allowlist, wrapped type, extension built without SSIK) live in
`SSIKSolver.native_kinematics` and surface as the reasons they raise. The
FK-agreement check stays in `lower`, because it is about the problem
(`RobotModel` versus the native arm), not about any one IK library.

Adding a simulator or an IK library:

1. **Implement the interface in C++** in a target that links only
   `sscbirrt::core` (plus the library you integrate), the way
   `sscbirrt::mujoco` and `sscbirrt::ssik` do. `StateValidator` has one
   virtual, `is_valid(q)`. `ForwardKinematics` has `dof()` and `fk(q)`;
   `IKSolver` has `dof()` and `solve(T, seed)` returning every solution.
2. **Own extension module.** Bind with pybind11 in a module of your own,
   declaring `StateValidator` (or the kinematics bases) registered by
   `sscbirrt._native` as the base class; pybind11 shares registered types
   across modules built with the same pybind11. `sscbirrt._native` must not
   learn to import your library: a missing library is a reason, not an
   import error.
3. **Pin and verify the library version** at import, as the MuJoCo module
   compares its build header, the loaded library, and the Python package.
4. **Python integration object.** A `CollisionChecker` that also satisfies
   `ValidatorIntegration`, or an `IKSolver` that also satisfies
   `KinematicsIntegration`. `fresh()` gives each solve its own scratch state
   (the MuJoCo one allocates an `mjData`); `native_kinematics()` raises
   `NativeUnsupported` with a reason for any instance that has no native
   form rather than returning something approximate.
5. **One implementation, or a parity corpus.** Either the Python form calls
   the native one (as `NativeCollisionChecker.is_valid` does) or the two are
   checked against each other on a checked-in corpus with a `--check` tool,
   as `tools/mujoco_collision_corpus.py` and `tools/tsr_conformance.py` do.

## Artifacts

**Decision parity** (#84): `tools/mujoco_collision_corpus.py` builds MuJoCo
scenes from MJCF strings checked into the tool, covering self-collision,
robot-environment, held-object-environment, allowed gripper-object, mocap
bodies, contact margins, cylinders, box-box, and plane-mesh, and for each
records configurations near and across contact with mj_manipulator's
`CollisionChecker` decision and the native `SceneValidator` decision, into
`tests/reference/mujoco_collision_corpus.json`. The test asserts the
decisions agree on every configuration; where they cannot (a policy
mj_manipulator has and the snapshot does not express) the corpus records
the difference by name, and the addendum is amended before the release.

**Release artifact** (#88): two cases added to `tools/reference_artifact.py`
under the Menagerie UR5e with the Robotiq gripper: a TSR-goal query among
table obstacles, and a held-object query with a grasped cylinder and an
allowed gripper contact, both from snapshots, native SSIK, native
collision, `native_python_calls == 0`, independent post-validation with the
Python checker on the same snapshot, and the provenance block. Both run on
the Python backend as well and are compared as the other cases are.

## Concept map, addendum

| Python | Native | Relation |
|---|---|---|
| `mj_manipulator.CollisionChecker` (snapshot mode) | `SceneValidator` | same decisions on the corpus; the gripper-base name rule is resolved in Python to `allowed_bodies` |
| `MuJoCoCollisionChecker` | `SceneValidator` with no attachments | the legacy checker counts every contact in the world, environment against environment included, so it rejects any world with resting contacts; the policy ignores those. It never accepts what the validator rejects (checked on the corpus). |
| `GraspManager.update_attached_poses` | steps 4 and 5 | same transform and free-joint write |
| `mj_forward` in the checker | `mj_kinematics` then `mj_collision` | same contacts; no dynamics run |
| live `MjModel`, `MjData` | `Scene` from MJB, `Snapshot` by value | difference by design (#93): no borrowed pointers, no observed mutation |

## Bulletproofing

The maintainer's direction for this milestone is correctness first;
mj_manipulator is refactored onto this boundary afterward, so the boundary
must hold without it. Beyond the rules above:

- **Ingress is exhaustive and tested one rule at a time.** Every rejection
  in Scene and Snapshot has a test that triggers exactly it: unloadable
  MJB bytes, a truncated buffer, a version mismatch in each of the three
  places, an unknown joint, a duplicate joint, a ball or free controlled
  joint, an unknown body, site, or geom, an attachment object without a
  free joint or whose free joint is not its first joint, a `qpos` of the
  wrong length or with a NaN, mocap arrays of the wrong length or with a
  non-unit quaternion, a non-rigid `T_gripper_object`, an allowed body id
  out of range, and an attachment whose gripper body is not in the arm.
- **Property tests** (Hypothesis) on the Python side generate snapshots
  and configurations against small MJCF worlds and assert the invariants
  that do not need an oracle: a snapshot never changes after capture even
  when the live `MjData` does; the same scene and snapshot give the same
  decision for the same configuration on every call and across two
  validators; a configuration valid with no attachment stays valid when an
  attached object is placed far from everything; decisions are independent
  of the order in which attachments and allowed bodies are listed.
- **Decision parity is the oracle** where one exists: the corpus against
  mj_manipulator's checker on every coverage item of #84, and the
  `mj_kinematics` versus `mj_forward` equivalence checked on the same
  corpus by running both in Python and comparing `ncon` and the contact
  geom pairs as sets.
- **Sanitizers and leaks.** The C++ scene tests run under ASan, UBSan, and
  LSan with MuJoCo linked, creating and destroying scenes and validators
  in loops, so a missing `mj_deleteModel` or `mj_deleteData` fails CI.
- **Isolation.** A test runs two solves concurrently from two threads on
  one scene with two validators and asserts both results equal their
  single-threaded runs.
- **Determinism.** Fixed seeds are repeatable across two solves and across
  the Python one-call path, on the release cases.
- **Import safety.** A CI job installs the wheel into an environment
  without `mujoco` and asserts `import sscbirrt` and a native finite
  problem work, and that the MuJoCo scene reports its reason.
- **Version binding** is tested by building against 3.14.0 and importing
  with the pinned wheel only; a test monkeypatches the reported Python
  version to a different string and asserts the scene refuses with a
  message naming both.

## Decisions

1. **A separate extension module** for the MuJoCo scene: adopted. The core
   imports without MuJoCo.
2. **Exact MuJoCo pin** in the `mujoco` extra, refusal on mismatch:
   adopted.
3. **Which MuJoCo**: 3.14.0, the latest release, per the maintainer; the
   workspace moves to it with this milestone.
4. **The gripper-base rule stays in Python** as data on the snapshot:
   adopted.
5. **`mj_kinematics` in place of `mj_forward`** for the query: adopted,
   with the equivalence checked on the corpus.

## TSRs from sstsr's C++ core (#184)

sstsr 3.3 ships its TSR and TSR-chain rules as C++ in the wheel (headers
and sources behind the CMake target `sstsr::sstsr_cpp`, found through
`tsr.get_cmake_dir()`). `sscbirrt::tsr` now uses them instead of its own
port of the single-TSR rules, which is deleted. One implementation of
sstsr's contract, owned by sstsr; the conformance corpus above stays as a
check of the binding against sstsr's Python.

- **Regions.** `TSRConfigurationSet` holds a
  `std::variant<sstsr::TSR, sstsr::TSRChain>` and keeps its algorithm:
  distance, closest transform and sampling go to the region; IK, limits and
  iterated projection are sscbirrt's, as in `tsr_set.py`. A chain's distance
  is the residual of its inverse. No warm starts: the native set calls
  exactly what the Python set calls.
- **The chain contract is sstsr's.** On exact paths (a single link, every
  link fixed) the C++ is bit-for-bit with the Python. The cold inverse is
  projected Levenberg-Marquardt in C++ and L-BFGS-B in Python, held to
  properties only: the residual is an upper bound, and "not found" is not
  a certificate. Parity is therefore semantic, as everywhere else, and the
  reference artifact gains a UR5e chain case (`ur5e_mujoco_tsr_chain_crank`)
  whose start, goal and path constraint are one chain.
- **Boundary.** `sscbirrt::core` keeps its own `Transform` and stays
  standard-library only; a pose crosses into sstsr as a copy of the same
  row-major 16 doubles. The planner's `Rng` is sstsr's engine type
  (static-asserted), so seeded solves stay repeatable. The RPY helpers the
  old port used are removed from the core.
- **Build.** `sstsr>=3.3,<4` is a build and runtime requirement. CMake finds
  `sstsr_cpp` from `SSCBIRRT_SSTSR_CMAKE_DIR` or by asking the interpreter,
  with no version pin, and links it privately into `sscbirrt_tsr` only: its
  sources compile into whatever links it. An editable sstsr install has no
  CMake package; build against a wheel.

## Roots during the search: CBiRRT's P_sample (#196)

Through 3.2, both backends collected roots once, before the search, and froze
them. `goal_bias` and `start_bias` only made a member of the *other* role's set
a turn's extension target, discarded if the extension fell short. CBiRRT
(Berenson et al. 2009, 2011) instead keeps growing the start and goal sets
during the search. In both backends now:

- **One tree per role, with a virtual root.** Each tree hangs off a virtual
  root that is not a configuration, never a nearest-neighbour candidate, never
  the end of an edge, and never on a path. Every start or goal member in the
  tree is its child, whether it joined before the search or during it. The
  representation is unchanged: a node with no parent (`parent = -1` in C++,
  `None` in Python) is such a child, and `Tree::add_root` / `RRTree.add_root`
  add one.
- **One draw, one rule.** `draw_roots` / `_draw_roots` takes one draw from a
  set and keeps up to `max_per_draw` admissible candidates, visited in random
  order when the draw offers more (#168), skipping explicit seeds. Admissible
  means inside the space, valid, and inside the path constraint. Collection
  before the search is this step repeated until `num_tree_roots` roots exist
  or `sample_draws` draws are spent. Its RNG consumption is unchanged.
- **The coin.** A role *grows* when its set is not finite and can be sampled,
  the predicate collection already used. On tree `a`'s turn, if `a`'s role
  grows and `unit(rng) < p_a` (`start_sample_probability` or
  `goal_sample_probability`), the turn makes one draw from `a`'s own set and
  adds every kept candidate as a root, with its provenance. That is the whole
  turn, and it counts as an iteration. Otherwise the turn is unchanged: one
  free-space sample, then extend `a` and connect `b`. A finite role never
  tosses the coin and consumes no random draw for it. Steering toward a
  member is gone: the connect step already pulls each tree toward the other's
  members.
- **Results.** `start_source` / `goal_source` name the root the path descends
  from, whenever it joined. `SolveStats.search_roots` and
  `PlanResult.stats["search_roots"]` count the roots added during the search.
  `RootReport` and `start_roots` / `goal_roots` keep describing collection
  before the search.
- **Configuration.** The C++ `PlannerConfig` fields are renamed. In Python,
  `goal_bias` and `start_bias` are deprecated aliases of the new names until
  4.0, and they warn that their meaning changed.

## Windings share a verdict (#200)

SSIK enumerates every in-limit winding of every geometric branch: on the
UR5e's ±2π joints, 320 solutions per pose for a median of 12 physically
distinct arm poses. q and q + 2πk place the links identically, so a
collision verdict cannot tell windings apart. Root collection now judges each
physical configuration once per draw. Two declarations, never inferred,
license it:

- **`IKSolver::revolute_joints()`** (Python: `revolute_joints`). These are the
  joints on which full turns are the same physical configuration. SSIK
  declares every joint; the default is none.
- **`StateValidator::full_turn_invariant()`** (Python: `full_turn_invariant`).
  The validator's verdict never changes under a full turn. The MuJoCo scene
  validator and `MuJoCoCollisionChecker` declare it when every controlled joint
  is a hinge. A validator over raw joint values, such as `JointBoxObstacles`,
  does not.

`TSRConfigurationSet` tags each candidate with `Sample::key`: the revolute
joints reduced to [-π, π], every joint rounded to nanoradians. `AnyOf`
preserves the key. In `draw_roots`, when the validator declares invariance,
the first candidate with a key runs the validator and later candidates with
that key reuse its verdict. The joint-space and path-constraint checks still
run per candidate, because limits do tell windings apart, and only the
validator's invariance is declared. Candidates, their order and their
verdicts are unchanged, so results are identical apart from cost.
`state_checks` still counts every candidate; `reused_verdicts` counts the
validator calls saved.
