// The non-TSR cases of tools/reference_artifact.py, checked with the artifact's own validation predicates.
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "harness.hpp"
#include "sscbirrt/sscbirrt.hpp"

using namespace sscbirrt;
constexpr double kPi = std::numbers::pi;
constexpr double kInf = std::numeric_limits<double>::infinity();

namespace {

// The planar reference arm's joint space: both joints in [-pi, pi].
std::shared_ptr<JointSpace> planar(std::vector<bool> angular = {}) {
  return std::make_shared<JointSpace>(std::vector<double>{-kPi, -kPi}, std::vector<double>{kPi, kPi}, std::move(angular));
}

// FiniteSet under the space metric with the config's membership tolerance, as the artifact's _finite().
std::shared_ptr<FiniteSet> finite(const JointSpace& space, std::vector<Config> qs) {
  return std::make_shared<FiniteSet>(std::move(qs), 1e-3, [&space](ConfigView a, ConfigView b) { return space.distance(a, b); });
}

PlannerConfig base() {
  PlannerConfig c;
  c.smooth_path = true;
  c.step_size = 0.2;
  c.connection_tolerance = 1e-3;
  c.edge_resolution = 0.05;
  c.timeout_seconds = 30.0;
  return c;
}

PlanningProblem problem(std::shared_ptr<JointSpace> space, SetPtr start, SetPtr goal, SetPtr constraint = nullptr,
                        std::shared_ptr<const StateValidator> validator = nullptr) {
  PlanningProblem p;
  p.space = space;
  p.start = std::move(start);
  p.goal = std::move(goal);
  p.validator = validator ? validator : std::make_shared<AcceptAll>();
  p.path_constraint = std::move(constraint);
  return p;
}

// tools/reference_artifact.py::validate, in C++.
struct Validation {
  bool all_in_space, first_in_start, last_in_goal, all_admissible, edges_ok, raw_steps_ok;
  bool all() const { return all_in_space && first_in_start && last_in_goal && all_admissible && edges_ok && raw_steps_ok; }
};

Validation validate(const PlanningProblem& p, const PlannerConfig& c, const std::vector<Config>& path) {
  const JointSpace& space = *p.space;
  auto admissible = [&p](ConfigView q) { return !Planner::why_inadmissible(p, q).has_value(); };
  Validation v{true, p.start->contains(path.front()), p.goal->contains(path.back()), true, true, true};
  for (const Config& q : path) {
    v.all_in_space = v.all_in_space && space.contains(q);
    v.all_admissible = v.all_admissible && admissible(q);
  }
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    const Config d = space.direction(path[k], path[k + 1]);
    double norm = 0.0, raw = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
      norm += d[i] * d[i];
      raw = std::max(raw, std::fabs(path[k + 1][i] - path[k][i]));
    }
    const int n = std::max(1, static_cast<int>(std::ceil(std::sqrt(norm) / c.resolution())));
    for (int i = 1; i <= n; ++i) {
      Config q(d.size());
      for (std::size_t j = 0; j < d.size(); ++j) q[j] = path[k][j] + (static_cast<double>(i) / n) * d[j];
      if (!admissible(q)) v.edges_ok = false;
    }
    if (raw > c.step_size + 1e-9) v.raw_steps_ok = false;
  }
  return v;
}

SolveOptions seeded(std::uint64_t seed) {
  SolveOptions o;
  o.seed = seed;
  return o;
}

// A sampleable, non-finite region on joint 0 (for root-draw cancellation and bias tests).
class Band final : public StateSet, public SetSampler {
 public:
  Band(double lo, double hi) : lo_(lo), hi_(hi) {}
  bool contains(ConfigView q) const override { return q[0] >= lo_ && q[0] <= hi_; }
  std::vector<Sample> sample(Rng& rng) const override { return {Sample{Config{lo_ + unit(rng) * (hi_ - lo_), 0.0}, {}}}; }
  const SetSampler* sampler() const override { return this; }
  std::string describe() const override { return "Band"; }

 private:
  double lo_, hi_;
};

}  // namespace

TEST(config_ranges_are_validated_with_pythons_messages) {
  PlannerConfig c;
  c.step_size = 0.0;
  bool caught = false;
  try {
    Planner p(c);
  } catch (const std::invalid_argument& e) {
    caught = std::string(e.what()) == "step_size must be positive, got 0";
  }
  CHECK(caught);
  c = PlannerConfig{};
  c.goal_sample_probability = 1.5;
  CHECK_THROWS(Planner{c}, std::invalid_argument);
  c = PlannerConfig{};
  c.start_sample_probability = -0.1;
  CHECK_THROWS(Planner{c}, std::invalid_argument);
  c = PlannerConfig{};
  c.extend_steps = 0;
  CHECK_THROWS(Planner{c}, std::invalid_argument);
  c = PlannerConfig{};
  c.connection_tolerance = 0.0;  // zero is exact, allowed
  Planner ok(c);
  (void)ok;
}

TEST(edge_resolution_defaults_to_005_independent_of_step_size) {
  PlannerConfig c;  // #204
  c.step_size = 0.3;
  CHECK(c.resolution() == 0.05);
  c.edge_resolution = std::nullopt;
  CHECK(c.resolution() == 0.3);
}

TEST(fixed_to_fixed) {
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  PlanResult r = planner.solve(p, seeded(0));
  CHECK(r.success());
  CHECK(validate(p, planner.config(), r.path).all());
  CHECK(r.start_source == Provenance{0} && r.goal_source == Provenance{0});
  CHECK(r.path.front() == (Config{0.0, 0.0}));
  CHECK(r.tree_start && r.tree_goal);
  CHECK(r.start_roots.roots == 1 && r.goal_roots.roots == 1);
}

TEST(same_seed_same_path) {
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  PlanResult a = planner.solve(p, seeded(7)), b = planner.solve(p, seeded(7));
  CHECK(a.path == b.path && a.iterations == b.iterations);
}

TEST(multiple_roots_report_provenance) {
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.0}, {0.1, 0.1}}), finite(*s, {{1.0, 0.5}, {-1.0, 0.5}}));
  PlanResult r = planner.solve(p, seeded(1));
  CHECK(r.success() && validate(p, planner.config(), r.path).all());
  CHECK(r.start_source.size() == 1 && r.goal_source.size() == 1);
  const Config& q0 = r.path.front();
  CHECK(q0 == (r.start_source[0] == 0 ? Config{0.0, 0.0} : Config{0.1, 0.1}));
  CHECK(r.tree_start->size() >= 2 && r.tree_goal->size() >= 2);
}

TEST(nested_finite_goal_filters_by_predicate) {
  auto s = planar();
  Planner planner(base());
  Config near{0.2, 0.1}, far{2.5, 2.5}, bad{0.2, -0.5};
  auto goal = std::make_shared<AnyOf>(std::vector<SetPtr>{
      std::make_shared<AllOf>(std::vector<SetPtr>{finite(*s, {bad, near}),
                                                  std::make_shared<PredicateSet>([](ConfigView q) { return q[1] > 0; }, "q1>0")}),
      finite(*s, {far})});
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), goal);
  PlanResult r = planner.solve(p, seeded(2));
  CHECK(r.success() && validate(p, planner.config(), r.path).all());
  // Two admissible goal roots, near (0,1) and far (1,0); which one the trees connect to depends on the
  // draws. The invariants: the filtered member (0,0) is never a root, provenance names the reached root.
  CHECK(r.goal_roots.explicit_candidates == 2 && r.goal_roots.roots == 2);
  CHECK(r.goal_source == (Provenance{0, 1}) || r.goal_source == (Provenance{1, 0}));
  const Config& reached = r.path.back();
  CHECK(s->distance(reached, r.goal_source == (Provenance{0, 1}) ? near : far) <= 1e-3);
}

TEST(wrapped_seam_output_is_continuous) {
  auto s = planar({true, false});
  Planner planner(base());
  auto p = problem(s, finite(*s, {{3.0, 0.0}}), finite(*s, {{-3.0, 0.0}}));
  PlanResult r = planner.solve(p, seeded(3));
  CHECK(r.success());
  Validation v = validate(p, planner.config(), r.path);
  CHECK(v.all());
  CHECK(r.path.back()[0] > 3.0);  // the goal re-expressed past +pi, 0.28 rad from the start
  CHECK_NEAR(s->distance(r.path.back(), Config{-3.0, 0.0}), 0.0, 1e-9);
}

TEST(rejection_only_constraint_is_respected) {
  auto s = planar();
  Planner planner(base());
  auto half = std::make_shared<PredicateSet>([](ConfigView q) { return q[1] >= -1e-9; }, "q1>=0");
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.3}}), half);
  PlanResult r = planner.solve(p, seeded(4));
  CHECK(r.success() && validate(p, planner.config(), r.path).all());
  CHECK(!half->supports(Capability::Projector));
}

TEST(projected_constraint_uses_the_projector) {
  // A band on joint 1 with a projector (clamp), as a TSR-induced set would provide.
  class Band1 final : public StateSet, public SetViolation, public SetProjector {
   public:
    bool contains(ConfigView q) const override { return std::fabs(q[1]) <= 0.3 + 1e-9; }
    double violation(ConfigView q) const override { return std::max(0.0, std::fabs(q[1]) - 0.3); }
    std::optional<Config> project(ConfigView, ConfigView q) const override {
      Config out = to_config(q);
      out[1] = std::clamp(out[1], -0.3, 0.3);
      return out;
    }
    const SetViolation* violator() const override { return this; }
    const SetProjector* projector() const override { return this; }
    std::string describe() const override { return "Band1"; }
  };
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.2}}), finite(*s, {{-2.0, -0.2}}), std::make_shared<Band1>());
  PlanResult r = planner.solve(p, seeded(5));
  CHECK(r.success() && validate(p, planner.config(), r.path).all());
}

TEST(all_of_finite_goal_is_enumerated) {
  auto s = planar();
  Planner planner(base());
  auto goal = std::make_shared<AllOf>(std::vector<SetPtr>{
      finite(*s, {{1.0, 0.5}, {2.0, 2.0}}), std::make_shared<PredicateSet>([](ConfigView q) { return q[0] < 1.5; })});
  CHECK(goal->is_finite() && goal->members().size() == 1);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), goal);
  PlanResult r = planner.solve(p, seeded(8));
  CHECK(r.success() && r.goal_source == Provenance{0} && r.goal_roots.roots == 1);
}

TEST(timeout_reports_before_any_iteration) {
  auto s = planar();
  PlannerConfig c = base();
  c.timeout_seconds = 1e-9;
  Planner planner(c);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  PlanResult r = planner.solve(p, seeded(9));
  CHECK(r.status == Status::Timeout && r.reason.starts_with("Timeout"));
  CHECK(r.iterations == 0 && r.tree_sizes == std::make_pair(1, 1));
}

TEST(cancellation_at_the_three_points) {
  auto s = planar();
  Planner planner(base());
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  auto token = std::make_shared<CancellationToken>();
  token->cancel();
  SolveOptions o = seeded(10);
  o.cancel = token;
  PlanResult r = planner.solve(p, o);
  CHECK(r.status == Status::Aborted && r.reason == "Aborted by user");  // finite sets: no draws polled
  CHECK(r.iterations == 0 && r.tree_sizes == std::make_pair(1, 1));

  auto sampled = problem(s, finite(*s, {{0.0, 0.0}}), std::make_shared<Band>(1.0, 2.0));
  PlanResult rs = planner.solve(sampled, o);
  CHECK(rs.status == Status::Aborted && rs.reason == "Aborted by user during goal root collection");
  CHECK(rs.tree_sizes == std::make_pair(1, 0));

  // Smoothing: cancel from inside the state validator once the trees have plenty of nodes; the poll before
  // the next smoothing attempt stops smoothing and the result is still a valid success.
  struct CancelLate final : StateValidator {
    std::shared_ptr<CancellationToken> t;
    mutable int calls = 0;
    bool is_valid(ConfigView) const override {
      if (++calls > 400) t->cancel();
      return true;
    }
  };
  auto late = std::make_shared<CancelLate>();
  late->t = std::make_shared<CancellationToken>();
  auto far = problem(s, finite(*s, {{-2.5, 0.5}}), finite(*s, {{2.5, -0.5}}), nullptr, late);
  PlannerConfig cs = base();
  cs.smoothing_iterations = 200;
  cs.smoothing_patience = 200;
  SolveOptions os = seeded(11);
  os.cancel = late->t;
  PlanResult rl = Planner(cs).solve(far, os);
  CHECK(rl.success() && validate(far, cs, rl.path).all());
  CHECK(late->t->cancelled());
}

TEST(unreachable_reports_max_iterations) {
  auto s = planar();
  PlannerConfig c = base();
  c.max_iterations = 300;
  Planner planner(c);
  auto wall = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{{{0.45, -kInf}, {0.55, kInf}}});
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.0}}), nullptr, wall);
  PlanResult r = planner.solve(p, seeded(11));
  CHECK(r.status == Status::MaxIterations && r.reason.starts_with("Max iterations"));
  CHECK(r.iterations == 300);
  for (const Node& n : r.tree_start->nodes()) CHECK(n.q[0] <= 0.45);
}

TEST(no_roots_is_an_exception_with_a_report) {
  auto s = planar();
  Planner planner(base());
  auto wall = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{{{-1.0, -1.0}, {1.0, 1.0}}});
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{2.0, 2.0}}), nullptr, wall);
  bool caught = false;
  try {
    planner.solve(p, seeded(0));
  } catch (const NoRoots& e) {
    caught = e.role == "start" && e.report.only_collisions() && e.report.explicit_rejected == 1 && e.report.details.size() == 1;
  }
  CHECK(caught);
  // Mixed reasons: one explicit start outside the space, one in collision.
  auto mixed = problem(s, finite(*s, {{9.0, 0.0}, {0.0, 0.0}}), finite(*s, {{2.0, 2.0}}), nullptr, wall);
  caught = false;
  try {
    planner.solve(mixed, seeded(0));
  } catch (const NoRoots& e) {
    caught = !e.report.only_collisions() && e.report.outside_space == 1 && e.report.in_collision == 1;
  }
  CHECK(caught);
  // An empty role is malformed input, as Python's ValueError.
  CHECK_THROWS(planner.solve(problem(s, std::make_shared<EmptySet>(), finite(*s, {{0.0, 0.0}})), seeded(0)), std::invalid_argument);
  // Neither finite nor sampleable.
  auto pred = std::make_shared<PredicateSet>([](ConfigView) { return true; });
  CHECK_THROWS(planner.solve(problem(s, pred, finite(*s, {{0.0, 0.0}})), seeded(0)), UnsupportedCapability);
}

TEST(custom_motion_validator_is_trusted_but_contract_checked) {
  auto s = planar();
  Planner planner(base());
  struct Liar final : MotionValidator {
    LocalMotion validate(ConfigView, ConfigView) const override { return LocalMotion{{}, true}; }  // reached with nothing
  };
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  p.motion_validator = std::make_shared<Liar>();
  CHECK_THROWS(planner.solve(p, seeded(0)), ContractError);

  // A restriction on top of the default keeps the discrete checks and still plans.
  auto q = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  q.motion_validator = std::make_shared<RestrictedMotionValidator>(
      planner.default_motion_validator(q), [](ConfigView a, ConfigView b) { return std::fabs(b[0] - a[0]) < 0.5; });
  PlanResult r = planner.solve(q, seeded(0));
  CHECK(r.success() && validate(q, planner.config(), r.path).all());
}

TEST(custom_sampler_replaces_free_space_targets) {
  struct AlwaysGoal final : SpaceSampler {
    Config sample(Rng&) const override { return {1.0, 0.5}; }
  };
  auto s = planar();
  PlannerConfig c = base();  // finite start and goal: neither tree tosses the P_sample coin
  c.smooth_path = false;
  Planner planner(c);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), finite(*s, {{1.0, 0.5}}));
  p.sampler = std::make_shared<AlwaysGoal>();
  PlanResult r = planner.solve(p, seeded(0));
  CHECK(r.success() && r.iterations == 1);
}

// CBiRRT's P_sample (#196). A full-height wall at joint 0 in [0.45, 0.55]; the goal is a band beyond it (unreachable)
// or a band on the start's side.
namespace {
std::shared_ptr<JointBoxObstacles> wall() {
  return std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{{{0.45, -kInf}, {0.55, kInf}}});
}
}  // namespace

TEST(p_sample_zero_adds_no_roots_during_the_search) {
  auto s = planar();
  PlannerConfig c = base();
  c.max_iterations = 200;
  c.num_tree_roots = 1;
  c.goal_sample_probability = 0.0;
  Planner planner(c);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), std::make_shared<Band>(1.0, 1.2), nullptr, wall());
  PlanResult r = planner.solve(p, seeded(3));
  CHECK(r.status == Status::MaxIterations && r.stats.search_roots == 0);
  int roots = 0;
  for (const Node& n : r.tree_goal->nodes()) roots += n.parent < 0;
  CHECK(roots == 1);
}

TEST(p_sample_one_adds_a_root_on_every_goal_turn_and_none_to_a_finite_start) {
  auto s = planar();
  PlannerConfig c = base();
  c.max_iterations = 200;
  c.num_tree_roots = 1;
  c.goal_sample_probability = 1.0;
  c.start_sample_probability = 1.0;  // the start is finite: its tree never tosses the coin
  Planner planner(c);
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), std::make_shared<Band>(1.0, 1.2), nullptr, wall());
  PlanResult r = planner.solve(p, seeded(3));
  CHECK(r.status == Status::MaxIterations);
  CHECK(r.stats.search_roots == 100);  // 100 goal turns, one admissible candidate per Band draw
  int goal_roots = 0, start_roots = 0;
  for (const Node& n : r.tree_goal->nodes()) {
    goal_roots += n.parent < 0;
    if (n.parent < 0) CHECK(n.q[0] >= 1.0 && n.q[0] <= 1.2);
  }
  CHECK(goal_roots == 101);  // its own turns only add roots; it still grows on the start's turns, by connecting
  for (const Node& n : r.tree_start->nodes()) start_roots += n.parent < 0;
  CHECK(start_roots == 1);
  CHECK(r.goal_roots.roots == 1);  // RootReport covers the collection before the search only
}

TEST(a_path_to_a_root_added_during_the_search_reports_its_provenance) {
  auto s = planar();
  PlannerConfig c = base();
  c.num_tree_roots = 1;
  c.goal_sample_probability = 0.5;
  Planner planner(c);
  auto goal = std::make_shared<AnyOf>(
      std::vector<SetPtr>{std::make_shared<Band>(1.0, 1.2), std::make_shared<Band>(-1.2, -1.0)},
      std::vector<double>{1.0, 1.0});
  auto p = problem(s, finite(*s, {{0.0, 0.0}}), goal, nullptr, wall());
  int checked = 0;
  for (std::uint64_t seed = 0; seed < 40; ++seed) {
    PlanResult r = planner.solve(p, seeded(seed));
    CHECK(r.success() && validate(p, planner.config(), r.path).all());
    CHECK(r.goal_source == Provenance{1});  // only the band on the start's side is reachable
    CHECK(r.path.back()[0] >= -1.2 - 1e-9 && r.path.back()[0] <= -1.0 + 1e-9);
    // The first goal root came from the unreachable band: the path ends on a root added during the search.
    if (r.tree_goal->nodes()[0].source == Provenance{0}) {
      CHECK(r.stats.search_roots > 0);
      ++checked;
    }
  }
  CHECK(checked > 0);
}

// #200: windings of one physical configuration share the validator's verdict, only when it declares invariance.
namespace {
struct CosValidator final : StateValidator {
  explicit CosValidator(bool invariant) : invariant(invariant) {}
  bool is_valid(ConfigView q) const override {
    ++calls;
    return std::cos(q[0]) > -0.5;  // depends on q[0] only through its angle
  }
  bool full_turn_invariant() const override { return invariant; }
  bool invariant;
  mutable int calls = 0;
};

// Each draw offers one angle in three windings (as SSIK enumerates them), keyed as the same physical configuration.
class Windings final : public StateSet, public SetSampler {
 public:
  bool contains(ConfigView q) const override { return std::fabs(q[1] - 1.0) < 1e-9; }
  std::vector<Sample> sample(Rng& rng) const override {
    const double a = -kPi + unit(rng) * 2.0 * kPi;
    std::vector<Sample> out;
    for (double w : {a, a + 2.0 * kPi, a - 2.0 * kPi}) {
      Config q{w, 1.0};
      std::vector<std::int64_t> key = full_turn_key(q, {true, false});
      out.push_back(Sample{std::move(q), {}, std::move(key)});
    }
    return out;
  }
  const SetSampler* sampler() const override { return this; }
  std::string describe() const override { return "Windings"; }
};
}  // namespace

TEST(full_turn_key_identifies_windings_and_only_windings) {
  CHECK(full_turn_key(Config{0.3, 1.0}, {true, false}) == full_turn_key(Config{0.3 + 2 * kPi, 1.0}, {true, false}));
  CHECK(full_turn_key(Config{0.3, 1.0}, {true, false}) != full_turn_key(Config{0.3, 1.0 + 2 * kPi}, {true, false}));
  CHECK(full_turn_key(Config{0.3, 1.0}, {true, false}) != full_turn_key(Config{0.31, 1.0}, {true, false}));
}

TEST(windings_share_a_verdict_only_when_the_validator_declares_invariance) {
  auto s = std::make_shared<JointSpace>(std::vector<double>{-7.0, -1.5}, std::vector<double>{7.0, 1.5});
  PlannerConfig c = base();
  c.num_tree_roots = 30;
  c.max_per_draw = 3;
  c.goal_sample_probability = 0.0;
  Planner planner(c);
  auto run = [&](bool invariant) {
    auto v = std::make_shared<CosValidator>(invariant);
    auto p = problem(s, finite(*s, {{0.0, 1.0}}), std::make_shared<Windings>(), nullptr, v);
    PlanResult r = planner.solve(p, seeded(5));
    return std::make_pair(r, v->calls);
  };
  auto [shared, shared_calls] = run(true);
  auto [unshared, unshared_calls] = run(false);
  CHECK(shared.success() && unshared.success());
  CHECK(shared.path == unshared.path && shared.iterations == unshared.iterations);  // identical but for the cost
  CHECK(shared.goal_roots.roots == unshared.goal_roots.roots);
  CHECK(shared.stats.state_checks == unshared.stats.state_checks);  // every candidate is still judged
  CHECK(shared.stats.reused_verdicts > 0 && unshared.stats.reused_verdicts == 0);
  CHECK(shared_calls < unshared_calls);
}

HARNESS_MAIN()
