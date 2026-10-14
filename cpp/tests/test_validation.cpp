// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include <cmath>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>

#include "harness.hpp"
#include "sscbirrt/planner.hpp"
#include "sscbirrt/validation.hpp"

using namespace sscbirrt;

namespace {
const double inf = std::numeric_limits<double>::infinity();
auto plane(std::vector<bool> angular = {}) {
  return std::make_shared<JointSpace>(std::vector<double>{-2.0, -2.0}, std::vector<double>{2.0, 2.0}, angular);
}
auto wall(double lo, double hi) {  // sscbirrt.testing.Wall(axis=0, lo, hi)
  return std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{{{lo, -inf}, {hi, inf}}});
}
}  // namespace

TEST(a_valid_motion_counts_its_states) {
  Validator v(plane(), std::make_shared<AcceptAll>(), nullptr, ValidationPolicy{0.1});
  const ValidationReport r = v.validate_motion(Config{0.0, 0.0}, Config{1.0, 0.0});
  CHECK(r.valid && r.subject == "motion" && r.intervals == 1 && r.checks == 11 && !r.witness);
  CHECK(r.implementation == std::string("sscbirrt ") + version());
}

TEST(the_first_invalid_state_is_located) {
  Validator v(plane(), wall(0.45, 0.55), nullptr, ValidationPolicy{0.1});
  const ValidationReport r = v.validate_path({{0.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
  CHECK(!r.valid && r.intervals == 2 && r.checks == 16);
  const InvalidWitness& w = *r.witness;
  CHECK(w.failure == Failure::Collision && *w.segment == 1 && *w.fraction == 0.5 && w.detail == "in collision");
  CHECK_NEAR(w.configuration[0], 0.5, 1e-12);
  CHECK(!v.validate_path({{0.5, 0.0}}).witness->segment);  // a one-waypoint path is a state
}

TEST(limits_and_constraint_members_are_named) {
  auto s = plane();
  CHECK(Validator(s, std::make_shared<AcceptAll>()).validate_state(Config{2.5, 0.0}).witness->failure ==
        Failure::PositionLimits);
  auto all = std::make_shared<AllOf>(std::vector<SetPtr>{std::make_shared<FiniteSet>(std::vector<Config>{{0, 0}, {1, 0}}, 0.5),
                                                          std::make_shared<FiniteSet>(std::vector<Config>{{0, 0}}, 0.5)});
  Validator v(s, std::make_shared<AcceptAll>(), all);
  CHECK(v.validate_state(Config{0.0, 0.0}).valid);
  const auto w = v.validate_state(Config{1.0, 0.0}).witness;
  CHECK(w->failure == Failure::PathConstraint && *w->constraint == 1);
}

TEST(motions_are_raw_linear) {
  auto s = plane({true, false});
  const double a = 0.75 * std::numbers::pi;
  CHECK(!Validator(s, wall(-0.1, 0.1)).validate_motion(Config{-a, 0.0}, Config{a, 0.0}).valid);  // the long raw line
  CHECK(Validator(s, wall(2.9, 3.3)).validate_motion(Config{-a, 0.0}, Config{a, 0.0}).valid);    // not the short way
}

TEST(malformed_input_throws) {
  Validator v(plane(), std::make_shared<AcceptAll>());
  CHECK_THROWS(v.validate_state(Config{0.0}), std::invalid_argument);
  CHECK_THROWS(v.validate_motion(Config{0.0, 0.0}, Config{0.0, std::nan("")}), std::invalid_argument);
  CHECK_THROWS(v.validate_path({}), std::invalid_argument);
  CHECK_THROWS(Validator(plane(), std::make_shared<AcceptAll>(), nullptr, ValidationPolicy{0.0}), std::invalid_argument);
}

TEST(the_planner_returns_only_a_validated_path) {
  PlanningProblem p;
  p.space = plane();
  p.start = std::make_shared<FiniteSet>(std::vector<Config>{{-1.0, 0.0}});
  p.goal = std::make_shared<FiniteSet>(std::vector<Config>{{1.0, 0.0}});
  p.validator = std::make_shared<JointBoxObstacles>(std::vector<JointBoxObstacles::Box>{{{-0.1, -1.0}, {0.1, 1.0}}});
  PlannerConfig cfg;
  cfg.edge_resolution = 0.03;
  SolveOptions o;
  o.seed = 1;
  const PlanResult r = Planner(cfg).solve(p, o);
  CHECK(r.success() && r.validation && r.validation->valid);
  const ValidationReport again = Validator::for_problem(p, cfg).validate_path(r.path);
  CHECK(again.valid && again.checks == r.validation->checks && again.policy.resolution == 0.03);

  p.motion_validator = Planner(cfg).default_motion_validator(p);
  CHECK_THROWS(Validator::for_problem(p, cfg), std::invalid_argument);
  CHECK(!Planner(cfg).solve(p, o).validation);  // a custom motion validator owns its motions
}

TEST(a_path_the_validator_rejects_is_not_returned) {
  // A box over raw values on a continuous joint is not full-turn invariant: the trees meet across +-pi with valid
  // wrapped edges, and unwrapping the path moves waypoints a turn up into the raw box (as in test_validation.py).
  PlanningProblem p;
  p.space = plane({true, false});
  p.start = std::make_shared<FiniteSet>(std::vector<Config>{{3.0, 0.0}});
  p.goal = std::make_shared<FiniteSet>(std::vector<Config>{{-3.0, 0.0}});
  p.validator = wall(3.25, 3.3);
  PlannerConfig cfg;
  cfg.smooth_path = false;
  SolveOptions o;
  o.seed = 0;
  const PlanResult r = Planner(cfg).solve(p, o);
  CHECK(r.status == Status::InvalidResult && r.path.empty());
  CHECK(r.reason.rfind("Returned path failed final validation: collision at segment", 0) == 0);
  CHECK(r.validation && r.validation->witness->configuration[0] > 3.25);
}

HARNESS_MAIN()
