// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include <cmath>
#include <memory>

#include "harness.hpp"
#include "sscbirrt/planner.hpp"
#include "sscbirrt/trajectory.hpp"
#include "sscbirrt/validity.hpp"

using namespace sscbirrt;
using namespace sscbirrt::trajectory;

namespace {

struct Case {
  double x0, v0, x1, v1, vmax, amax;
};

Case draw(Rng& rng, bool moving) {
  const auto u = [&](double lo, double hi) { return lo + (hi - lo) * unit(rng); };
  Case c{};
  c.vmax = u(0.3, 3.0);
  c.amax = u(0.5, 8.0);
  c.x0 = u(-3.0, 3.0);
  c.x1 = u(-3.0, 3.0);
  c.v0 = moving ? u(-c.vmax, c.vmax) : 0.0;
  c.v1 = moving ? u(-c.vmax, c.vmax) : 0.0;
  return c;
}

std::vector<Config> dense_u(double step = 0.05) {  // three straight runs, two corners
  const std::vector<Config> corners{{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}};
  std::vector<Config> out{corners.front()};
  for (std::size_t i = 0; i + 1 < corners.size(); ++i) {
    const int n = static_cast<int>(std::ceil(1.0 / step));
    for (int k = 1; k <= n; ++k) {
      const double t = static_cast<double>(k) / n;
      out.push_back({corners[i][0] + t * (corners[i + 1][0] - corners[i][0]),
                     corners[i][1] + t * (corners[i + 1][1] - corners[i][1])});
    }
  }
  return out;
}

PlanningProblem problem(std::shared_ptr<const StateValidator> validator) {
  PlanningProblem p;
  p.space = std::make_shared<JointSpace>(std::vector<double>{-3.0, -3.0}, std::vector<double>{3.0, 3.0});
  p.validator = std::move(validator);
  return p;
}

Planner planner() {
  PlannerConfig c;
  c.step_size = 0.1;
  c.edge_resolution = 0.02;
  return Planner(c);
}

const std::vector<double> kV{1.0, 1.0}, kA{2.0, 2.0};

}  // namespace

TEST(min_time_is_valid_and_nothing_is_faster) {
  Rng rng(0);
  for (int i = 0; i < 3000; ++i) {
    const Case c = draw(rng, true);
    const auto r = min_time(c.x0, c.v0, c.x1, c.v1, c.vmax, c.amax);
    CHECK(r.has_value());
    CHECK(verify(*r, c.x1, c.v1, c.vmax, c.amax));
    for (double f : {0.9, 0.98}) CHECK(!fixed_time(c.x0, c.v0, c.x1, c.v1, f * r->duration(), c.vmax, c.amax));
  }
}

TEST(fixed_time_is_exact_and_complete_from_rest) {
  Rng rng(1);
  for (bool moving : {false, true}) {
    for (int i = 0; i < 2000; ++i) {
      const Case c = draw(rng, moving);
      const double T = min_time(c.x0, c.v0, c.x1, c.v1, c.vmax, c.amax)->duration();
      for (double f : {1.1, 1.5, 3.0}) {
        const auto r = fixed_time(c.x0, c.v0, c.x1, c.v1, f * T, c.vmax, c.amax);
        if (r) {
          CHECK(verify(*r, c.x1, c.v1, c.vmax, c.amax, f * T));
        } else {
          CHECK(moving);  // rest to rest, every longer duration is feasible
        }
      }
    }
  }
}

TEST(split_preserves_the_motion) {
  const Ramp r = *min_time(0.0, 0.5, 2.0, -0.3, 1.0, 2.0);
  const auto [a, b] = r.split(r.duration() * 0.37);
  CHECK_NEAR(a.duration() + b.duration(), r.duration(), 1e-12);
  CHECK_NEAR(a.end().first, b.x0, 1e-12);
  CHECK_NEAR(a.end().second, b.v0, 1e-12);
  CHECK_NEAR(b.end().first, r.end().first, 1e-12);
}

TEST(synchronized_joints_share_one_duration_within_their_limits) {
  Rng rng(2);
  const std::vector<double> vmax{1.0, 2.0, 0.5}, amax{3.0, 1.0, 2.0};
  const auto u = [&](double lo, double hi) { return lo + (hi - lo) * unit(rng); };
  int found = 0;
  for (int i = 0; i < 300; ++i) {
    Config q0(3), q1(3), v0(3), v1(3);
    for (int j = 0; j < 3; ++j) {
      q0[j] = u(-2, 2);
      q1[j] = u(-2, 2);
      v0[j] = u(-1, 1) * vmax[j];
      v1[j] = u(-1, 1) * vmax[j];
    }
    const auto piece = synchronize(q0, v0, q1, v1, vmax, amax);
    if (!piece) continue;
    ++found;
    for (int j = 0; j < 3; ++j) CHECK(verify(piece->ramps[j], q1[j], v1[j], vmax[j], amax[j], piece->duration()));
  }
  CHECK(found > 250);
}

TEST(initial_trajectory_is_at_rest_at_every_corner) {
  const Trajectory traj = initial_trajectory(coalesce(dense_u()), kV, kA);
  CHECK(traj.pieces.size() == 3);
  Config q, v;
  for (double t : traj.breaks()) {
    traj.state(t, q, v);
    CHECK(std::fabs(v[0]) < 1e-9 && std::fabs(v[1]) < 1e-9);
  }
  for (const Piece& p : traj.pieces) CHECK(p.validated == "planned");
  const Trajectory one = initial_trajectory({{0.5, 0.5}}, kV, kA);  // start is the goal
  CHECK(one.duration() == 0.0);
  one.state(0.0, q, v);
  CHECK(q == (Config{0.5, 0.5}));
}

TEST(splice_keeps_position_and_velocity_continuous) {
  const Trajectory traj = initial_trajectory(coalesce(dense_u()), kV, kA);
  const double t1 = 0.3 * traj.duration(), t2 = 0.8 * traj.duration();
  Config qa, va, qb, vb;
  traj.state(t1, qa, va);
  traj.state(t2, qb, vb);
  const auto piece = synchronize(qa, va, qb, vb, kV, kA);
  CHECK(piece.has_value());
  const Trajectory out = traj.splice(t1, t2, *piece);
  CHECK_NEAR(out.duration(), traj.duration() - (t2 - t1) + piece->duration(), 1e-9);
  Config q, v, q2, v2;
  for (double t : out.breaks()) {  // the two sides of every break agree
    out.state(t - 1e-9, q, v);
    out.state(t + 1e-9, q2, v2);
    for (int j = 0; j < 2; ++j) {
      CHECK_NEAR(q[j], q2[j], 1e-7);
      CHECK_NEAR(v[j], v2[j], 1e-7);
    }
  }
}

TEST(shortcuts_are_faster_at_rest_at_the_ends_and_deterministic) {
  const PlanningProblem p = problem(std::make_shared<AcceptAll>());
  ShortcutOptions o;
  o.iterations = 200;
  o.seed = 3;
  ShortcutReport rep;
  const Trajectory a = smooth_path_in_time(planner(), p, dense_u(), kV, kA, o, &rep);
  const Trajectory b = smooth_path_in_time(planner(), p, dense_u(), kV, kA, o);
  const Trajectory raw = initial_trajectory(coalesce(dense_u()), kV, kA);
  CHECK(a.duration() < raw.duration());
  CHECK(rep.accepted > 0 && rep.attempts.size() == 200);
  CHECK(a.duration() == b.duration());
  Config q, v;
  a.state(0.0, q, v);
  CHECK(q == (Config{0.0, 0.0}) && std::fabs(v[0]) < 1e-9 && std::fabs(v[1]) < 1e-9);
  a.state(a.duration(), q, v);
  CHECK_NEAR(q[0], 0.0, 1e-9);
  CHECK_NEAR(q[1], 1.0, 1e-9);
  CHECK(std::fabs(v[0]) < 1e-9 && std::fabs(v[1]) < 1e-9);
  bool shortcut_piece = false;
  for (const Piece& piece : a.pieces) {
    CHECK(piece.validated == "planned" || piece.validated == "shortcut");
    shortcut_piece = shortcut_piece || piece.validated == "shortcut";
    for (std::size_t j = 0; j < 2; ++j) CHECK(verify(piece.ramps[j], piece.ramps[j].end().first, piece.ramps[j].end().second, 1.0, 2.0));
  }
  CHECK(shortcut_piece);
}

TEST(a_shortcut_never_gets_deeper_into_an_obstacle_than_the_resolution) {
  // A block filling the inside of the U except a margin around the path.
  const PlanningProblem p = problem(std::make_shared<JointBoxObstacles>(
      std::vector<JointBoxObstacles::Box>{{{0.1, 0.1}, {0.9, 0.9}}}));
  ShortcutOptions o;
  o.iterations = 300;
  ShortcutReport rep;
  const Trajectory traj = smooth_path_in_time(planner(), p, dense_u(), kV, kA, o, &rep);
  const double res = 0.02;
  Config q, v;
  for (int i = 0; i <= 4000; ++i) {
    traj.state(traj.duration() * i / 4000.0, q, v);
    CHECK(!(0.1 + res < q[0] && q[0] < 0.9 - res && 0.1 + res < q[1] && q[1] < 0.9 - res));
  }
  bool collided = false;
  for (const Attempt& at : rep.attempts) collided = collided || at.rejected.find("in collision") != std::string::npos;
  CHECK(collided);
}

HARNESS_MAIN()
