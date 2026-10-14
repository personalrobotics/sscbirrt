// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

// Bounded-acceleration trajectory smoothing with time-optimized shortcuts (#214, not yet public API).
//
// The native form of sscbirrt._trajectory, which is the reference: every deterministic function here computes what
// its Python namesake computes, and the shortcut loop makes the same decisions; only the random stream differs
// (unit() over mt19937_64 against numpy). After Hauser and Ng-Thow-Hing, "Fast smoothing of manipulator
// trajectories using optimal bounded-acceleration shortcuts" (ICRA 2010).
//
// - A ramp is one joint's motion over an interval: constant-acceleration segments.
// - A piece is one interval in which every joint follows its own ramp, all of the same duration.
// - A trajectory is a sequence of pieces. Position and velocity are continuous; acceleration is piecewise constant.
//
// Every ramp is verified independently of the formulas that built it, so a mistake in the algebra can only reject
// a shortcut, never admit an invalid one.

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sscbirrt/cancel.hpp"
#include "sscbirrt/motion.hpp"
#include "sscbirrt/planner.hpp"
#include "sscbirrt/problem.hpp"
#include "sscbirrt/types.hpp"
#include "sscbirrt/validation.hpp"

namespace sscbirrt::trajectory {

struct Segment {
  double duration = 0.0;
  double acceleration = 0.0;
};

// One joint's motion: start state and constant-acceleration segments.
struct Ramp {
  double x0 = 0.0;
  double v0 = 0.0;
  std::vector<Segment> segments;

  double duration() const;
  std::array<double, 3> state(double t) const;  // position, velocity, acceleration at t (clamped to the ramp)
  double position(double t) const;              // as Python's vectorized Ramp.evaluate (clamped to the ramp)
  std::pair<double, double> end() const;
  std::pair<Ramp, Ramp> split(double t) const;  // the ramp before and after t
};

// The ramp ends at (x1, v1), stays within |v| <= vmax and |a| <= amax, and lasts `duration` if given.
bool verify(const Ramp& ramp, double x1, double v1, double vmax, double amax,
            std::optional<double> duration = std::nullopt);

// The time-optimal ramp from (x0, v0) to (x1, v1), or nullopt.
std::optional<Ramp> min_time(double x0, double v0, double x1, double v1, double vmax, double amax);

// A ramp lasting exactly T, least acceleration first, or nullopt ("not at this T").
std::optional<Ramp> fixed_time(double x0, double v0, double x1, double v1, double T, double vmax, double amax);

// One interval in which every joint follows its own ramp. `validated` records how its geometry was checked:
// "planned" (the planner's own edge) or "shortcut" (checked by the smoother through the problem's boundary).
struct Piece {
  std::vector<Ramp> ramps;
  std::string validated;

  double duration() const;
  std::pair<Piece, Piece> split(double t) const;
};

// The fastest common duration at which every joint can move between its boundary states (2% steps up to 4x the
// slowest joint's minimum time), or nullopt.
std::optional<Piece> synchronize(const Config& q0, const Config& v0, const Config& q1, const Config& v1,
                                 const std::vector<double>& vmax, const std::vector<double>& amax);

struct Trajectory {
  std::vector<Piece> pieces;
  std::vector<double> velocity_limits;
  std::vector<double> acceleration_limits;
  std::vector<Config> path;  // the validated path it was timed from, if any (#230); splice keeps it

  std::vector<double> breaks() const;
  double duration() const;
  void state(double t, Config& q, Config& v) const;     // clamped to [0, duration]
  Trajectory splice(double t1, double t2, Piece piece) const;  // [t1, t2] replaced by piece
};

// The path's true corners: endpoints and every waypoint where the direction turns by at least `tolerance`.
std::vector<Config> coalesce(const std::vector<Config>& path, double tolerance = 1e-9);

// Rest-to-rest motion along the chord q0 -> q1, exactly on the chord. Validated "planned".
Piece straight_line(const Config& q0, const Config& q1, const std::vector<double>& vmax,
                    const std::vector<double>& amax);

// The planned polyline at rest at every corner: exactly the planned geometry.
Trajectory initial_trajectory(const std::vector<Config>& corners, const std::vector<double>& vmax,
                              const std::vector<double>& amax);

// Positions along a piece no farther apart than `resolution` in joint space.
std::vector<Config> piece_samples(const Piece& piece, double resolution, const std::vector<double>& vmax);

// When shortcutting stops, and how it draws. The defaults equal sscbirrt._trajectory.ShortcutOptions' (a test holds
// them equal). Every rule but time_budget depends only on the trajectory, so a seed gives the same result anywhere.
struct ShortcutOptions {
  int iterations = 400;  // attempts at most: a deterministic backstop
  // Converged: stop once the last convergence_window attempts shortened the trajectory by less than
  // convergence_tolerance of its duration. nullopt disables the rule.
  std::optional<int> convergence_window = 40;
  double convergence_tolerance = 0.005;
  std::optional<int> patience;        // stop after this many attempts without improvement
  std::optional<double> time_budget;  // seconds; the one machine-dependent rule
  std::uint64_t seed = 0;
  double min_interval = 1e-3;  // seconds; shorter windows are not worth an attempt
  std::shared_ptr<const CancellationToken> cancel;  // polled before each attempt; may be null
};

struct Attempt {
  int iteration = 0;
  double t1 = 0.0, t2 = 0.0;
  std::optional<double> gain;
  std::string rejected;  // empty when accepted
  bool accepted = false;
  double elapsed = 0.0;  // seconds from the start of the loop to the end of this attempt (a diagnostic)
};

struct ShortcutReport {
  std::vector<Attempt> attempts;
  int accepted = 0;
  double seconds = 0.0;
  std::string stopped;  // "iterations", "converged", "patience", "time_budget" or "aborted"
};

// Why q is inadmissible, or nullopt.
using WhyInadmissible = std::function<std::optional<std::string>(ConfigView)>;

// Why a sampled motion is inadmissible, or nullopt: every sample admissible, every chord accepted by the motion
// validator and, when `recheck_returned`, every configuration it returns admissible (the planner's edge rule).
std::optional<std::string> validate_samples(const std::vector<Config>& points, const MotionValidator& motion,
                                            const WhyInadmissible& why, bool recheck_returned);

// Why splicing `piece` into [t1, t2] is refused, or nullopt: its start state, the new piece, then the two pieces the
// splice cuts, through validator.check_piece (#230). A cut piece still on the source path is not re-checked.
std::optional<std::string> accepts(const Trajectory& traj, double t1, double t2, const Piece& piece,
                                   const Validator& validator, const PathEvidence* evidence);

// Randomized time-optimized shortcuts. Deterministic for a seed and an iteration budget. With `validator`, a shortcut
// is accepted only if `accepts` says so; otherwise its samples go through `motion` and `why` (a custom motion
// validator's problem).
Trajectory shortcut(Trajectory traj, const MotionValidator& motion, const WhyInadmissible& why, bool recheck_returned,
                    double resolution, const ShortcutOptions& options, ShortcutReport* report = nullptr,
                    const Validator* validator = nullptr);

// The whole smoother on a planned path: coalesce, the initial trajectory (with `path` as its source path), then
// shortcuts accepted by Validator::for_problem's trajectory rule, or, for a problem with a custom motion validator,
// validated through it at the planner's resolution.
Trajectory smooth_path_in_time(const Planner& planner, const PlanningProblem& problem, const std::vector<Config>& path,
                               const std::vector<double>& vmax, const std::vector<double>& amax,
                               const ShortcutOptions& options, ShortcutReport* report = nullptr);

}  // namespace sscbirrt::trajectory
