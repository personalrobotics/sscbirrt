// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sscbirrt/config.hpp"
#include "sscbirrt/problem.hpp"
#include "sscbirrt/sets.hpp"
#include "sscbirrt/space.hpp"
#include "sscbirrt/types.hpp"
#include "sscbirrt/validity.hpp"

namespace sscbirrt {

namespace trajectory {
struct Piece;
struct Trajectory;
}  // namespace trajectory

// Motion validation (#230): one implementation, used by the planner and by anyone checking its output. Mirrors
// sscbirrt.validation operation by operation.
//
// Motions are raw-linear: validate_motion(q0, q1) checks the straight line in raw joint values, the motion a runtime
// executes; a continuous joint is not wrapped. The guarantee is discrete: states at most policy.resolution apart
// (Euclidean, joint space) are checked, and nothing is claimed between them.
//
// A timed trajectory is checked over its whole law (validate_trajectory). A piece whose trace lies on the
// trajectory's source path is checked at that path's own states within its span; any other piece at uniform time
// steps short enough that no joint-space step exceeds the resolution, plus every bounded joint's out-of-limits
// turning point. A trajectory that breaks its own invariants throws MalformedTrajectory.
//
// A Validator borrows its components and is not thread-safe (a SceneValidator owns one mjData): one per thread.
// Nothing is cached, so no earlier call can change a later answer.

// The sscbirrt version this core was built as, e.g. "3.8.0".
const char* version();

// What a validator checks and what a valid answer guarantees: today's edge rule, with no hidden setting.
struct ValidationPolicy {
  double resolution = 0.05;  // largest joint-space step between checked states

  static constexpr const char* name = "joint_l2_subdivision";
  static constexpr const char* mechanism = "joint_subdivision";
  static constexpr const char* guarantee = "discrete";
  static constexpr double ambiguity_band_m = 0.0;

  void validate() const;  // throws std::invalid_argument unless resolution is positive and finite
  std::vector<std::pair<std::string, std::string>> parameters() const;  // sorted
  std::string summary() const;
};

enum class Failure { PositionLimits, Collision, PathConstraint, Unsupported };
const char* failure_name(Failure f);  // "position_limits", "collision", "path_constraint", "unsupported"

// The first invalid state found, where it is, and why.
struct InvalidWitness {
  Failure failure = Failure::Collision;
  Config configuration;
  std::optional<int> segment;      // 0 for a motion, the segment for a path, the piece for a trajectory; none for a state
  std::optional<double> fraction;  // i/n on that interval, for a motion or path
  std::optional<double> time_s;    // for a trajectory
  std::vector<Contact> contacts;   // when the validator names them
  std::optional<int> constraint;   // path constraint: the first violated member of an AllOf, else 0
  std::string detail;
};

struct ValidationReport {
  std::string subject;  // "state", "motion", "path", "trajectory"
  bool valid = false;
  int intervals = 0;    // 0 for a state, 1 for a motion, waypoints - 1 for a path, pieces for a trajectory
  long checks = 0;      // states checked, the invalid one included
  ValidationPolicy policy;
  std::string implementation;  // "sscbirrt X.Y.Z"
  std::optional<InvalidWitness> witness;  // present exactly when not valid
  std::optional<double> duration_s;       // a trajectory's
};

// A trajectory that breaks its own invariants: shapes, finiteness, durations, continuity, limits, rest at both ends.
class MalformedTrajectory : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

inline constexpr const char* kLaw = "piecewise_constant_acceleration";
inline constexpr double kOnPath = 1e-8;  // a trajectory point this near the source path (relative, >= 1) lies on it

// The indices 0..n-1 of the states along a motion in van der Corput order (sscbirrt.validation.spread_order): level
// by level the positions ceil(j n / 2^L) for odd j, repeats skipped, any position left appended.
std::vector<int> spread_order(int n);

// The trajectory's own invariants, or MalformedTrajectory naming the first one broken.
void well_formed(const trajectory::Trajectory& traj, int dof);

// The reference joint of a piece moving monotonically along the straight line a -> b (every ramp the reference's
// scaled by (b_j - a_j) / (b_r - a_r)), or nullopt.
std::optional<int> straight_reference(const trajectory::Piece& piece, const Config& a, const Config& b);

// The states validate_path checks on a source path, each with its path parameter k + i/n.
class PathEvidence {
 public:
  PathEvidence(const std::vector<Config>& path, double resolution);
  std::optional<double> locate(const Config& x) const;  // the first point of the path within kOnPath of x
  // For a piece moving forward along a straight line on the path: the states after its start (the path's strictly
  // inside its span, then its end) and its reference joint.
  std::optional<std::pair<std::vector<std::pair<double, Config>>, int>> straight_span(const trajectory::Piece& piece) const;

 private:
  std::vector<Config> nodes_;
  std::vector<std::pair<double, Config>> states_;
};

// A length within this relative tolerance of a whole number of resolutions counts as that number: the planner stores
// every edge sample as a node, so a path's pieces are often exactly one resolution long, and recomputed from the nodes
// one may come out a rounding error longer.
inline constexpr double kRounding = 1e-9;

// Intervals a straight motion of length `distance` is split into: max(1, ceil(distance / resolution - kRounding)).
int subdivisions(double distance, double resolution);

// The one state rule: why q may not appear on a path, or nullopt. In order: joint space (limits), validator, path
// constraint. `valid` is the validator's verdict when already known (#200). The planner's admissibility uses it.
std::optional<std::pair<Failure, std::string>> classify(const JointSpace& space, const StateValidator& validator,
                                                        const StateSet* path_constraint, ConfigView q,
                                                        std::optional<bool> valid = std::nullopt);

class Validator {
 public:
  Validator(std::shared_ptr<const JointSpace> space, std::shared_ptr<const StateValidator> validator,
            std::shared_ptr<const StateSet> path_constraint = nullptr, ValidationPolicy policy = {});

  // The validator a planner with `config` applies to `problem`. Throws std::invalid_argument for a problem with a
  // custom motion validator: it replaces the edge rule, so no ValidationPolicy describes its motions.
  static Validator for_problem(const PlanningProblem& problem, const PlannerConfig& config);

  // Malformed input (wrong length, non-finite values, an empty path) throws std::invalid_argument.
  ValidationReport validate_state(ConfigView q) const;
  ValidationReport validate_motion(ConfigView q0, ConfigView q1) const;
  ValidationReport validate_path(const std::vector<Config>& path) const;
  // Its start, then every piece in time order; throws MalformedTrajectory. Uses traj.path, when not empty, as the
  // source path.
  ValidationReport validate_trajectory(const trajectory::Trajectory& traj) const;

  // One piece's states after its start: (states checked, the first invalid one in time order). With `spread`, the
  // same states in van der Corput order, returning the first invalid one found: the smoother needs only the verdict.
  std::pair<long, std::optional<InvalidWitness>> check_piece(const trajectory::Piece& piece, int index, double t0,
                                                             const std::vector<double>& vmax,
                                                             const PathEvidence* evidence, bool spread = false) const;
  std::optional<InvalidWitness> check_state(ConfigView q) const { return witness(q, std::nullopt, std::nullopt); }

  const ValidationPolicy& policy() const { return policy_; }

 private:
  std::optional<InvalidWitness> witness(ConfigView q, std::optional<int> segment, std::optional<double> fraction,
                                        std::optional<double> time_s = std::nullopt) const;
  std::vector<double> time_steps(const trajectory::Piece& piece, const std::vector<double>& vmax) const;
  std::pair<long, std::optional<InvalidWitness>> segment(ConfigView q0, ConfigView q1, int index) const;
  ValidationReport report(const char* subject, int intervals, long checks, std::optional<InvalidWitness> w) const;
  void check_config(ConfigView q, const std::string& name) const;

  std::shared_ptr<const JointSpace> space_;
  std::shared_ptr<const StateValidator> validator_;
  std::shared_ptr<const StateSet> path_constraint_;
  ValidationPolicy policy_;
};

}  // namespace sscbirrt
