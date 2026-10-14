// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <optional>
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

// Motion validation (#230): one implementation, used by the planner and by anyone checking its output. Mirrors
// sscbirrt.validation operation by operation.
//
// Motions are raw-linear: validate_motion(q0, q1) checks the straight line in raw joint values, the motion a runtime
// executes; a continuous joint is not wrapped. The guarantee is discrete: states at most policy.resolution apart
// (Euclidean, joint space) are checked, and nothing is claimed between them.
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
  std::optional<int> segment;      // the interval it lies on: 0 for a motion, the segment for a path; none for a state
  std::optional<double> fraction;  // i/n on that interval
  std::vector<Contact> contacts;   // when the validator names them
  std::optional<int> constraint;   // path constraint: the first violated member of an AllOf, else 0
  std::string detail;
};

struct ValidationReport {
  std::string subject;  // "state", "motion", "path"
  bool valid = false;
  int intervals = 0;    // 0 for a state, 1 for a motion, waypoints - 1 for a path
  long checks = 0;      // states checked, the invalid one included
  ValidationPolicy policy;
  std::string implementation;  // "sscbirrt X.Y.Z"
  std::optional<InvalidWitness> witness;  // present exactly when not valid
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

  const ValidationPolicy& policy() const { return policy_; }

 private:
  std::optional<InvalidWitness> witness(ConfigView q, std::optional<int> segment, std::optional<double> fraction) const;
  std::pair<long, std::optional<InvalidWitness>> segment(ConfigView q0, ConfigView q1, int index) const;
  ValidationReport report(const char* subject, int intervals, long checks, std::optional<InvalidWitness> w) const;
  void check_config(ConfigView q, const std::string& name) const;

  std::shared_ptr<const JointSpace> space_;
  std::shared_ptr<const StateValidator> validator_;
  std::shared_ptr<const StateSet> path_constraint_;
  ValidationPolicy policy_;
};

}  // namespace sscbirrt
