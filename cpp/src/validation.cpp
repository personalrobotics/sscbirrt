// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// Compiled without floating-point contraction (CMakeLists.txt), so every sample rounds as sscbirrt.validation's does.
#include "sscbirrt/validation.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace sscbirrt {

#ifndef SSCBIRRT_VERSION
#define SSCBIRRT_VERSION "unknown"
#endif

const char* version() { return SSCBIRRT_VERSION; }

void ValidationPolicy::validate() const {
  if (!(std::isfinite(resolution) && resolution > 0.0)) {
    throw std::invalid_argument("resolution must be a positive finite number, got " + std::to_string(resolution));
  }
}

std::vector<std::pair<std::string, std::string>> ValidationPolicy::parameters() const {
  return {{"endpoints", "checked"}, {"interpolation", "raw_linear"}, {"rounding", "1e-09"}};
}

std::string ValidationPolicy::summary() const {
  std::ostringstream os;
  os << "Valid at states at most " << resolution << " apart (Euclidean, joint space, to a relative 1e-09"
     << ") along each raw-linear motion, endpoints included; nothing is claimed between them.";
  return os.str();
}

const char* failure_name(Failure f) {
  switch (f) {
    case Failure::PositionLimits: return "position_limits";
    case Failure::Collision: return "collision";
    case Failure::PathConstraint: return "path_constraint";
    case Failure::Unsupported: return "unsupported";
  }
  return "?";
}

int subdivisions(double distance, double resolution) {
  return std::max(1, static_cast<int>(std::ceil(distance / resolution - kRounding)));
}

std::optional<std::pair<Failure, std::string>> classify(const JointSpace& space, const StateValidator& validator,
                                                        const StateSet* path_constraint, ConfigView q,
                                                        std::optional<bool> valid) {
  if (auto why = space.why_invalid(q)) return std::pair{Failure::PositionLimits, "outside joint space (" + *why + ")"};
  if (!(valid ? *valid : validator.is_valid(q))) return std::pair{Failure::Collision, std::string("in collision")};
  if (path_constraint && !path_constraint->contains(q)) {
    return std::pair{Failure::PathConstraint, std::string("violates path constraints")};
  }
  return std::nullopt;
}

Validator::Validator(std::shared_ptr<const JointSpace> space, std::shared_ptr<const StateValidator> validator,
                     std::shared_ptr<const StateSet> path_constraint, ValidationPolicy policy)
    : space_(std::move(space)),
      validator_(std::move(validator)),
      path_constraint_(std::move(path_constraint)),
      policy_(policy) {
  if (!space_) throw std::invalid_argument("Validator needs a space");
  if (!validator_) throw std::invalid_argument("Validator needs a state validator");
  policy_.validate();
}

Validator Validator::for_problem(const PlanningProblem& problem, const PlannerConfig& config) {
  if (problem.motion_validator) {
    throw std::invalid_argument(
        "the problem's motion validator replaces the default edge rule, so no ValidationPolicy describes its motions");
  }
  return Validator(problem.space, problem.validator, problem.path_constraint, ValidationPolicy{config.resolution()});
}

void Validator::check_config(ConfigView q, const std::string& name) const {
  if (static_cast<int>(q.size()) != space_->dof()) {
    throw std::invalid_argument(name + ": length " + std::to_string(q.size()) + " != " + std::to_string(space_->dof()));
  }
  for (double v : q) {
    if (!std::isfinite(v)) throw std::invalid_argument(name + ": non-finite entries");
  }
}

std::optional<InvalidWitness> Validator::witness(ConfigView q, std::optional<int> segment,
                                                 std::optional<double> fraction) const {
  auto why = classify(*space_, *validator_, path_constraint_.get(), q);
  if (!why) return std::nullopt;
  InvalidWitness w;
  w.failure = why->first;
  w.detail = std::move(why->second);
  w.configuration = to_config(q);
  w.segment = segment;
  w.fraction = fraction;
  if (w.failure == Failure::Collision) w.contacts = validator_->contacts(q);
  if (w.failure == Failure::PathConstraint) {
    w.constraint = 0;
    if (const auto* all = dynamic_cast<const AllOf*>(path_constraint_.get())) {
      for (std::size_t i = 0; i < all->children().size(); ++i) {
        if (!all->children()[i]->contains(q)) {
          w.constraint = static_cast<int>(i);
          break;
        }
      }
    }
  }
  return w;
}

std::pair<long, std::optional<InvalidWitness>> Validator::segment(ConfigView q0, ConfigView q1, int index) const {
  const std::size_t dof = q0.size();
  Config d(dof);
  double dist = 0.0;
  for (std::size_t k = 0; k < dof; ++k) {
    d[k] = q1[k] - q0[k];
    dist += d[k] * d[k];
  }
  const int n = subdivisions(std::sqrt(dist), policy_.resolution);
  Config q(dof);
  for (int i = 1; i <= n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    if (i == n) {
      q.assign(q1.begin(), q1.end());
    } else {
      for (std::size_t k = 0; k < dof; ++k) q[k] = q0[k] + t * d[k];
    }
    if (auto w = witness(q, index, t)) return {i, std::move(w)};
  }
  return {n, std::nullopt};
}

ValidationReport Validator::report(const char* subject, int intervals, long checks,
                                   std::optional<InvalidWitness> w) const {
  ValidationReport r;
  r.subject = subject;
  r.valid = !w.has_value();
  r.intervals = intervals;
  r.checks = checks;
  r.policy = policy_;
  r.implementation = std::string("sscbirrt ") + version();
  r.witness = std::move(w);
  return r;
}

ValidationReport Validator::validate_state(ConfigView q) const {
  check_config(q, "q");
  return report("state", 0, 1, witness(q, std::nullopt, std::nullopt));
}

ValidationReport Validator::validate_motion(ConfigView q0, ConfigView q1) const {
  check_config(q0, "q0");
  check_config(q1, "q1");
  auto w = witness(q0, 0, 0.0);
  long checks = 1;
  if (!w) {
    auto [more, w1] = segment(q0, q1, 0);
    checks += more;
    w = std::move(w1);
  }
  return report("motion", 1, checks, std::move(w));
}

ValidationReport Validator::validate_path(const std::vector<Config>& path) const {
  if (path.empty()) throw std::invalid_argument("path: empty");
  for (std::size_t i = 0; i < path.size(); ++i) check_config(path[i], "path[" + std::to_string(i) + "]");
  const bool many = path.size() > 1;
  auto w = witness(path[0], many ? std::optional<int>(0) : std::nullopt, many ? std::optional<double>(0.0) : std::nullopt);
  long checks = 1;
  for (std::size_t k = 0; k + 1 < path.size() && !w; ++k) {
    auto [more, wk] = segment(path[k], path[k + 1], static_cast<int>(k));
    checks += more;
    w = std::move(wk);
  }
  return report("path", static_cast<int>(path.size()) - 1, checks, std::move(w));
}

}  // namespace sscbirrt
