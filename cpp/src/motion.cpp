// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/motion.hpp"

#include <cmath>
#include <stdexcept>

#include "sscbirrt/validation.hpp"
#include "sscbirrt/validity.hpp"

namespace sscbirrt {

JointBoxObstacles::JointBoxObstacles(std::vector<Box> boxes) : boxes_(std::move(boxes)) {
  for (const Box& b : boxes_) {
    if (b.lo.size() != b.hi.size() || b.lo.empty()) throw std::invalid_argument("box lo and hi must be nonempty and equal length");
  }
}

bool JointBoxObstacles::is_valid(ConfigView q) const {
  for (const Box& b : boxes_) {
    if (b.lo.size() != q.size()) throw std::invalid_argument("box dimension does not match the configuration");
    bool inside = true;
    for (std::size_t i = 0; i < q.size() && inside; ++i) inside = b.lo[i] < q[i] && q[i] < b.hi[i];
    if (inside) return false;
  }
  return true;
}

DiscreteMotionValidator::DiscreteMotionValidator(std::shared_ptr<const JointSpace> space,
                                                 std::function<bool(ConfigView)> is_admissible, double resolution)
    : space_(std::move(space)), is_admissible_(std::move(is_admissible)), resolution_(resolution) {
  if (!space_) throw std::invalid_argument("DiscreteMotionValidator needs a space");
  if (!is_admissible_) throw std::invalid_argument("DiscreteMotionValidator needs an admissibility predicate");
  if (!(resolution_ > 0.0)) throw std::invalid_argument("resolution must be positive");
}

LocalMotion DiscreteMotionValidator::validate(ConfigView q_from, ConfigView q_to) const {
  if (!space_->contains(q_to)) return LocalMotion{};
  const Config d = space_->direction(q_from, q_to);
  double dist = 0.0;
  for (double v : d) dist += v * v;
  dist = std::sqrt(dist);
  if (dist == 0.0) return LocalMotion{{}, true};
  const int n = subdivisions(dist, resolution_);  // the validator's subdivision (#230)
  LocalMotion out;
  for (int i = 1; i <= n; ++i) {
    Config q;
    if (i == n) {
      q = to_config(q_to);
    } else {
      q.resize(d.size());
      const double t = static_cast<double>(i) / static_cast<double>(n);
      for (std::size_t k = 0; k < d.size(); ++k) q[k] = q_from[k] + t * d[k];
    }
    if (!is_admissible_(q)) return out;  // reached stays false; the admissible prefix is kept
    out.configs.push_back(std::move(q));
  }
  out.reached = true;
  return out;
}

std::optional<std::vector<Config>> DiscreteMotionValidator::validate_whole(ConfigView q_from, ConfigView q_to) const {
  if (!space_->contains(q_to)) return std::nullopt;
  const Config d = space_->direction(q_from, q_to);
  double dist = 0.0;
  for (double v : d) dist += v * v;
  dist = std::sqrt(dist);
  if (dist == 0.0) return std::vector<Config>{};
  const int n = subdivisions(dist, resolution_);
  std::vector<Config> configs(static_cast<std::size_t>(n));
  for (int i = 1; i <= n; ++i) {
    Config& q = configs[static_cast<std::size_t>(i) - 1];
    if (i == n) {
      q = to_config(q_to);
    } else {
      q.resize(d.size());
      const double t = static_cast<double>(i) / static_cast<double>(n);
      for (std::size_t k = 0; k < d.size(); ++k) q[k] = q_from[k] + t * d[k];
    }
  }
  for (const int k : spread_order(n)) {
    if (!is_admissible_(configs[static_cast<std::size_t>(k)])) return std::nullopt;
  }
  return configs;
}

RestrictedMotionValidator::RestrictedMotionValidator(std::shared_ptr<const MotionValidator> base,
                                                     std::function<bool(ConfigView, ConfigView)> accepts)
    : base_(std::move(base)), accepts_(std::move(accepts)) {
  if (!base_) throw std::invalid_argument("RestrictedMotionValidator needs a base validator");
  if (!accepts_) throw std::invalid_argument("RestrictedMotionValidator needs an accepts predicate");
}

LocalMotion RestrictedMotionValidator::validate(ConfigView q_from, ConfigView q_to) const {
  if (!accepts_(q_from, q_to)) return LocalMotion{};
  return base_->validate(q_from, q_to);
}

std::optional<std::vector<Config>> RestrictedMotionValidator::validate_whole(ConfigView q_from, ConfigView q_to) const {
  if (!accepts_(q_from, q_to)) return std::nullopt;
  return base_->validate_whole(q_from, q_to);
}

}  // namespace sscbirrt
