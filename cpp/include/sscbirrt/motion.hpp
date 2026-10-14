// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "sscbirrt/space.hpp"
#include "sscbirrt/types.hpp"

namespace sscbirrt {

// The validated part of a local motion from q_from toward q_to. If reached is true on a
// nonzero motion, configs is nonempty and configs.back() equals q_to exactly; the planner
// throws ContractError otherwise.
struct LocalMotion {
  std::vector<Config> configs;  // validated configurations after q_from, in order
  bool reached = false;
};

// Two modes (#230). validate answers how far from the start the motion is valid: its prefix becomes tree nodes.
// validate_whole answers whether the whole motion is valid, for callers that discard the motion on a no (shortcuts):
// the configurations validate would return with reached = true, or nullopt. The default answers through validate.
class MotionValidator {
 public:
  virtual ~MotionValidator() = default;
  virtual LocalMotion validate(ConfigView q_from, ConfigView q_to) const = 0;
  virtual std::optional<std::vector<Config>> validate_whole(ConfigView q_from, ConfigView q_to) const {
    LocalMotion m = validate(q_from, q_to);
    if (!m.reached) return std::nullopt;
    return std::move(m.configs);
  }
};

// Straight segment sampled every `resolution` along the space's direction: n = max(1, ceil(d / resolution)),
// samples q_from + (i/n) * direction for i = 1..n, the last being q_to exactly. Stops at the first
// inadmissible sample with reached = false.
class DiscreteMotionValidator final : public MotionValidator {
 public:
  DiscreteMotionValidator(std::shared_ptr<const JointSpace> space, std::function<bool(ConfigView)> is_admissible,
                          double resolution);
  LocalMotion validate(ConfigView q_from, ConfigView q_to) const override;
  // The same samples, checked in van der Corput order (spread_order), stopping at the first inadmissible one found.
  std::optional<std::vector<Config>> validate_whole(ConfigView q_from, ConfigView q_to) const override;

 private:
  std::shared_ptr<const JointSpace> space_;
  std::function<bool(ConfigView)> is_admissible_;
  double resolution_;
};

// A base validator plus an extra motion predicate; a rejected motion is an empty LocalMotion.
class RestrictedMotionValidator final : public MotionValidator {
 public:
  RestrictedMotionValidator(std::shared_ptr<const MotionValidator> base,
                            std::function<bool(ConfigView, ConfigView)> accepts);
  LocalMotion validate(ConfigView q_from, ConfigView q_to) const override;
  std::optional<std::vector<Config>> validate_whole(ConfigView q_from, ConfigView q_to) const override;

 private:
  std::shared_ptr<const MotionValidator> base_;
  std::function<bool(ConfigView, ConfigView)> accepts_;
};

}  // namespace sscbirrt
