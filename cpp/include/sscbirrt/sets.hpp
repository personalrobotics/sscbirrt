// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sscbirrt/types.hpp"

namespace sscbirrt {

struct Sample {
  Config q;
  Provenance source;  // AnyOf prepends its child index; FiniteSet contributes the member index
  // Equal for candidates of one draw that differ only by full turns of revolute joints: the same physical
  // configuration, so a validator that declares full_turn_invariant() judges them alike (#200). Empty: no key.
  std::vector<std::int64_t> key = {};
};

// The key of q: each joint the IK declares revolute reduced to [-pi, pi], every joint rounded to nanoradians.
std::vector<std::int64_t> full_turn_key(ConfigView q, const std::vector<bool>& revolute);

enum class Capability { Sampler, Distance, Violation, Projector };
const char* capability_name(Capability c);

class SetSampler {
 public:
  virtual ~SetSampler() = default;
  virtual std::vector<Sample> sample(Rng& rng) const = 0;  // candidates of one draw, unfiltered
};

class SetDistance {
 public:
  virtual ~SetDistance() = default;
  virtual double distance(ConfigView q) const = 0;  // nonnegative, geometric, before tolerance
};

class SetViolation {
 public:
  virtual ~SetViolation() = default;
  virtual double violation(ConfigView q) const = 0;  // nonnegative; exactly zero iff contains(q)
};

class SetProjector {
 public:
  virtual ~SetProjector() = default;
  virtual std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const = 0;
};

// Membership is the only requirement. Capabilities are reported through accessors
// that return a pointer or nullptr (the C++ form of Python's supports()).
class StateSet {
 public:
  virtual ~StateSet() = default;
  virtual bool contains(ConfigView q) const = 0;

  virtual const SetSampler* sampler() const { return nullptr; }
  virtual const SetDistance* distancer() const { return nullptr; }
  virtual const SetViolation* violator() const { return nullptr; }
  virtual const SetProjector* projector() const { return nullptr; }
  bool supports(Capability c) const;
  virtual std::string why_unsupported(Capability c) const;  // empty if supported

  virtual bool is_finite() const { return false; }
  virtual std::vector<Sample> members() const { return {}; }  // exhaustive iff is_finite()
  virtual std::vector<Sample> seeds() const { return {}; }    // explicit configurations anywhere inside

  virtual std::string describe() const = 0;  // Python's repr, for messages
};

using SetPtr = std::shared_ptr<const StateSet>;

// ---------------------------------------------------------------------------
// Leaves
// ---------------------------------------------------------------------------

class FiniteSet final : public StateSet, public SetSampler, public SetDistance, public SetViolation {
 public:
  explicit FiniteSet(std::vector<Config> members, double tolerance = 1e-6, Metric metric = euclidean);

  bool contains(ConfigView q) const override { return distance(q) <= tolerance_; }
  double distance(ConfigView q) const override;
  double violation(ConfigView q) const override;
  std::vector<Sample> sample(Rng& rng) const override;  // one member, drawn uniformly, source {index}

  const SetSampler* sampler() const override { return this; }
  const SetDistance* distancer() const override { return this; }
  const SetViolation* violator() const override { return this; }

  bool is_finite() const override { return true; }
  std::vector<Sample> members() const override;
  std::vector<Sample> seeds() const override { return members(); }
  std::string describe() const override;

  const std::vector<Config>& configs() const { return members_; }
  double tolerance() const { return tolerance_; }

 private:
  std::vector<Config> members_;
  double tolerance_;
  Metric metric_;
};

class EmptySet final : public StateSet {
 public:
  bool contains(ConfigView) const override { return false; }
  bool is_finite() const override { return true; }
  std::string describe() const override { return "EmptySet()"; }
};

class PredicateSet final : public StateSet {
 public:
  explicit PredicateSet(std::function<bool(ConfigView)> predicate, std::string name = "");
  bool contains(ConfigView q) const override { return predicate_(q); }
  std::string describe() const override;

 private:
  std::function<bool(ConfigView)> predicate_;
  std::string name_;
};

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

class IntersectionProjection {
 public:
  virtual ~IntersectionProjection() = default;
  virtual void check_requirements(const std::vector<SetPtr>& children) const { (void)children; }
  virtual std::optional<Config> project(const std::vector<SetPtr>& children, ConfigView q_previous,
                                        ConfigView q_proposed) const = 0;
};

class IntersectionSampling {
 public:
  virtual ~IntersectionSampling() = default;
  virtual void check_requirements(const std::vector<SetPtr>& children) const { (void)children; }
  virtual std::vector<Sample> sample(const std::vector<SetPtr>& children, Rng& rng) const = 0;
};

// Union. One child delegates every capability. Several children: distance and violation
// are the min if every child has them; sampling needs weights (the mixture policy) and
// every child to sample, and prepends the chosen child's index to each sample's source;
// projection needs every child to project and returns the successful result nearest
// q_proposed under metric.
class AnyOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
 public:
  AnyOf(std::vector<SetPtr> children, std::optional<std::vector<double>> weights = std::nullopt,
        Metric metric = euclidean);

  bool contains(ConfigView q) const override;
  double distance(ConfigView q) const override;
  double violation(ConfigView q) const override;
  std::vector<Sample> sample(Rng& rng) const override;
  std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const override;

  const SetSampler* sampler() const override { return has_sampler_ ? this : nullptr; }
  const SetDistance* distancer() const override { return has_distance_ ? this : nullptr; }
  const SetViolation* violator() const override { return has_violation_ ? this : nullptr; }
  const SetProjector* projector() const override { return has_projector_ ? this : nullptr; }
  std::string why_unsupported(Capability c) const override;

  bool is_finite() const override;
  std::vector<Sample> members() const override;
  std::vector<Sample> seeds() const override;
  std::string describe() const override;

  const std::vector<SetPtr>& children() const { return children_; }
  const std::optional<std::vector<double>>& weights() const { return weights_; }  // normalized

 private:
  std::vector<SetPtr> children_;
  std::optional<std::vector<double>> weights_;
  Metric metric_;
  bool has_sampler_ = false, has_distance_ = false, has_violation_ = false, has_projector_ = false;
};

// Intersection. One child delegates every capability. Several children: distance and
// violation are the max if every child has them; sampling and projection need a named
// strategy, whose check_requirements() is checked here at construction.
class AllOf final : public StateSet, public SetSampler, public SetDistance, public SetViolation, public SetProjector {
 public:
  AllOf(std::vector<SetPtr> children, std::shared_ptr<const IntersectionProjection> projection = nullptr,
        std::shared_ptr<const IntersectionSampling> sampling = nullptr);

  bool contains(ConfigView q) const override;
  double distance(ConfigView q) const override;
  double violation(ConfigView q) const override;
  std::vector<Sample> sample(Rng& rng) const override;
  std::optional<Config> project(ConfigView q_previous, ConfigView q_proposed) const override;

  const SetSampler* sampler() const override { return has_sampler_ ? this : nullptr; }
  const SetDistance* distancer() const override { return has_distance_ ? this : nullptr; }
  const SetViolation* violator() const override { return has_violation_ ? this : nullptr; }
  const SetProjector* projector() const override { return has_projector_ ? this : nullptr; }
  std::string why_unsupported(Capability c) const override;

  bool is_finite() const override;
  std::vector<Sample> members() const override;
  std::vector<Sample> seeds() const override;
  std::string describe() const override;

  const std::vector<SetPtr>& children() const { return children_; }

 private:
  std::vector<SetPtr> children_;
  std::shared_ptr<const IntersectionProjection> projection_;
  std::shared_ptr<const IntersectionSampling> sampling_;
  bool has_sampler_ = false, has_distance_ = false, has_violation_ = false, has_projector_ = false;
};

// Repeatedly project onto the unsatisfied child with the largest violation. Progress is
// the lexicographic decrease of the descending-sorted violation profile by at least
// progress_tolerance in some position; gives up after a full sweep without progress, when
// a projector returns nullopt or leaves the point unchanged, or at max_iters.
class MostViolatedProjection final : public IntersectionProjection {
 public:
  explicit MostViolatedProjection(int max_iters = 50, double progress_tolerance = 1e-6);
  void check_requirements(const std::vector<SetPtr>& children) const override;
  std::optional<Config> project(const std::vector<SetPtr>& children, ConfigView q_previous,
                                ConfigView q_proposed) const override;

 private:
  bool improved(const std::vector<double>& profile, const std::vector<double>& best) const;
  int max_iters_;
  double progress_tolerance_;
};

// Draw from one child and keep the candidates every other child contains.
class RejectionSampling final : public IntersectionSampling {
 public:
  explicit RejectionSampling(int source = 0) : source_(source) {}
  void check_requirements(const std::vector<SetPtr>& children) const override;
  std::vector<Sample> sample(const std::vector<SetPtr>& children, Rng& rng) const override;

 private:
  int source_;
};

}  // namespace sscbirrt
