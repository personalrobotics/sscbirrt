// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/sets.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <sstream>
#include <stdexcept>

#include "sscbirrt/errors.hpp"

namespace sscbirrt {

namespace {

std::string join_children(const std::vector<SetPtr>& children) {
  std::string s = "[";
  for (std::size_t i = 0; i < children.size(); ++i) {
    if (i) s += ", ";
    s += children[i]->describe();
  }
  return s + "]";
}

bool all_support(const std::vector<SetPtr>& children, Capability c) {
  return std::all_of(children.begin(), children.end(), [c](const SetPtr& s) { return s->supports(c); });
}

Provenance prepend(int i, const Provenance& p) {
  Provenance out;
  out.reserve(p.size() + 1);
  out.push_back(i);
  out.insert(out.end(), p.begin(), p.end());
  return out;
}

bool same_config(const Config& a, const Config& b) { return a == b; }  // bitwise, as np.array_equal

}  // namespace

const char* capability_name(Capability c) {
  switch (c) {
    case Capability::Sampler: return "SetSampler";
    case Capability::Distance: return "SetDistance";
    case Capability::Violation: return "SetViolation";
    case Capability::Projector: return "SetProjector";
  }
  return "?";
}

std::string RootReport::summary() const {
  std::string s;
  auto add = [&s](int n, const char* what) {
    if (n == 0) return;
    if (!s.empty()) s += ", ";
    s += std::to_string(n) + " " + what;
  };
  add(draws_empty, "IK unreachable");
  add(outside_space, "outside joint space");
  add(in_collision, "in collision");
  add(constraint_violated, "constraint violated");
  return s;
}

bool StateSet::supports(Capability c) const {
  switch (c) {
    case Capability::Sampler: return sampler() != nullptr;
    case Capability::Distance: return distancer() != nullptr;
    case Capability::Violation: return violator() != nullptr;
    case Capability::Projector: return projector() != nullptr;
  }
  return false;
}

std::string StateSet::why_unsupported(Capability c) const {
  if (supports(c)) return "";
  return describe() + " does not support " + capability_name(c);
}

// ---------------------------------------------------------------------------
// FiniteSet
// ---------------------------------------------------------------------------

FiniteSet::FiniteSet(std::vector<Config> members, double tolerance, Metric metric)
    : members_(std::move(members)), tolerance_(tolerance), metric_(std::move(metric)) {
  if (members_.empty()) throw std::invalid_argument("FiniteSet requires at least one configuration");
  for (const Config& m : members_) {
    if (m.size() != members_.front().size()) throw std::invalid_argument("FiniteSet members must have equal length");
  }
}

double FiniteSet::distance(ConfigView q) const {
  double best = std::numeric_limits<double>::infinity();
  for (const Config& m : members_) best = std::min(best, metric_(q, m));
  return best;
}

double FiniteSet::violation(ConfigView q) const { return std::max(0.0, distance(q) - tolerance_); }

std::vector<Sample> FiniteSet::sample(Rng& rng) const {
  const auto i = index(rng, members_.size());
  return {Sample{members_[i], Provenance{static_cast<int>(i)}}};
}

std::vector<Sample> FiniteSet::members() const {
  std::vector<Sample> out;
  out.reserve(members_.size());
  for (std::size_t i = 0; i < members_.size(); ++i) out.push_back({members_[i], Provenance{static_cast<int>(i)}});
  return out;
}

std::string FiniteSet::describe() const { return "FiniteSet(" + std::to_string(members_.size()) + " configurations)"; }

// ---------------------------------------------------------------------------
// PredicateSet
// ---------------------------------------------------------------------------

PredicateSet::PredicateSet(std::function<bool(ConfigView)> predicate, std::string name)
    : predicate_(std::move(predicate)), name_(std::move(name)) {
  if (!predicate_) throw std::invalid_argument("PredicateSet requires a predicate");
}

std::string PredicateSet::describe() const { return "PredicateSet(" + (name_.empty() ? "<predicate>" : name_) + ")"; }

// ---------------------------------------------------------------------------
// AnyOf
// ---------------------------------------------------------------------------

AnyOf::AnyOf(std::vector<SetPtr> children, std::optional<std::vector<double>> weights, Metric metric)
    : children_(std::move(children)), weights_(std::move(weights)), metric_(std::move(metric)) {
  if (children_.empty()) throw std::invalid_argument("AnyOf requires at least one child set");
  for (const SetPtr& c : children_) {
    if (!c) throw std::invalid_argument("AnyOf child must not be null");
  }
  if (weights_) {
    if (weights_->size() != children_.size()) {
      throw std::invalid_argument("weights length (" + std::to_string(weights_->size()) +
                                  ") must match number of children (" + std::to_string(children_.size()) + ")");
    }
    double sum = 0.0;
    for (double w : *weights_) {
      if (w < 0.0) throw std::invalid_argument("weights must be nonnegative and sum to a positive value");
      sum += w;
    }
    if (!(sum > 0.0)) throw std::invalid_argument("weights must be nonnegative and sum to a positive value");
    for (double& w : *weights_) w /= sum;
  }
  has_distance_ = all_support(children_, Capability::Distance);
  has_violation_ = all_support(children_, Capability::Violation);
  if (children_.size() == 1) {
    has_sampler_ = children_[0]->supports(Capability::Sampler);
    has_projector_ = children_[0]->supports(Capability::Projector);
  } else {
    has_sampler_ = weights_.has_value() && all_support(children_, Capability::Sampler);
    has_projector_ = all_support(children_, Capability::Projector);
  }
}

std::string AnyOf::why_unsupported(Capability c) const {
  if (supports(c)) return "";
  std::string why = "every child must support it";
  if (c == Capability::Sampler && children_.size() > 1 && !weights_) {
    why = "a multi-child union needs an explicit mixture policy (pass weights=)";
  }
  return "AnyOf with " + std::to_string(children_.size()) + " children does not support " + capability_name(c) + ": " + why;
}

bool AnyOf::contains(ConfigView q) const {
  return std::any_of(children_.begin(), children_.end(), [q](const SetPtr& c) { return c->contains(q); });
}

double AnyOf::distance(ConfigView q) const {
  if (!has_distance_) throw UnsupportedCapability(why_unsupported(Capability::Distance));
  double best = std::numeric_limits<double>::infinity();
  for (const SetPtr& c : children_) best = std::min(best, c->distancer()->distance(q));
  return best;
}

double AnyOf::violation(ConfigView q) const {
  if (!has_violation_) throw UnsupportedCapability(why_unsupported(Capability::Violation));
  double best = std::numeric_limits<double>::infinity();
  for (const SetPtr& c : children_) best = std::min(best, c->violator()->violation(q));
  return best;
}

std::vector<std::int64_t> full_turn_key(ConfigView q, const std::vector<bool>& revolute) {
  std::vector<std::int64_t> key(q.size());
  for (std::size_t i = 0; i < q.size(); ++i) {
    const double v = i < revolute.size() && revolute[i] ? std::remainder(q[i], 2.0 * std::numbers::pi) : q[i];
    key[i] = std::llround(v * 1e9);
  }
  return key;
}

std::vector<Sample> AnyOf::sample(Rng& rng) const {
  if (!has_sampler_) throw UnsupportedCapability(why_unsupported(Capability::Sampler));
  std::size_t i = 0;
  if (children_.size() > 1) {
    const double u = unit(rng);
    double acc = 0.0;
    i = children_.size() - 1;
    for (std::size_t k = 0; k < children_.size(); ++k) {
      acc += (*weights_)[k];
      if (u < acc) {
        i = k;
        break;
      }
    }
  }
  std::vector<Sample> out = children_[i]->sampler()->sample(rng);
  for (Sample& s : out) s.source = prepend(static_cast<int>(i), s.source);
  return out;
}

std::optional<Config> AnyOf::project(ConfigView q_previous, ConfigView q_proposed) const {
  if (!has_projector_) throw UnsupportedCapability(why_unsupported(Capability::Projector));
  if (children_.size() == 1) return children_[0]->projector()->project(q_previous, q_proposed);
  std::optional<Config> best;
  double best_dist = std::numeric_limits<double>::infinity();
  for (const SetPtr& c : children_) {
    std::optional<Config> q = c->projector()->project(q_previous, q_proposed);
    if (!q) continue;
    const double d = metric_(q_proposed, *q);
    if (d < best_dist) {
      best = std::move(q);
      best_dist = d;
    }
  }
  return best;
}

bool AnyOf::is_finite() const {
  return std::all_of(children_.begin(), children_.end(), [](const SetPtr& c) { return c->is_finite(); });
}

std::vector<Sample> AnyOf::members() const {
  if (!is_finite()) return {};
  std::vector<Sample> out;
  for (std::size_t i = 0; i < children_.size(); ++i) {
    for (Sample& m : children_[i]->members()) out.push_back({std::move(m.q), prepend(static_cast<int>(i), m.source)});
  }
  return out;
}

std::vector<Sample> AnyOf::seeds() const {
  std::vector<Sample> out;
  for (std::size_t i = 0; i < children_.size(); ++i) {
    for (Sample& m : children_[i]->seeds()) out.push_back({std::move(m.q), prepend(static_cast<int>(i), m.source)});
  }
  return out;
}

std::string AnyOf::describe() const { return "AnyOf(" + join_children(children_) + ")"; }

// ---------------------------------------------------------------------------
// AllOf
// ---------------------------------------------------------------------------

AllOf::AllOf(std::vector<SetPtr> children, std::shared_ptr<const IntersectionProjection> projection,
             std::shared_ptr<const IntersectionSampling> sampling)
    : children_(std::move(children)), projection_(std::move(projection)), sampling_(std::move(sampling)) {
  if (children_.empty()) throw std::invalid_argument("AllOf requires at least one child set");
  for (const SetPtr& c : children_) {
    if (!c) throw std::invalid_argument("AllOf child must not be null");
  }
  has_distance_ = all_support(children_, Capability::Distance);
  has_violation_ = all_support(children_, Capability::Violation);
  if (children_.size() == 1) {
    has_sampler_ = children_[0]->supports(Capability::Sampler);
    has_projector_ = children_[0]->supports(Capability::Projector);
  } else {
    if (projection_) {
      projection_->check_requirements(children_);
      has_projector_ = true;
    }
    if (sampling_) {
      sampling_->check_requirements(children_);
      has_sampler_ = true;
    }
  }
}

std::string AllOf::why_unsupported(Capability c) const {
  if (supports(c)) return "";
  std::string why = "every child must support it";
  if (c == Capability::Projector) why = "a multi-child intersection needs an explicit projection strategy (pass projection=)";
  if (c == Capability::Sampler) why = "a multi-child intersection needs an explicit sampling strategy (pass sampling=)";
  return "AllOf with " + std::to_string(children_.size()) + " children does not support " + capability_name(c) + ": " + why;
}

bool AllOf::contains(ConfigView q) const {
  return std::all_of(children_.begin(), children_.end(), [q](const SetPtr& c) { return c->contains(q); });
}

double AllOf::distance(ConfigView q) const {
  if (!has_distance_) throw UnsupportedCapability(why_unsupported(Capability::Distance));
  double worst = 0.0;
  for (const SetPtr& c : children_) worst = std::max(worst, c->distancer()->distance(q));
  return worst;
}

double AllOf::violation(ConfigView q) const {
  if (!has_violation_) throw UnsupportedCapability(why_unsupported(Capability::Violation));
  double worst = 0.0;
  for (const SetPtr& c : children_) worst = std::max(worst, c->violator()->violation(q));
  return worst;
}

std::vector<Sample> AllOf::sample(Rng& rng) const {
  if (!has_sampler_) throw UnsupportedCapability(why_unsupported(Capability::Sampler));
  if (children_.size() == 1) return children_[0]->sampler()->sample(rng);
  return sampling_->sample(children_, rng);
}

std::optional<Config> AllOf::project(ConfigView q_previous, ConfigView q_proposed) const {
  if (!has_projector_) throw UnsupportedCapability(why_unsupported(Capability::Projector));
  if (children_.size() == 1) return children_[0]->projector()->project(q_previous, q_proposed);
  return projection_->project(children_, q_previous, q_proposed);
}

bool AllOf::is_finite() const {
  return std::any_of(children_.begin(), children_.end(), [](const SetPtr& c) { return c->is_finite(); });
}

std::vector<Sample> AllOf::members() const {
  for (std::size_t i = 0; i < children_.size(); ++i) {
    if (!children_[i]->is_finite()) continue;
    std::vector<Sample> out;
    for (Sample& m : children_[i]->members()) {
      bool keep = true;
      for (std::size_t j = 0; j < children_.size() && keep; ++j) {
        if (j != i && !children_[j]->contains(m.q)) keep = false;
      }
      if (keep) out.push_back(std::move(m));
    }
    return out;
  }
  return {};
}

std::vector<Sample> AllOf::seeds() const {
  if (is_finite()) return members();
  std::vector<Sample> out;
  for (const SetPtr& c : children_) {
    for (Sample& m : c->seeds()) {
      if (!contains(m.q)) continue;
      const bool dup = std::any_of(out.begin(), out.end(), [&m](const Sample& k) { return same_config(m.q, k.q); });
      if (!dup) out.push_back(std::move(m));
    }
  }
  return out;
}

std::string AllOf::describe() const { return "AllOf(" + join_children(children_) + ")"; }

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

MostViolatedProjection::MostViolatedProjection(int max_iters, double progress_tolerance)
    : max_iters_(max_iters), progress_tolerance_(progress_tolerance) {
  if (max_iters_ < 1) throw std::invalid_argument("max_iters must be at least 1, got " + std::to_string(max_iters_));
  if (!(progress_tolerance_ > 0.0)) throw std::invalid_argument("progress_tolerance must be positive");
}

void MostViolatedProjection::check_requirements(const std::vector<SetPtr>& children) const {
  for (std::size_t i = 0; i < children.size(); ++i) {
    if (!(children[i]->supports(Capability::Violation) && children[i]->supports(Capability::Projector))) {
      throw UnsupportedCapability("MostViolatedProjection requires every child to support violation and projection; child " +
                                  std::to_string(i) + " (" + children[i]->describe() + ") does not");
    }
  }
}

bool MostViolatedProjection::improved(const std::vector<double>& profile, const std::vector<double>& best) const {
  for (std::size_t i = 0; i < profile.size() && i < best.size(); ++i) {
    if (best[i] - profile[i] >= progress_tolerance_) return true;
    if (profile[i] - best[i] >= progress_tolerance_) return false;
  }
  return false;
}

std::optional<Config> MostViolatedProjection::project(const std::vector<SetPtr>& children, ConfigView q_previous,
                                                      ConfigView q_proposed) const {
  Config q = to_config(q_proposed);
  std::optional<std::vector<double>> best_profile;
  std::size_t stale = 0;
  for (int it = 0; it < max_iters_; ++it) {
    std::vector<double> violations(children.size());
    std::size_t worst = 0;
    for (std::size_t i = 0; i < children.size(); ++i) {
      violations[i] = children[i]->contains(q) ? 0.0 : children[i]->violator()->violation(q);
      if (violations[i] > violations[worst]) worst = i;  // first maximum wins, as np.argmax
    }
    if (violations[worst] <= 0.0) return q;

    std::vector<double> profile = violations;
    std::sort(profile.begin(), profile.end(), std::greater<double>());
    if (!best_profile || improved(profile, *best_profile)) {
      best_profile = std::move(profile);
      stale = 0;
    } else if (++stale >= children.size()) {
      return std::nullopt;  // a full sweep without progress: cycling or stuck
    }

    std::optional<Config> next = children[worst]->projector()->project(q_previous, q);
    if (!next) return std::nullopt;
    if (same_config(*next, q)) return std::nullopt;  // the projector cannot move this point while its child is violated
    q = std::move(*next);
  }
  return std::nullopt;
}

void RejectionSampling::check_requirements(const std::vector<SetPtr>& children) const {
  if (source_ < 0 || static_cast<std::size_t>(source_) >= children.size()) {
    throw std::invalid_argument("RejectionSampling source index " + std::to_string(source_) + " out of range for " +
                                std::to_string(children.size()) + " children");
  }
  if (!children[static_cast<std::size_t>(source_)]->supports(Capability::Sampler)) {
    throw UnsupportedCapability("RejectionSampling requires child " + std::to_string(source_) + " (" +
                                children[static_cast<std::size_t>(source_)]->describe() + ") to support sampling");
  }
}

std::vector<Sample> RejectionSampling::sample(const std::vector<SetPtr>& children, Rng& rng) const {
  const auto src = static_cast<std::size_t>(source_);
  std::vector<Sample> out;
  for (Sample& s : children[src]->sampler()->sample(rng)) {
    bool keep = true;
    for (std::size_t j = 0; j < children.size() && keep; ++j) {
      if (j != src && !children[j]->contains(s.q)) keep = false;
    }
    if (keep) out.push_back(std::move(s));
  }
  return out;
}

}  // namespace sscbirrt
