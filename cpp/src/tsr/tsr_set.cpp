// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/tsr/tsr_set.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sscbirrt::tsr {

double region_distance(const Region& region, const Transform& T) {
  const sstsr::Transform pose = to_sstsr(T);
  if (const auto* chain = std::get_if<TSRChain>(&region)) return chain->distance(pose).first;
  return std::get<TSR>(region).distance(pose);
}

std::pair<double, Transform> region_closest_transform(const Region& region, const Transform& T) {
  const sstsr::Transform pose = to_sstsr(T);
  const auto [dist, closest] = std::visit([&pose](const auto& r) { return r.closest_transform(pose); }, region);
  return {dist, from_sstsr(closest)};
}

Transform region_sample(const Region& region, Rng& rng) {
  return from_sstsr(std::visit([&rng](const auto& r) { return r.sample(rng); }, region));
}

double region_volume(const Region& region) {
  if (const auto* chain = std::get_if<TSRChain>(&region)) {
    double total = 0.0;
    for (const TSR& link : chain->tsrs()) total += link.volume();
    return total;
  }
  return std::get<TSR>(region).volume();
}

TSRConfigurationSet::TSRConfigurationSet(Region region, std::shared_ptr<const ForwardKinematics> fk,
                                         std::shared_ptr<const IKSolver> ik, std::shared_ptr<const JointSpace> space,
                                         double tolerance, int max_projection_iters, double progress_tolerance)
    : region_(std::move(region)),
      fk_(std::move(fk)),
      ik_(std::move(ik)),
      space_(std::move(space)),
      tolerance_(tolerance),
      max_projection_iters_(max_projection_iters),
      progress_tolerance_(progress_tolerance) {
  if (!fk_ || !ik_ || !space_) throw std::invalid_argument("TSRConfigurationSet needs forward kinematics, an IK solver, and a space");
  if (fk_->dof() != space_->dof() || ik_->dof() != space_->dof()) {
    throw std::invalid_argument("TSRConfigurationSet: kinematics dof " + std::to_string(fk_->dof()) + "/" +
                                std::to_string(ik_->dof()) + " does not match the space's " + std::to_string(space_->dof()));
  }
  if (tolerance_ < 0.0) throw std::invalid_argument("membership_tolerance must be nonnegative");
  if (max_projection_iters_ < 1) throw std::invalid_argument("max_projection_iters must be at least 1");
  if (!(progress_tolerance_ > 0.0)) throw std::invalid_argument("projection_progress_tolerance must be positive");
}

bool TSRConfigurationSet::within_limits(ConfigView q) const {
  if (q.size() != static_cast<std::size_t>(space_->dof())) return false;
  for (int i = 0; i < space_->dof(); ++i) {
    if (space_->angular()[static_cast<std::size_t>(i)]) continue;
    const double v = q[static_cast<std::size_t>(i)];
    if (v < space_->lower()[static_cast<std::size_t>(i)] || v > space_->upper()[static_cast<std::size_t>(i)]) return false;
  }
  return true;
}

double TSRConfigurationSet::distance(ConfigView q) const { return region_distance(region_, fk_->fk(q)); }

double TSRConfigurationSet::violation(ConfigView q) const { return std::max(0.0, distance(q) - tolerance_); }

std::vector<Sample> TSRConfigurationSet::sample(Rng& rng) const {
  const Transform pose = region_sample(region_, rng);
  std::vector<Sample> out;
  const std::vector<bool> revolute = ik_->revolute_joints();
  for (Config& q : ik_->solve(pose, {})) {
    if (!within_limits(q)) continue;
    std::vector<std::int64_t> key = revolute.empty() ? std::vector<std::int64_t>{} : full_turn_key(q, revolute);
    out.push_back(Sample{std::move(q), {}, std::move(key)});
  }
  return out;
}

std::optional<Config> TSRConfigurationSet::project(ConfigView, ConfigView q_proposed) const {
  Config q = to_config(q_proposed);
  double prev_dist = std::numeric_limits<double>::infinity();
  for (int it = 0; it < max_projection_iters_; ++it) {
    const auto [dist, target] = region_closest_transform(region_, fk_->fk(q));
    if (dist <= tolerance_) return q;
    if (prev_dist - dist < progress_tolerance_) return std::nullopt;
    prev_dist = dist;

    std::optional<Config> best;
    double best_d = std::numeric_limits<double>::infinity();
    for (Config& sol : ik_->solve(target, q)) {
      if (!within_limits(sol)) continue;
      const double d = space_->distance(q, sol);
      if (d < best_d) {
        best_d = d;
        best = std::move(sol);
      }
    }
    if (!best) return std::nullopt;
    q = std::move(*best);
  }
  return std::nullopt;
}

std::string TSRConfigurationSet::describe() const { return "TSRConfigurationSet(tolerance=" + std::to_string(tolerance_) + ")"; }

std::vector<double> tsr_weights(const std::vector<std::shared_ptr<const TSRConfigurationSet>>& sets) {
  std::vector<double> w;
  w.reserve(sets.size());
  bool any = false;
  for (const auto& s : sets) {
    w.push_back(region_volume(s->region()));
    any = any || w.back() > 0.0;
  }
  if (!any) std::fill(w.begin(), w.end(), 1.0);
  return w;
}

}  // namespace sscbirrt::tsr
