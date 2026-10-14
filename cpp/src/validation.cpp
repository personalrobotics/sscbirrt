// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// Compiled without floating-point contraction (CMakeLists.txt), so every sample rounds as sscbirrt.validation's does.
#include "sscbirrt/validation.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "sscbirrt/trajectory.hpp"

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
  return {{"endpoints", "checked"},
          {"interpolation", "raw_linear"},
          {"rounding", "1e-09"},
          {"trajectory", "source_path_states_or_speed_bounded_time_steps"},
          {"trajectory_limits", "exact"}};
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
                                                 std::optional<double> fraction, std::optional<double> time_s) const {
  auto why = classify(*space_, *validator_, path_constraint_.get(), q);
  if (!why) return std::nullopt;
  InvalidWitness w;
  w.failure = why->first;
  w.detail = std::move(why->second);
  w.configuration = to_config(q);
  w.segment = segment;
  w.fraction = fraction;
  w.time_s = time_s;
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

void Validator::segment_entries(ConfigView q0, ConfigView q1, int index, std::vector<Entry>& out) const {
  const std::size_t dof = q0.size();
  Config d(dof);
  double dist = 0.0;
  for (std::size_t k = 0; k < dof; ++k) {
    d[k] = q1[k] - q0[k];
    dist += d[k] * d[k];
  }
  const int n = subdivisions(std::sqrt(dist), policy_.resolution);
  for (int i = 1; i <= n; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(n);
    Entry e;
    if (i == n) {
      e.q.assign(q1.begin(), q1.end());
    } else {
      e.q.resize(dof);
      for (std::size_t k = 0; k < dof; ++k) e.q[k] = q0[k] + t * d[k];
    }
    e.segment = index;
    e.fraction = t;
    out.push_back(std::move(e));
  }
}

std::pair<long, std::optional<InvalidWitness>> Validator::search(const std::vector<Entry>& entries, bool first) const {
  std::vector<bool> checked(entries.size(), false);
  long checks = 0;
  for (const int k : spread_order(static_cast<int>(entries.size()))) {
    ++checks;
    auto w = entry_witness(entries[static_cast<std::size_t>(k)]);
    if (!w) {
      checked[static_cast<std::size_t>(k)] = true;
      continue;
    }
    if (first) {
      for (int i = 0; i < k; ++i) {
        if (checked[static_cast<std::size_t>(i)]) continue;
        ++checks;
        if (auto earlier = entry_witness(entries[static_cast<std::size_t>(i)])) return {checks, std::move(earlier)};
      }
    }
    return {checks, std::move(w)};
  }
  return {checks, std::nullopt};
}

ValidationReport Validator::validate_state(ConfigView q) const {
  check_config(q, "q");
  std::vector<Entry> entries(1);
  entries[0].q = to_config(q);
  auto [checks, w] = search(entries, true);
  return report("state", 0, checks, std::move(w));
}

ValidationReport Validator::validate_motion(ConfigView q0, ConfigView q1) const {
  check_config(q0, "q0");
  check_config(q1, "q1");
  std::vector<Entry> entries(1);
  entries[0].q = to_config(q0);
  entries[0].segment = 0;
  entries[0].fraction = 0.0;
  segment_entries(q0, q1, 0, entries);
  auto [checks, w] = search(entries, true);
  return report("motion", 1, checks, std::move(w));
}

ValidationReport Validator::validate_path(const std::vector<Config>& path) const {
  if (path.empty()) throw std::invalid_argument("path: empty");
  for (std::size_t i = 0; i < path.size(); ++i) check_config(path[i], "path[" + std::to_string(i) + "]");
  const bool many = path.size() > 1;
  std::vector<Entry> entries(1);
  entries[0].q = path[0];
  if (many) {
    entries[0].segment = 0;
    entries[0].fraction = 0.0;
  }
  for (std::size_t k = 0; k + 1 < path.size(); ++k) segment_entries(path[k], path[k + 1], static_cast<int>(k), entries);
  auto [checks, w] = search(entries, true);
  return report("path", static_cast<int>(path.size()) - 1, checks, std::move(w));
}

// ---------------------------------------------------------------------------------------------------------------------
// Trajectories: sscbirrt.validation's trajectory functions, operation by operation
// ---------------------------------------------------------------------------------------------------------------------

namespace {

constexpr double kRel = 1e-9;   // relative slack on velocity and acceleration bounds
constexpr double kTolX = 1e-9;  // position continuity (relative to at least 1)
constexpr double kTolV = 1e-7;  // velocity continuity and rest (relative to max(1, vmax))

double norm(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v * v;
  return std::sqrt(s);
}

double dot(const Config& a, const Config& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

double max_abs(const Config& x) {
  double m = 0.0;
  for (double v : x) m = std::max(m, std::fabs(v));
  return m;
}

// The time a monotone ramp reaches x, by bisection (60 halvings of [0, T]).
double time_at(const trajectory::Ramp& ramp, double x, double T) {
  double lo = 0.0, hi = T;
  const bool rising = ramp.position(T) >= ramp.position(0.0);
  for (int i = 0; i < 60; ++i) {
    const double mid = 0.5 * (lo + hi);
    if ((ramp.position(mid) < x) == rising) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

std::string where(std::size_t k, std::size_t j) { return "piece " + std::to_string(k) + ", joint " + std::to_string(j); }

}  // namespace

void well_formed(const trajectory::Trajectory& traj, int dof) {
  const auto n = static_cast<std::size_t>(dof);
  const std::vector<double>& vmax = traj.velocity_limits;
  const std::vector<double>& amax = traj.acceleration_limits;
  for (const auto* lim : {&vmax, &amax}) {
    bool ok = lim->size() == n;
    for (double v : *lim) ok = ok && std::isfinite(v) && v > 0.0;
    if (!ok) throw MalformedTrajectory("limits must be " + std::to_string(dof) + " positive finite values");
  }
  if (traj.pieces.empty()) throw MalformedTrajectory("no pieces");
  std::vector<std::pair<double, double>> previous;
  for (std::size_t k = 0; k < traj.pieces.size(); ++k) {
    const trajectory::Piece& piece = traj.pieces[k];
    if (piece.ramps.size() != n) throw MalformedTrajectory("piece " + std::to_string(k) + ": wrong number of ramps");
    const double T = piece.duration();
    for (std::size_t j = 0; j < n; ++j) {
      const trajectory::Ramp& r = piece.ramps[j];
      bool finite = std::isfinite(r.x0) && std::isfinite(r.v0);
      bool negative = false;
      for (const auto& seg : r.segments) {
        finite = finite && std::isfinite(seg.duration) && std::isfinite(seg.acceleration);
        negative = negative || seg.duration < -1e-12;
      }
      if (!finite || negative) throw MalformedTrajectory(where(k, j) + ": non-finite values or a negative duration");
      if (std::fabs(r.duration() - T) > 1e-9 * std::max(1.0, T)) throw MalformedTrajectory(where(k, j) + ": duration");
      double v = r.v0;
      if (std::fabs(v) > vmax[j] * (1 + kRel)) throw MalformedTrajectory(where(k, j) + ": velocity beyond its limit");
      for (const auto& seg : r.segments) {
        if (std::fabs(seg.acceleration) > amax[j] * (1 + kRel)) {
          throw MalformedTrajectory(where(k, j) + ": acceleration beyond its limit");
        }
        v = v + seg.acceleration * seg.duration;
        if (std::fabs(v) > vmax[j] * (1 + kRel)) throw MalformedTrajectory(where(k, j) + ": velocity beyond its limit");
      }
    }
    if (!previous.empty()) {
      for (std::size_t j = 0; j < n; ++j) {
        const auto [x, v] = previous[j];
        const trajectory::Ramp& r = piece.ramps[j];
        if (std::fabs(r.x0 - x) > kTolX * std::max(1.0, std::fabs(x)) ||
            std::fabs(r.v0 - v) > kTolV * std::max(1.0, vmax[j])) {
          throw MalformedTrajectory(where(k, j) + ": discontinuous with the piece before");
        }
      }
    }
    previous.clear();
    for (const auto& r : piece.ramps) previous.push_back(r.end());
  }
  for (std::size_t j = 0; j < n; ++j) {
    const double tol = kTolV * std::max(1.0, vmax[j]);
    if (std::fabs(traj.pieces.front().ramps[j].v0) > tol || std::fabs(previous[j].second) > tol) {
      throw MalformedTrajectory("joint " + std::to_string(j) + ": not at rest at both ends");
    }
  }
}

std::optional<int> straight_reference(const trajectory::Piece& piece, const Config& a, const Config& b) {
  const std::size_t n = a.size();
  Config D(n);
  for (std::size_t j = 0; j < n; ++j) D[j] = b[j] - a[j];
  std::size_t ref = 0;
  for (std::size_t j = 1; j < n; ++j) {
    if (std::fabs(D[j]) > std::fabs(D[ref])) ref = j;
  }
  if (D[ref] == 0.0) return std::nullopt;
  const trajectory::Ramp& r0 = piece.ramps[ref];
  for (std::size_t j = 0; j < n; ++j) {
    const trajectory::Ramp& r = piece.ramps[j];
    if (r.segments.size() != r0.segments.size()) return std::nullopt;
    for (std::size_t k = 0; k < r.segments.size(); ++k) {
      if (r.segments[k].duration != r0.segments[k].duration) return std::nullopt;
    }
    const double ratio = D[j] / D[ref];
    if (std::fabs(r.v0 - ratio * r0.v0) > 1e-9 * std::max(1.0, std::fabs(r0.v0))) return std::nullopt;
    for (std::size_t k = 0; k < r.segments.size(); ++k) {
      const double ar = r0.segments[k].acceleration;
      if (std::fabs(r.segments[k].acceleration - ratio * ar) > 1e-9 * std::max(1.0, std::fabs(ar))) return std::nullopt;
    }
  }
  const double sign = D[ref] > 0 ? 1.0 : -1.0;
  double v = r0.v0;
  if (sign * v < -1e-9 * std::max(1.0, std::fabs(v))) return std::nullopt;
  for (const auto& seg : r0.segments) {
    v = v + seg.acceleration * seg.duration;
    if (sign * v < -1e-9 * std::max(1.0, std::fabs(seg.acceleration * seg.duration))) return std::nullopt;
  }
  return static_cast<int>(ref);
}

PathEvidence::PathEvidence(const std::vector<Config>& path, double resolution) : nodes_(path) {
  if (nodes_.empty()) throw MalformedTrajectory("the source path is empty");
  states_.emplace_back(0.0, nodes_.front());
  for (std::size_t k = 0; k + 1 < nodes_.size(); ++k) {
    const Config& q0 = nodes_[k];
    const Config& q1 = nodes_[k + 1];
    Config d(q0.size());
    for (std::size_t j = 0; j < d.size(); ++j) d[j] = q1[j] - q0[j];
    const int n = subdivisions(norm(d), resolution);
    for (int i = 1; i <= n; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(n);
      Config q(q0.size());
      if (i == n) {
        q = q1;
      } else {
        for (std::size_t j = 0; j < q.size(); ++j) q[j] = q0[j] + t * d[j];
      }
      states_.emplace_back(static_cast<double>(k) + t, std::move(q));
    }
  }
}

std::optional<double> PathEvidence::locate(const Config& x) const {
  const double tol = kOnPath * std::max(1.0, max_abs(x));
  const std::size_t n = x.size();
  if (nodes_.size() == 1) {
    Config e(n);
    for (std::size_t j = 0; j < n; ++j) e[j] = nodes_[0][j] - x[j];
    return norm(e) <= tol ? std::optional<double>(0.0) : std::nullopt;
  }
  Config s(n), e(n);
  for (std::size_t k = 0; k + 1 < nodes_.size(); ++k) {
    const Config& p = nodes_[k];
    for (std::size_t j = 0; j < n; ++j) {
      s[j] = nodes_[k + 1][j] - p[j];
      e[j] = x[j] - p[j];
    }
    const double ss = dot(s, s);
    const double f = ss == 0.0 ? 0.0 : std::min(1.0, std::max(0.0, dot(e, s) / ss));
    for (std::size_t j = 0; j < n; ++j) e[j] = p[j] + f * s[j] - x[j];
    if (norm(e) <= tol) return static_cast<double>(k) + f;
  }
  return std::nullopt;
}

std::optional<std::pair<std::vector<std::pair<double, Config>>, int>> PathEvidence::straight_span(
    const trajectory::Piece& piece) const {
  const std::size_t n = piece.ramps.size();
  const double T = piece.duration();
  Config a(n), b(n);
  for (std::size_t j = 0; j < n; ++j) {
    a[j] = piece.ramps[j].position(0.0);
    b[j] = piece.ramps[j].position(T);
  }
  const auto ref = straight_reference(piece, a, b);
  if (!ref) return std::nullopt;
  const auto la = locate(a), lb = locate(b);
  if (!la || !lb || *lb < *la) return std::nullopt;
  Config D(n), e(n);
  for (std::size_t j = 0; j < n; ++j) D[j] = b[j] - a[j];
  const double DD = dot(D, D);
  const double tol = kOnPath * std::max({1.0, max_abs(a), max_abs(b)});
  for (auto k = static_cast<long>(std::floor(*la)) + 1; k < static_cast<long>(std::ceil(*lb)); ++k) {
    const Config& p = nodes_[static_cast<std::size_t>(k)];
    for (std::size_t j = 0; j < n; ++j) e[j] = p[j] - a[j];
    const double f = std::min(1.0, std::max(0.0, dot(e, D) / DD));
    for (std::size_t j = 0; j < n; ++j) e[j] = a[j] + f * D[j] - p[j];
    if (norm(e) > tol) return std::nullopt;
  }
  std::vector<std::pair<double, Config>> out;
  for (const auto& [lam, q] : states_) {
    if (*la < lam && lam < *lb) out.emplace_back(lam, q);
  }
  out.emplace_back(*lb, b);
  return std::pair{std::move(out), *ref};
}

std::vector<double> Validator::time_steps(const trajectory::Piece& piece, const std::vector<double>& vmax) const {
  const double T = piece.duration();
  const int n = std::max(1, static_cast<int>(std::ceil(T * norm(vmax) / policy_.resolution - kRounding)));
  const double step = T / n;
  std::vector<double> ts;
  for (int i = 1; i <= n; ++i) ts.push_back(i == n ? T : i * step);
  for (std::size_t j = 0; j < piece.ramps.size(); ++j) {
    if (!space_->angular().empty() && space_->angular()[j]) continue;
    const trajectory::Ramp& r = piece.ramps[j];
    double start = 0.0, x = r.x0, v = r.v0;
    for (const auto& seg : r.segments) {
      const double d = seg.duration, acc = seg.acceleration;
      if (acc != 0.0) {
        const double tau = -v / acc;
        if (0.0 < tau && tau < d) {
          const double peak = x + v * tau + 0.5 * acc * tau * tau;
          if (!(space_->lower()[j] <= peak && peak <= space_->upper()[j])) ts.push_back(start + tau);
        }
      }
      const double xn = x + v * d + 0.5 * acc * d * d;
      v = v + acc * d;
      x = xn;
      start = start + d;
    }
  }
  std::sort(ts.begin(), ts.end());
  return ts;
}

std::vector<int> spread_order(int n) {
  std::vector<bool> seen(static_cast<std::size_t>(n) + 1, false);
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(n));
  for (int level = 1; static_cast<int>(out.size()) < n && (1LL << level) <= 4LL * n; ++level) {
    const long long denom = 1LL << level;
    for (long long j = 1; j < denom; j += 2) {
      const long long p = (j * n + denom - 1) >> level;
      if (1 <= p && p <= n && !seen[static_cast<std::size_t>(p)]) {
        seen[static_cast<std::size_t>(p)] = true;
        out.push_back(static_cast<int>(p) - 1);
      }
    }
  }
  for (int p = 1; p <= n; ++p) {
    if (!seen[static_cast<std::size_t>(p)]) out.push_back(p - 1);
  }
  return out;
}

std::optional<InvalidWitness> Validator::entry_witness(const Entry& e) const {
  auto w = witness(e.q, e.segment, e.fraction, e.time_s);
  if (w && e.ramp) {
    const auto r = static_cast<std::size_t>(e.ref);
    w->time_s = e.t0 + time_at(*e.ramp, e.q[r], e.ramp_T);
  }
  return w;
}

void Validator::piece_entries(const trajectory::Piece& piece, int index, double t0, const std::vector<double>& vmax,
                              const PathEvidence* evidence, std::vector<Entry>& out) const {
  const double T = piece.duration();
  if (evidence) {
    if (auto span = evidence->straight_span(piece)) {
      auto& [states, ref] = *span;
      for (std::size_t i = 0; i < states.size(); ++i) {
        Entry e;
        e.q = std::move(states[i].second);
        e.segment = index;
        if (i + 1 == states.size()) {
          e.time_s = t0 + T;
        } else {
          e.ramp = &piece.ramps[static_cast<std::size_t>(ref)];
          e.ref = ref;
          e.ramp_T = T;
          e.t0 = t0;
        }
        out.push_back(std::move(e));
      }
      return;
    }
  }
  for (const double t : time_steps(piece, vmax)) {
    Entry e;
    e.q.resize(piece.ramps.size());
    for (std::size_t j = 0; j < e.q.size(); ++j) e.q[j] = piece.ramps[j].position(t);
    e.segment = index;
    e.time_s = t0 + t;
    out.push_back(std::move(e));
  }
}

std::pair<long, std::optional<InvalidWitness>> Validator::check_piece(const trajectory::Piece& piece, int index,
                                                                      double t0, const std::vector<double>& vmax,
                                                                      const PathEvidence* evidence, bool first) const {
  std::vector<Entry> entries;
  piece_entries(piece, index, t0, vmax, evidence, entries);
  return search(entries, first);
}

ValidationReport Validator::validate_trajectory(const trajectory::Trajectory& traj) const {
  well_formed(traj, space_->dof());
  std::optional<PathEvidence> evidence;
  if (!traj.path.empty()) evidence.emplace(traj.path, policy_.resolution);
  const std::vector<double> breaks = traj.breaks();
  std::vector<Entry> entries(1);
  for (const auto& r : traj.pieces.front().ramps) entries[0].q.push_back(r.x0);
  entries[0].segment = 0;
  entries[0].time_s = 0.0;
  for (std::size_t k = 0; k < traj.pieces.size(); ++k) {
    piece_entries(traj.pieces[k], static_cast<int>(k), breaks[k], traj.velocity_limits, evidence ? &*evidence : nullptr,
                  entries);
  }
  auto [checks, w] = search(entries, true);
  ValidationReport r = report("trajectory", static_cast<int>(traj.pieces.size()), checks, std::move(w));
  r.duration_s = breaks.back();
  return r;
}

}  // namespace sscbirrt
