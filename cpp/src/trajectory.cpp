// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// The native form of sscbirrt._trajectory (#214). Each function mirrors its Python namesake expression by expression,
// in the same order of floating-point operations, so the two agree to rounding (this file is built without
// floating-point contraction; see CMakeLists.txt). Python is the reference: change it first.
#include "sscbirrt/trajectory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace sscbirrt::trajectory {

namespace {

constexpr double kEpsT = 1e-12;    // times this close to zero are zero
constexpr double kTolX = 1e-9;     // endpoint tolerance (joint units)
constexpr double kTolRel = 1e-9;   // relative slack on velocity and acceleration bounds in verification

std::vector<Segment> clean(std::initializer_list<Segment> segments) {
  std::vector<Segment> out;
  for (const Segment& s : segments) {
    if (s.duration > kEpsT) out.push_back({std::max(s.duration, 0.0), s.acceleration});
  }
  return out;
}

double max_abs_acceleration(const Ramp& r) {
  double m = 0.0;
  bool any = false;
  for (const Segment& s : r.segments) {
    m = any ? std::max(m, std::fabs(s.acceleration)) : std::fabs(s.acceleration);
    any = true;
  }
  return m;
}

double norm(const std::vector<double>& x) {
  double s = 0.0;
  for (double v : x) s += v * v;
  return std::sqrt(s);
}

// Angle between two nonzero vectors, accurate near zero (sscbirrt.smoothing._angle).
double angle(const Config& a, const Config& b) {
  const double na = norm(a), nb = norm(b);
  double dn = 0.0, sn = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    const double ua = a[k] / na, ub = b[k] / nb;
    dn += (ua - ub) * (ua - ub);
    sn += (ua + ub) * (ua + ub);
  }
  return 2.0 * std::atan2(std::sqrt(dn), std::sqrt(sn));
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// One joint
// ---------------------------------------------------------------------------------------------------------------------

double Ramp::duration() const {
  double d = 0.0;
  for (const Segment& s : segments) d += s.duration;
  return d;
}

std::array<double, 3> Ramp::state(double t) const {
  double x = x0, v = v0;
  double a = segments.empty() ? 0.0 : segments.front().acceleration;
  double remaining = std::min(std::max(t, 0.0), duration());
  for (const Segment& s : segments) {
    a = s.acceleration;
    const double step = std::min(s.duration, remaining);
    const double xn = x + v * step + 0.5 * a * step * step;
    v = v + a * step;
    x = xn;
    remaining -= step;
    if (remaining <= 0.0) break;
  }
  return {x, v, a};
}

double Ramp::position(double t) const {
  t = std::min(std::max(t, 0.0), duration());
  double start = 0.0, xs = x0, vs = v0;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    const double d = segments[i].duration, acc = segments[i].acceleration;
    if (t <= start + d || i + 1 == segments.size()) {
      const double tau = t - start;
      return xs + vs * tau + 0.5 * acc * tau * tau;
    }
    const double xn = xs + vs * d + 0.5 * acc * d * d;
    vs = vs + acc * d;
    xs = xn;
    start = start + d;
  }
  return x0;
}

std::pair<double, double> Ramp::end() const {
  const auto s = state(duration());
  return {s[0], s[1]};
}

std::pair<Ramp, Ramp> Ramp::split(double t) const {
  std::vector<Segment> before, after;
  double elapsed = 0.0;
  for (const Segment& s : segments) {
    if (elapsed + s.duration <= t) {
      before.push_back(s);
    } else if (elapsed >= t) {
      after.push_back(s);
    } else {
      before.push_back({t - elapsed, s.acceleration});
      after.push_back({elapsed + s.duration - t, s.acceleration});
    }
    elapsed += s.duration;
  }
  const auto st = state(t);
  return {Ramp{x0, v0, std::move(before)}, Ramp{st[0], st[1], std::move(after)}};
}

bool verify(const Ramp& ramp, double x1, double v1, double vmax, double amax, std::optional<double> duration) {
  for (const Segment& s : ramp.segments) {
    if (s.duration < -kEpsT) return false;
  }
  for (const Segment& s : ramp.segments) {
    if (std::fabs(s.acceleration) > amax * (1 + kTolRel)) return false;
  }
  double v = ramp.v0;
  if (std::fabs(v) > vmax * (1 + kTolRel)) return false;
  for (const Segment& s : ramp.segments) {
    v += s.acceleration * s.duration;
    if (std::fabs(v) > vmax * (1 + kTolRel)) return false;
  }
  const auto [xe, ve] = ramp.end();
  const double scale = std::max(1.0, std::fabs(x1));
  if (std::fabs(xe - x1) > kTolX * scale || std::fabs(ve - v1) > 1e-7 * std::max(1.0, vmax)) return false;
  return !duration || std::fabs(ramp.duration() - *duration) <= 1e-9 * std::max(1.0, *duration);
}

std::optional<Ramp> min_time(double x0, double v0, double x1, double v1, double vmax, double amax) {
  const double dx = x1 - x0;
  if (std::fabs(dx) <= kTolX && std::fabs(v1 - v0) <= 1e-12) return Ramp{x0, v0, {}};
  std::optional<Ramp> best;
  for (const double s : {1.0, -1.0}) {
    const double a = s * amax;
    const double rad = a * dx + 0.5 * (v0 * v0 + v1 * v1);
    if (rad < 0) continue;
    const double vm = s * std::sqrt(rad);
    Ramp ramp;
    if (std::fabs(vm) <= vmax) {
      const double t1 = (vm - v0) / a, t2 = (vm - v1) / a;
      if (t1 < -kEpsT || t2 < -kEpsT) continue;
      ramp = Ramp{x0, v0, clean({{t1, a}, {t2, -a}})};
    } else {
      const double vc = s * vmax;
      const double t1 = (vc - v0) / a, t3 = (vc - v1) / a;
      if (t1 < -kEpsT || t3 < -kEpsT) continue;
      const double cruise = dx - ((vc * vc - v0 * v0) + (vc * vc - v1 * v1)) / (2 * a);
      const double t2 = cruise / vc;
      if (t2 < -kEpsT) continue;
      ramp = Ramp{x0, v0, clean({{t1, a}, {t2, 0.0}, {t3, -a}})};
    }
    if (verify(ramp, x1, v1, vmax, amax) && (!best || ramp.duration() < best->duration())) best = std::move(ramp);
  }
  return best;
}

std::optional<Ramp> fixed_time(double x0, double v0, double x1, double v1, double T, double vmax, double amax) {
  if (T <= kEpsT) {
    if (std::fabs(x1 - x0) <= kTolX && std::fabs(v1 - v0) <= 1e-12) return Ramp{x0, v0, {}};
    return std::nullopt;
  }
  const double dx = x1 - x0, dv = v1 - v0;
  const double D = dx - 0.5 * (v0 + v1) * T;
  std::vector<Ramp> candidates;
  if (std::fabs(D) <= 1e-14 * std::max(1.0, std::fabs(dx)) && std::fabs(dv) <= 1e-14) {
    candidates.push_back(Ramp{x0, v0, {{T, 0.0}}});
  } else {
    const double disc = 4 * D * D + T * T * dv * dv;
    for (const double a : {(2 * D + std::sqrt(disc)) / (T * T), (2 * D - std::sqrt(disc)) / (T * T)}) {
      if (a == 0.0) continue;
      double t1 = (a * T + dv) / (2 * a);
      if (t1 < -kEpsT || t1 > T + kEpsT) continue;
      t1 = std::min(std::max(t1, 0.0), T);
      candidates.push_back(Ramp{x0, v0, clean({{t1, a}, {T - t1, -a}})});
    }
  }
  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const Ramp& a, const Ramp& b) { return max_abs_acceleration(a) < max_abs_acceleration(b); });
  for (const Ramp& ramp : candidates) {
    if (verify(ramp, x1, v1, vmax, amax, T)) return ramp;
  }
  // Cruise at +-vmax: dx = vc T - s ((vc - v0)**2 + (vc - v1)**2) / (2 A)
  for (const double s : {1.0, -1.0}) {
    const double vc = s * vmax;
    const double denom = 2.0 * (vc * T - dx);
    if (s * denom <= 0) continue;
    const double A = s * ((vc - v0) * (vc - v0) + (vc - v1) * (vc - v1)) / denom;
    if (!(0 < A && A <= amax * (1 + kTolRel))) continue;
    const double a1 = vc != v0 ? std::copysign(A, vc - v0) : A;
    const double a3 = vc != v1 ? std::copysign(A, v1 - vc) : A;
    const double t1 = std::fabs(vc - v0) / A, t3 = std::fabs(vc - v1) / A;
    const double t2 = T - t1 - t3;
    if (t2 < -kEpsT) continue;
    Ramp ramp{x0, v0, clean({{t1, a1}, {t2, 0.0}, {t3, a3}})};
    if (verify(ramp, x1, v1, vmax, amax, T)) return ramp;
  }
  // Full acceleration with a cruise at an intermediate speed vc: with directions s1 = sign(vc - v0) and
  // s3 = sign(v1 - vc), dx = vc T - s1 (vc - v0)**2 / (2 A) + s3 (v1 - vc)**2 / (2 A).
  const double A = amax;
  for (const double s1 : {1.0, -1.0}) {
    for (const double s3 : {1.0, -1.0}) {
      const double qa = (s3 - s1) / (2 * A);
      const double qb = T + (s1 * v0 - s3 * v1) / A;
      const double qc = (s3 * v1 * v1 - s1 * v0 * v0) / (2 * A) - dx;
      std::vector<double> roots;
      if (std::fabs(qa) < 1e-15) {
        if (qb != 0) roots.push_back(-qc / qb);
      } else {
        const double disc = qb * qb - 4 * qa * qc;
        if (disc < 0) continue;
        roots = {(-qb + std::sqrt(disc)) / (2 * qa), (-qb - std::sqrt(disc)) / (2 * qa)};
      }
      for (const double vc : roots) {
        if (std::fabs(vc) > vmax * (1 + kTolRel)) continue;
        if ((vc - v0) * s1 < -1e-12 || (v1 - vc) * s3 < -1e-12) continue;
        const double t1 = std::fabs(vc - v0) / A, t3 = std::fabs(v1 - vc) / A;
        const double t2 = T - t1 - t3;
        if (t2 < -kEpsT) continue;
        Ramp ramp{x0, v0, clean({{t1, s1 * A}, {t2, 0.0}, {t3, s3 * A}})};
        if (verify(ramp, x1, v1, vmax, amax, T)) return ramp;
      }
    }
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------------------
// Many joints
// ---------------------------------------------------------------------------------------------------------------------

double Piece::duration() const {
  double d = 0.0;
  bool any = false;
  for (const Ramp& r : ramps) {
    d = any ? std::max(d, r.duration()) : r.duration();
    any = true;
  }
  return d;
}

std::pair<Piece, Piece> Piece::split(double t) const {
  Piece before{{}, validated}, after{{}, validated};
  for (const Ramp& r : ramps) {
    auto [a, b] = r.split(t);
    before.ramps.push_back(std::move(a));
    after.ramps.push_back(std::move(b));
  }
  return {std::move(before), std::move(after)};
}

std::optional<Piece> synchronize(const Config& q0, const Config& v0, const Config& q1, const Config& v1,
                                 const std::vector<double>& vmax, const std::vector<double>& amax) {
  const std::size_t n = q0.size();
  std::vector<Ramp> mins;
  mins.reserve(n);
  for (std::size_t j = 0; j < n; ++j) {
    auto m = min_time(q0[j], v0[j], q1[j], v1[j], vmax[j], amax[j]);
    if (!m) return std::nullopt;
    mins.push_back(std::move(*m));
  }
  double T0 = 0.0;
  for (std::size_t j = 0; j < n; ++j) T0 = j == 0 ? mins[j].duration() : std::max(T0, mins[j].duration());
  if (T0 <= kEpsT) {
    Piece rest;
    for (std::size_t j = 0; j < n; ++j) rest.ramps.push_back(Ramp{q0[j], v0[j], {}});
    return rest;
  }
  for (double T = T0; T <= 4.0 * T0; T *= 1.02) {
    Piece piece;
    piece.ramps.reserve(n);
    bool ok = true;
    for (std::size_t j = 0; j < n && ok; ++j) {
      if (std::fabs(mins[j].duration() - T) <= 1e-12 * std::max(1.0, T)) {
        piece.ramps.push_back(mins[j]);
        continue;
      }
      auto r = fixed_time(q0[j], v0[j], q1[j], v1[j], T, vmax[j], amax[j]);
      if (!r) {
        ok = false;
      } else {
        piece.ramps.push_back(std::move(*r));
      }
    }
    if (ok) return piece;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------------------------------------------------

std::vector<double> Trajectory::breaks() const {
  std::vector<double> br{0.0};
  double acc = 0.0;
  for (const Piece& p : pieces) {
    acc += p.duration();
    br.push_back(acc);
  }
  return br;
}

double Trajectory::duration() const { return breaks().back(); }

namespace {

std::pair<std::size_t, double> locate(const std::vector<double>& br, std::size_t n_pieces, double t) {
  const auto it = std::upper_bound(br.begin(), br.end(), t);
  long i = static_cast<long>(it - br.begin()) - 1;
  i = std::clamp(i, 0L, static_cast<long>(n_pieces) - 1);
  const auto k = static_cast<std::size_t>(i);
  return {k, t - br[k]};
}

}  // namespace

void Trajectory::state(double t, Config& q, Config& v) const {
  const std::vector<double> br = breaks();
  t = std::min(std::max(t, 0.0), br.back());
  const auto [i, tau] = locate(br, pieces.size(), t);
  const Piece& p = pieces[i];
  q.resize(p.ramps.size());
  v.resize(p.ramps.size());
  for (std::size_t j = 0; j < p.ramps.size(); ++j) {
    const auto s = p.ramps[j].state(tau);
    q[j] = s[0];
    v[j] = s[1];
  }
}

Trajectory Trajectory::splice(double t1, double t2, Piece piece) const {
  const std::vector<double> br = breaks();
  const auto [i1, tau1] = locate(br, pieces.size(), t1);
  const auto [i2, tau2] = locate(br, pieces.size(), t2);
  std::vector<Piece> all(pieces.begin(), pieces.begin() + static_cast<long>(i1));
  all.push_back(pieces[i1].split(tau1).first);
  all.push_back(std::move(piece));
  all.push_back(pieces[i2].split(tau2).second);
  all.insert(all.end(), pieces.begin() + static_cast<long>(i2) + 1, pieces.end());
  Trajectory out{{}, velocity_limits, acceleration_limits, path};
  for (Piece& p : all) {
    if (p.duration() > kEpsT) out.pieces.push_back(std::move(p));
  }
  return out;
}

std::vector<Config> coalesce(const std::vector<Config>& path, double tolerance) {
  if (path.empty()) throw std::invalid_argument("coalesce needs a nonempty path");
  std::vector<Config> pts{path.front()};
  for (std::size_t i = 1; i < path.size(); ++i) {
    Config d(path[i].size());
    for (std::size_t k = 0; k < d.size(); ++k) d[k] = path[i][k] - pts.back()[k];
    if (norm(d) > 0) pts.push_back(path[i]);
  }
  if (pts.size() <= 2) return pts;
  std::vector<Config> out{pts.front()};
  for (std::size_t i = 1; i + 1 < pts.size(); ++i) {
    Config a(pts[i].size()), b(pts[i].size());
    for (std::size_t k = 0; k < a.size(); ++k) {
      a[k] = pts[i][k] - out.back()[k];
      b[k] = pts[i + 1][k] - pts[i][k];
    }
    if (angle(a, b) >= tolerance) out.push_back(pts[i]);
  }
  out.push_back(pts.back());
  return out;
}

Piece straight_line(const Config& q0, const Config& q1, const std::vector<double>& vmax,
                    const std::vector<double>& amax) {
  const std::size_t n = q0.size();
  Config d(n);
  for (std::size_t j = 0; j < n; ++j) d[j] = q1[j] - q0[j];
  const double L = norm(d);
  Piece piece{{}, "planned"};
  if (L == 0.0) {
    for (std::size_t j = 0; j < n; ++j) piece.ramps.push_back(Ramp{q0[j], 0.0, {}});
    return piece;
  }
  double vs = std::numeric_limits<double>::infinity(), as = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j < n; ++j) {
    const double u = std::fabs(d[j]) / L;
    if (u > 0) {
      vs = std::min(vs, vmax[j] / u);
      as = std::min(as, amax[j] / u);
    }
  }
  const auto s = min_time(0.0, 0.0, L, 0.0, vs, as);
  if (!s) throw std::runtime_error("no trapezoid along the chord");  // unreachable for positive limits
  for (std::size_t j = 0; j < n; ++j) {
    Ramp r{q0[j], 0.0, {}};
    for (const Segment& seg : s->segments) r.segments.push_back({seg.duration, seg.acceleration * d[j] / L});
    piece.ramps.push_back(std::move(r));
  }
  return piece;
}

Trajectory initial_trajectory(const std::vector<Config>& corners, const std::vector<double>& vmax,
                              const std::vector<double>& amax) {
  if (corners.empty()) throw std::invalid_argument("initial_trajectory needs at least one corner");
  Trajectory traj{{}, vmax, amax, {}};
  for (std::size_t i = 0; i + 1 < corners.size(); ++i) {
    Piece p = straight_line(corners[i], corners[i + 1], vmax, amax);
    if (p.duration() > kEpsT) traj.pieces.push_back(std::move(p));
  }
  if (traj.pieces.empty()) traj.pieces.push_back(straight_line(corners.front(), corners.front(), vmax, amax));
  return traj;
}

// ---------------------------------------------------------------------------------------------------------------------
// Shortcuts in time
// ---------------------------------------------------------------------------------------------------------------------

std::vector<Config> piece_samples(const Piece& piece, double resolution, const std::vector<double>& vmax) {
  const double T = piece.duration();
  const int n = std::max(1, static_cast<int>(std::ceil(T * norm(vmax) / resolution)));
  const double step = T / n;
  std::vector<Config> out(static_cast<std::size_t>(n) + 1, Config(piece.ramps.size()));
  for (int i = 0; i <= n; ++i) {
    const double t = i == n ? T : i * step;
    for (std::size_t j = 0; j < piece.ramps.size(); ++j) out[static_cast<std::size_t>(i)][j] = piece.ramps[j].position(t);
  }
  return out;
}

std::optional<std::string> validate_samples(const std::vector<Config>& points, const MotionValidator& motion,
                                            const WhyInadmissible& why, bool recheck_returned) {
  if (auto w = why(points.front())) return "sample 0: " + *w;
  for (std::size_t i = 0; i + 1 < points.size(); ++i) {
    const LocalMotion m = motion.validate(points[i], points[i + 1]);
    if (!m.reached) {
      auto w = why(points[i + 1]);
      return "chord " + std::to_string(i) + ": " + (w ? *w : std::string("motion validator rejected it"));
    }
    if (recheck_returned) {
      for (const Config& q : m.configs) {
        if (auto w = why(q)) return "chord " + std::to_string(i) + ": " + *w;
      }
    }
  }
  return std::nullopt;
}

std::optional<std::string> accepts(const Trajectory& traj, double t1, double t2, const Piece& piece,
                                   const Validator& validator, const PathEvidence* evidence) {
  Config start(piece.ramps.size());
  for (std::size_t j = 0; j < piece.ramps.size(); ++j) start[j] = piece.ramps[j].x0;
  if (auto w = validator.check_state(start)) return "start: " + std::string(failure_name(w->failure)) + " (" + w->detail + ")";
  const std::vector<double> br = traj.breaks();
  const auto [i1, tau1] = locate(br, traj.pieces.size(), t1);
  const auto [i2, tau2] = locate(br, traj.pieces.size(), t2);
  const Piece before = traj.pieces[i1].split(tau1).first;
  const Piece after = traj.pieces[i2].split(tau2).second;
  // The new piece first: most candidates fail there. A cut piece still on the source path is not re-checked: its
  // states are the path's (valid when the path is) and the cut point, checked here as the new piece's start or end.
  const std::pair<const char*, const Piece*> parts[] = {{"shortcut", &piece}, {"before", &before}, {"after", &after}};
  for (const auto& [name, part] : parts) {
    if (part->duration() <= kEpsT) continue;
    if (part != &piece && evidence && evidence->straight_span(*part)) continue;
    auto [checks, w] = validator.check_piece(*part, 0, 0.0, traj.velocity_limits, evidence);
    (void)checks;
    if (w) return std::string(name) + ": " + failure_name(w->failure) + " (" + w->detail + ")";
  }
  return std::nullopt;
}

Trajectory shortcut(Trajectory traj, const MotionValidator& motion, const WhyInadmissible& why, bool recheck_returned,
                    double resolution, const ShortcutOptions& options, ShortcutReport* report,
                    const Validator* validator) {
  using Clock = std::chrono::steady_clock;
  Rng rng(options.seed);
  const std::vector<double> vmax = traj.velocity_limits, amax = traj.acceleration_limits;
  ShortcutReport local;
  ShortcutReport& rep = report ? *report : local;
  rep = ShortcutReport{};
  const auto started = Clock::now();
  const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - started).count(); };
  int since_gain = 0;
  rep.stopped = "iterations";
  std::vector<double> history{traj.duration()};  // duration after each number of attempts
  const int window = options.convergence_window.value_or(0);
  std::optional<PathEvidence> evidence;
  if (validator && !traj.path.empty()) evidence.emplace(traj.path, validator->policy().resolution);
  Config qa, va, qb, vb;
  for (int it = 0; it < options.iterations; ++it) {
    if (!rep.attempts.empty()) {
      rep.attempts.back().elapsed = elapsed();
      history.push_back(traj.duration());
    }
    if (options.cancel && options.cancel->cancelled()) {
      rep.stopped = "aborted";
      break;
    }
    if (options.time_budget && elapsed() > *options.time_budget) {
      rep.stopped = "time_budget";
      break;
    }
    if (options.patience && since_gain >= *options.patience) {
      rep.stopped = "patience";
      break;
    }
    const auto k = static_cast<std::size_t>(it);
    if (window > 0 && it >= window &&
        history[k - static_cast<std::size_t>(window)] - history[k] < options.convergence_tolerance * history[k]) {
      rep.stopped = "converged";
      break;
    }
    const double T = traj.duration();
    const double u1 = unit(rng) * T, u2 = unit(rng) * T;
    Attempt& rec = rep.attempts.emplace_back();
    rec.iteration = it;
    rec.t1 = std::min(u1, u2);
    rec.t2 = std::max(u1, u2);
    ++since_gain;
    if (rec.t2 - rec.t1 < options.min_interval) {
      rec.rejected = "window too short";
      continue;
    }
    traj.state(rec.t1, qa, va);
    traj.state(rec.t2, qb, vb);
    std::optional<Piece> piece = synchronize(qa, va, qb, vb, vmax, amax);
    if (!piece) {
      rec.rejected = "no common duration";
      continue;
    }
    const double gain = (rec.t2 - rec.t1) - piece->duration();
    rec.gain = gain;
    if (gain <= 1e-9) {
      rec.rejected = "not faster";
      continue;
    }
    piece->validated = "shortcut";
    std::optional<std::string> refused;
    if (validator) {
      refused = accepts(traj, rec.t1, rec.t2, *piece, *validator, evidence ? &*evidence : nullptr);
    } else {
      refused = validate_samples(piece_samples(*piece, resolution, vmax), motion, why, recheck_returned);
    }
    if (refused) {
      rec.rejected = std::move(*refused);
      continue;
    }
    traj = traj.splice(rec.t1, rec.t2, std::move(*piece));
    rec.accepted = true;
    ++rep.accepted;
    since_gain = 0;
  }
  rep.seconds = elapsed();
  if (!rep.attempts.empty()) rep.attempts.back().elapsed = rep.seconds;
  return traj;
}

Trajectory smooth_path_in_time(const Planner& planner, const PlanningProblem& problem, const std::vector<Config>& path,
                               const std::vector<double>& vmax, const std::vector<double>& amax,
                               const ShortcutOptions& options, ShortcutReport* report) {
  if (!problem.space || !problem.validator) throw std::invalid_argument("problem.space and problem.validator must not be null");
  if (path.empty()) throw std::invalid_argument("path must not be empty");
  const std::size_t dof = static_cast<std::size_t>(problem.space->dof());
  if (vmax.size() != dof || amax.size() != dof) throw std::invalid_argument("limits must have one entry per joint");
  for (std::size_t j = 0; j < dof; ++j) {
    if (!(vmax[j] > 0 && amax[j] > 0 && std::isfinite(vmax[j]) && std::isfinite(amax[j]))) {
      throw std::invalid_argument("velocity and acceleration limits must be positive and finite");
    }
  }
  std::shared_ptr<const MotionValidator> motion = problem.motion_validator;
  const bool custom = static_cast<bool>(motion);
  if (!custom) motion = planner.default_motion_validator(problem);
  const WhyInadmissible why = [&problem](ConfigView q) { return Planner::why_inadmissible(problem, q); };
  std::optional<Validator> validator;
  if (!custom) validator.emplace(Validator::for_problem(problem, planner.config()));
  Trajectory initial = initial_trajectory(coalesce(path), vmax, amax);
  initial.path = path;
  // A custom validator's configurations are re-checked, as the planner's edge rule does.
  return shortcut(std::move(initial), *motion, why, custom, planner.config().resolution(), options, report,
                  validator ? &*validator : nullptr);
}

}  // namespace sscbirrt::trajectory
