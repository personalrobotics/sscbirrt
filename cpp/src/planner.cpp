// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>

#include "sscbirrt/errors.hpp"

namespace sscbirrt {

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

const char* status_name(Status s) {
  switch (s) {
    case Status::Success: return "success";
    case Status::Timeout: return "timeout";
    case Status::Aborted: return "aborted";
    case Status::MaxIterations: return "max_iterations";
  }
  return "?";
}

Tree::Tree(const std::vector<Config>& roots, const std::vector<Provenance>& sources) {
  nodes_.reserve(roots.size());
  for (std::size_t i = 0; i < roots.size(); ++i) {
    nodes_.push_back(Node{roots[i], -1, i < sources.size() ? sources[i] : Provenance{}});
  }
}

int Tree::add(Config q, int parent) {
  nodes_.push_back(Node{std::move(q), parent, {}});
  return static_cast<int>(nodes_.size()) - 1;
}

int Tree::add_root(Config q, Provenance source) {
  nodes_.push_back(Node{std::move(q), -1, std::move(source)});
  return static_cast<int>(nodes_.size()) - 1;
}

int Tree::nearest(const JointSpace& space, ConfigView q) const {
  int best = 0;
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < nodes_.size(); ++i) {
    const double d = space.distance(nodes_[i].q, q);
    if (d < best_d) {
      best_d = d;
      best = static_cast<int>(i);
    }
  }
  return best;
}

std::vector<Config> Tree::path_to_root(int idx) const {
  std::vector<Config> path;
  for (int i = idx; i >= 0; i = nodes_[static_cast<std::size_t>(i)].parent) path.push_back(nodes_[static_cast<std::size_t>(i)].q);
  std::reverse(path.begin(), path.end());
  return path;
}

const Provenance& Tree::root_source(int idx) const {
  int i = idx;
  while (nodes_[static_cast<std::size_t>(i)].parent >= 0) i = nodes_[static_cast<std::size_t>(i)].parent;
  return nodes_[static_cast<std::size_t>(i)].source;
}

// ---------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

struct AbortedDuringRoots {
  std::string role;
  std::vector<Sample> roots;
};

std::string fmt(double v) {
  std::ostringstream os;
  os.precision(4);
  os << v;
  return os.str();
}

// One solve: owns the RNG, the trees, and the deadline. Borrows the problem and config.
class Solve {
 public:
  Solve(const PlannerConfig& config, const PlanningProblem& problem, const SolveOptions& options)
      : cfg_(config), p_(problem), opt_(options), rng_(options.seed ? *options.seed : std::random_device{}()) {}

  PlanResult run();

 private:
  bool aborted() const { return opt_.cancel && opt_.cancel->cancelled(); }
  bool admissible(ConfigView q) const {
    const auto t = Clock::now();
    const bool ok = !Planner::why_inadmissible(p_, q).has_value();
    ++stats_.state_checks;
    stats_.seconds_state_checks += std::chrono::duration<double>(Clock::now() - t).count();
    return ok;
  }
  std::optional<std::string> why_inadmissible_counted(ConfigView q) const {
    const auto t = Clock::now();
    auto why = Planner::why_inadmissible(p_, q);
    ++stats_.state_checks;
    stats_.seconds_state_checks += std::chrono::duration<double>(Clock::now() - t).count();
    return why;
  }
  // As Planner::why_inadmissible, with the validator's verdict already known (#200).
  std::optional<std::string> why_inadmissible_counted(ConfigView q, bool valid) const {
    const auto t = Clock::now();
    std::optional<std::string> why;
    if (auto w = p_.space->why_invalid(q)) why = "outside joint space (" + *w + ")";
    else if (!valid) why = std::string("in collision");
    else if (p_.path_constraint && !p_.path_constraint->contains(q)) why = std::string("violates path constraints");
    ++stats_.state_checks;
    stats_.seconds_state_checks += std::chrono::duration<double>(Clock::now() - t).count();
    return why;
  }
  std::vector<Sample> sample_counted(const StateSet& s) {
    const auto t = Clock::now();
    std::vector<Sample> out = s.sampler()->sample(rng_);
    ++stats_.set_samples;
    stats_.seconds_set_samples += std::chrono::duration<double>(Clock::now() - t).count();
    return out;
  }
  std::optional<Config> project_counted(const SetProjector& proj, ConfigView a, ConfigView b) const {
    const auto t = Clock::now();
    std::optional<Config> out = proj.project(a, b);
    ++stats_.set_projections;
    stats_.seconds_set_projections += std::chrono::duration<double>(Clock::now() - t).count();
    return out;
  }
  const MotionValidator& motion_validator() {
    if (p_.motion_validator) return *p_.motion_validator;
    if (!default_mv_) {
      default_mv_ = std::make_shared<DiscreteMotionValidator>(
          p_.space, [this](ConfigView q) { return admissible(q); }, cfg_.resolution());
    }
    return *default_mv_;
  }

  std::vector<Sample> roots(const StateSet& s, const std::string& role, RootReport& report);
  // One sampling draw: its admissible candidates, at most min(max_per_draw, room), skipping explicit seeds (#196).
  std::vector<Sample> draw_roots(const StateSet& s, const std::vector<Sample>& seeds, int room, RootReport* report);
  std::pair<int, bool> grow(Tree& tree, ConfigView target, std::optional<int> max_steps);
  std::pair<int, bool> extend_along_edge(Tree& tree, int start_idx, ConfigView target);
  std::vector<Config> extract_path(const Tree& ts, const Tree& tg, bool a_is_start, int idx_a, int idx_b) const;
  std::vector<Config> smooth(const std::vector<Config>& path);
  std::optional<std::vector<Config>> try_shortcut(ConfigView from, ConfigView to);
  // Endpoint windings (#212): nullopt when neither endpoint moved.
  std::optional<std::vector<Config>> shorten_endpoints(const std::vector<Config>& path);
  std::optional<std::vector<Config>> shorten_end(const std::vector<Config>& path, const StateSet& role);
  std::optional<std::vector<Config>> unwind(const std::vector<Config>& path, const std::vector<double>& prefix,
                                            const Config& target, double done);
  std::vector<Config> windings(const Config& q) const;
  PlanResult failure(Status status, std::string reason, int iterations) const;

  const PlannerConfig& cfg_;
  const PlanningProblem& p_;
  const SolveOptions& opt_;
  Rng rng_;
  std::shared_ptr<DiscreteMotionValidator> default_mv_;
  std::shared_ptr<Tree> tree_start_, tree_goal_;
  RootReport start_report_, goal_report_;
  mutable SolveStats stats_;
  Clock::time_point t0_;
};

std::vector<Sample> Solve::roots(const StateSet& s, const std::string& role, RootReport& report) {
  std::vector<Sample> out;
  auto count_reason = [&report](const std::string& why) {
    if (why == "in collision") ++report.in_collision;
    else if (why.rfind("outside joint space", 0) == 0) ++report.outside_space;
    else ++report.constraint_violated;
  };

  std::vector<Sample> explicit_seeds = s.seeds();
  report.explicit_candidates = static_cast<int>(explicit_seeds.size());
  for (Sample& m : explicit_seeds) {
    if (auto why = why_inadmissible_counted(m.q)) {
      ++report.explicit_rejected;
      count_reason(*why);
      const int legacy = m.source.empty() ? 0 : m.source.back();
      report.details.push_back(role + "[" + std::to_string(legacy) + "]: " + *why);
    } else {
      out.push_back(m);
    }
  }

  const bool sampled = !s.is_finite() && s.sampler() != nullptr;
  if (sampled) {
    for (int draw = 0; draw < cfg_.sample_draws; ++draw) {
      if (static_cast<int>(out.size()) >= cfg_.num_tree_roots) break;
      if (aborted()) throw AbortedDuringRoots{role, out};
      for (Sample& c : draw_roots(s, explicit_seeds, cfg_.num_tree_roots - static_cast<int>(out.size()), &report)) {
        out.push_back(std::move(c));
      }
    }
  }

  report.roots = static_cast<int>(out.size());
  if (!out.empty()) return out;

  const std::string lower_role = role == "Start" ? "start" : "goal";
  if (report.explicit_candidates == 0 && report.rejections() == 0) {
    throw std::invalid_argument("No valid " + lower_role + " configurations available. Provide either " + lower_role +
                                " or " + lower_role + "_tsrs.");
  }
  std::string message = "All " + lower_role + " configurations " + (report.only_collisions() ? "in collision" : "invalid") +
                        " (" + std::to_string(report.candidates()) + " candidates)";
  for (const std::string& d : report.details) message += "; " + d;
  if (report.draws > 0) message += "; sampling: " + report.summary();
  throw NoRoots(lower_role, report, message);
}

std::vector<Sample> Solve::draw_roots(const StateSet& s, const std::vector<Sample>& seeds, int room, RootReport* report) {
  std::vector<Sample> out;
  if (report) ++report->draws;
  std::vector<Sample> candidates = sample_counted(s);
  if (candidates.empty()) {
    if (report) ++report->draws_empty;
    return out;
  }
  const int limit = std::min(cfg_.max_per_draw.value_or(std::numeric_limits<int>::max()), room);
  if (static_cast<int>(candidates.size()) > limit) {
    // Visit a draw with more candidates than can be kept in a random order so the kept ones are a uniform subset,
    // not the first corner of an enumeration (#168). Fisher-Yates on index(), not std::shuffle, for portability.
    for (std::size_t k = candidates.size() - 1; k > 0; --k) std::swap(candidates[k], candidates[index(rng_, k + 1)]);
  }
  // Windings of one physical configuration share the validator's verdict when it declares that full turns cannot
  // change it (#200); the joint-space and path-constraint checks stay per candidate.
  const bool share = p_.validator->full_turn_invariant();
  std::map<std::vector<std::int64_t>, bool> verdicts;
  for (Sample& c : candidates) {
    if (static_cast<int>(out.size()) >= limit) break;
    const bool repeats_seed =
        std::any_of(seeds.begin(), seeds.end(), [&c](const Sample& e) { return e.source == c.source; });
    if (repeats_seed) continue;  // an explicit seed drawn again; already a root or already rejected
    std::optional<std::string> verdict;
    const auto known = share && !c.key.empty() ? verdicts.find(c.key) : verdicts.end();
    if (known != verdicts.end()) {
      verdict = why_inadmissible_counted(c.q, known->second);
      ++stats_.reused_verdicts;
    } else {
      verdict = why_inadmissible_counted(c.q);
      // The validator ran unless the joint-space check failed first.
      if (share && !c.key.empty() && !(verdict && verdict->rfind("outside joint space", 0) == 0)) {
        verdicts.emplace(c.key, !(verdict && *verdict == "in collision"));
      }
    }
    if (auto why = verdict) {
      if (report) {
        if (*why == "in collision") ++report->in_collision;
        else if (why->rfind("outside joint space", 0) == 0) ++report->outside_space;
        else ++report->constraint_violated;
      }
    } else {
      out.push_back(std::move(c));
    }
  }
  return out;
}

std::pair<int, bool> Solve::grow(Tree& tree, ConfigView target, std::optional<int> max_steps) {
  const JointSpace& space = *p_.space;
  const SetProjector* projector = p_.path_constraint ? p_.path_constraint->projector() : nullptr;

  if (!space.contains(target)) return {tree.nearest(space, target), false};

  int current = tree.nearest(space, target);
  int steps = 0;
  double prev_distance = std::numeric_limits<double>::infinity();

  while (true) {
    const Config& q_current = tree.nodes()[static_cast<std::size_t>(current)].q;
    const Config direction = space.direction(q_current, target);
    double distance = 0.0;
    for (double v : direction) distance += v * v;
    distance = std::sqrt(distance);

    if (distance == 0.0) return {current, true};  // already there; no duplicate node
    if (distance < cfg_.connection_tolerance) return extend_along_edge(tree, current, target);
    if (prev_distance - distance < cfg_.progress_tolerance) break;
    prev_distance = distance;
    if (max_steps && steps >= *max_steps) break;

    const double scale = std::min(distance, cfg_.step_size) / distance;
    Config q_new(direction.size());
    for (std::size_t i = 0; i < q_new.size(); ++i) q_new[i] = q_current[i] + scale * direction[i];
    if (!space.contains(q_new)) break;

    if (projector) {
      std::optional<Config> projected = project_counted(*projector, q_current, q_new);
      if (!projected) break;
      q_new = std::move(*projected);
      if (!space.contains(q_new)) break;  // a projector may move the point anywhere
    }

    const auto [idx, reached] = extend_along_edge(tree, current, q_new);
    current = idx;
    ++steps;
    if (!reached) break;
  }
  return {current, false};
}

std::pair<int, bool> Solve::extend_along_edge(Tree& tree, int start_idx, ConfigView target) {
  const JointSpace& space = *p_.space;
  if (!space.contains(target)) return {start_idx, false};
  const Config q_from = tree.nodes()[static_cast<std::size_t>(start_idx)].q;  // copy: tree may reallocate
  if (space.distance(q_from, target) == 0.0) return {start_idx, true};

  const MotionValidator& validator = motion_validator();
  const auto t_edge = Clock::now();
  LocalMotion motion = validator.validate(q_from, target);
  ++stats_.edge_checks;
  stats_.seconds_edge_checks += std::chrono::duration<double>(Clock::now() - t_edge).count();

  const Config target_c = to_config(target);
  if (motion.reached) {
    if (motion.configs.empty()) {
      throw ContractError("motion validator reported reached=true with no configurations for a motion of length " +
                          fmt(space.distance(q_from, target)));
    }
    if (motion.configs.back() != target_c) {
      throw ContractError("motion validator reported reached=true but did not end at the exact target");
    }
  }
  for (const Config& q : motion.configs) {
    if (q.size() != static_cast<std::size_t>(space.dof())) {
      throw ContractError("motion validator returned a configuration of length " + std::to_string(q.size()) +
                          " for a space of dof " + std::to_string(space.dof()));
    }
  }

  if (p_.motion_validator) {  // a custom validator: re-check what it returned
    std::size_t first_bad = motion.configs.size();
    for (std::size_t i = 0; i < motion.configs.size(); ++i) {
      if (!admissible(motion.configs[i])) {
        first_bad = i;
        break;
      }
    }
    if (motion.reached && first_bad < motion.configs.size()) return {start_idx, false};  // claimed success with an inadmissible state
    if (!motion.reached) motion.configs.resize(first_bad);
  }

  int current = start_idx;
  for (Config& q : motion.configs) current = tree.add(std::move(q), current);
  return {current, motion.reached};
}

std::vector<Config> Solve::extract_path(const Tree& ts, const Tree& tg, bool a_is_start, int idx_a, int idx_b) const {
  std::vector<Config> from_start = ts.path_to_root(a_is_start ? idx_a : idx_b);
  std::vector<Config> from_goal = tg.path_to_root(a_is_start ? idx_b : idx_a);
  std::vector<Config> path = std::move(from_start);
  path.insert(path.end(), from_goal.rbegin(), from_goal.rend());
  std::vector<Config> deduped;
  deduped.push_back(path.front());
  for (std::size_t i = 1; i < path.size(); ++i) {
    if (path[i] != deduped.back()) deduped.push_back(path[i]);
  }
  return deduped;
}

// Endpoint windings (#212); the reference is CBiRRT._shorten_endpoints. Deterministic: no random draws.
constexpr double kUnwindBy[] = {0.25, 0.5, 1.0};

std::optional<std::vector<Config>> Solve::shorten_endpoints(const std::vector<Config>& path) {
  std::vector<Config> out = path;
  bool moved = false;
  for (const bool goal : {true, false}) {
    if (aborted()) break;
    const StateSet& role = goal ? *p_.goal : *p_.start;
    if (role.is_finite()) continue;  // explicit members, which goal_index names
    std::vector<Config> oriented = out;
    if (!goal) std::reverse(oriented.begin(), oriented.end());
    if (auto shortened = shorten_end(oriented, role)) {
      if (!goal) std::reverse(shortened->begin(), shortened->end());
      out = std::move(*shortened);
      moved = true;
    }
  }
  if (!moved) return std::nullopt;
  return out;
}

std::optional<std::vector<Config>> Solve::shorten_end(const std::vector<Config>& path, const StateSet& role) {
  if (path.size() < 2) return std::nullopt;
  const JointSpace& space = *p_.space;
  std::vector<double> prefix{0.0};  // path length up to each waypoint
  for (std::size_t i = 0; i + 1 < path.size(); ++i) prefix.push_back(prefix.back() + space.distance(path[i], path[i + 1]));
  const double total = prefix.back();
  auto length = [&space](const std::vector<Config>& seg) {
    double s = 0.0;
    for (std::size_t i = 0; i + 1 < seg.size(); ++i) s += space.distance(seg[i], seg[i + 1]);
    return s;
  };

  std::vector<Config> candidates = windings(path.back());
  std::vector<double> bound(candidates.size());
  for (std::size_t c = 0; c < candidates.size(); ++c) bound[c] = space.distance(path.front(), candidates[c]);
  std::vector<std::size_t> order(candidates.size());
  for (std::size_t c = 0; c < order.size(); ++c) order[c] = c;
  std::stable_sort(order.begin(), order.end(), [&bound](std::size_t a, std::size_t b) { return bound[a] < bound[b]; });

  int attempts = 0;
  for (const std::size_t c : order) {
    if (attempts >= cfg_.smoothing_iterations) break;
    if (bound[c] >= total - 1e-9) break;  // sorted by this bound: no later candidate can be shorter either
    const Config& target = candidates[c];
    if (!role.contains(target) || !admissible(target)) continue;
    for (const double done : kUnwindBy) {
      if (attempts >= cfg_.smoothing_iterations || aborted()) return std::nullopt;
      ++attempts;
      if (auto unwound = unwind(path, prefix, target, done)) {
        if (length(*unwound) < total - 1e-9) return unwound;
      }
    }
  }
  return std::nullopt;
}

std::optional<std::vector<Config>> Solve::unwind(const std::vector<Config>& path, const std::vector<double>& prefix,
                                                 const Config& target, double done) {
  // The change of winding is made along the first `done` of the path's length; every waypoint after that is moved by
  // whole turns, the same pose, so the end of the path is physically unchanged. Every edge is re-validated.
  const double reach = done * prefix.back();
  const std::size_t dof = target.size();
  std::vector<Config> moved(path.size(), Config(dof));
  for (std::size_t i = 0; i < path.size(); ++i) {
    const double w = std::min(1.0, prefix[i] / reach);
    for (std::size_t k = 0; k < dof; ++k) moved[i][k] = path[i][k] + w * (target[k] - path.back()[k]);
  }
  moved.front() = path.front();
  moved.back() = target;
  for (std::size_t i = 1; i + 1 < moved.size(); ++i) {
    if (!admissible(moved[i])) return std::nullopt;  // an edge would end there; states are cheaper than edges
  }
  // Past the turn, an edge is the path's own validated edge moved by whole turns. When the validator declares its
  // verdict unchanged by full turns (#200) and nothing else judges motion, that edge needs no new check.
  const bool same = p_.validator->full_turn_invariant() && !p_.path_constraint && !p_.motion_validator;
  std::vector<Config> out{moved.front()};
  for (std::size_t i = 0; i + 1 < moved.size(); ++i) {
    if (same && prefix[i] >= reach) {
      out.push_back(moved[i + 1]);
      continue;
    }
    auto edge = try_shortcut(moved[i], moved[i + 1]);
    if (!edge) return std::nullopt;
    out.insert(out.end(), edge->begin() + 1, edge->end());
  }
  return out;
}

std::vector<Config> Solve::windings(const Config& q) const {
  // q + 2*pi*k, each k_j in {-1, 0, 1}, lexicographic over joints, within the limits. Only bounded joints whose limits
  // admit a full turn are offset; with more than 10 such joints none are proposed (up to 3^10 candidates).
  const JointSpace& space = *p_.space;
  const double two_pi = 2.0 * std::numbers::pi;
  std::vector<std::vector<double>> choices(q.size());
  int wide = 0;
  for (std::size_t j = 0; j < q.size(); ++j) {
    choices[j].push_back(0.0);
    if (!space.angular()[j]) {
      choices[j].clear();
      for (const double o : {-two_pi, 0.0, two_pi}) {
        if (o == 0.0 || (space.lower()[j] <= q[j] + o && q[j] + o <= space.upper()[j])) choices[j].push_back(o);
      }
    }
    if (choices[j].size() > 1) ++wide;
  }
  std::vector<Config> out;
  if (wide > 10) return out;
  std::vector<std::size_t> digit(q.size(), 0);
  while (true) {
    bool any = false;
    Config candidate(q.size());
    for (std::size_t j = 0; j < q.size(); ++j) {
      const double o = choices[j][digit[j]];
      any = any || o != 0.0;
      candidate[j] = q[j] + o;
    }
    if (any && space.contains(candidate)) out.push_back(std::move(candidate));
    std::size_t j = q.size();  // odometer, the last joint fastest (itertools.product)
    while (j > 0 && ++digit[j - 1] == choices[j - 1].size()) digit[--j] = 0;
    if (j == 0) break;
  }
  return out;
}

std::optional<std::vector<Config>> Solve::try_shortcut(ConfigView from, ConfigView to) {
  Tree temp(std::vector<Config>{to_config(from)});
  const auto [idx, reached] = grow(temp, to, std::nullopt);
  if (!reached) return std::nullopt;
  return temp.path_to_root(idx);
}

std::vector<Config> Solve::smooth(const std::vector<Config>& path) {
  if (path.size() <= 2) return path;
  const JointSpace& space = *p_.space;
  auto length = [&space](const std::vector<Config>& seg, std::size_t a, std::size_t b) {
    double s = 0.0;
    for (std::size_t i = a; i < b; ++i) s += space.distance(seg[i], seg[i + 1]);
    return s;
  };

  std::vector<Config> smoothed = path;
  int without_improvement = 0;
  for (int it = 0; it < cfg_.smoothing_iterations; ++it) {
    if (smoothed.size() <= 2) break;
    if (without_improvement >= cfg_.smoothing_patience) break;
    if (aborted()) break;  // a valid path exists; return it as smoothed so far

    const std::size_t n = smoothed.size();
    const std::size_t i = index(rng_, n - 2);            // [0, n-3]
    const std::size_t j = i + 2 + index(rng_, n - (i + 2));  // [i+2, n-1]

    bool improved = false;
    if (auto shortcut = try_shortcut(smoothed[i], smoothed[j])) {
      if (length(*shortcut, 0, shortcut->size() - 1) < length(smoothed, i, j) - 1e-9) {
        std::vector<Config> next(smoothed.begin(), smoothed.begin() + static_cast<std::ptrdiff_t>(i));
        next.insert(next.end(), shortcut->begin(), shortcut->end());
        next.insert(next.end(), smoothed.begin() + static_cast<std::ptrdiff_t>(j) + 1, smoothed.end());
        smoothed = std::move(next);
        improved = true;
      }
    }
    without_improvement = improved ? 0 : without_improvement + 1;
  }
  return smoothed;
}

PlanResult Solve::failure(Status status, std::string reason, int iterations) const {
  if (stats_.seconds_search == 0.0 && tree_start_) stats_.seconds_search = std::chrono::duration<double>(Clock::now() - t0_).count();
  PlanResult r;
  r.status = status;
  r.reason = std::move(reason);
  r.iterations = iterations;
  r.planning_seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
  r.tree_sizes = {tree_start_ ? tree_start_->size() : 0, tree_goal_ ? tree_goal_->size() : 0};
  r.start_roots = start_report_;
  r.goal_roots = goal_report_;
  r.stats = stats_;
  if (opt_.keep_trees) {
    r.tree_start = tree_start_;
    r.tree_goal = tree_goal_;
  }
  return r;
}

PlanResult Solve::run() {
  t0_ = Clock::now();
  const auto t_roots = Clock::now();
  // Roots. Cancellation here is a search failure: Aborted, with what was gathered.
  std::vector<Sample> start_roots, goal_roots;
  auto make_tree = [](const std::vector<Sample>& roots) {
    std::vector<Config> qs;
    std::vector<Provenance> src;
    for (const Sample& s : roots) {
      qs.push_back(s.q);
      src.push_back(s.source);
    }
    return std::make_shared<Tree>(qs, src);
  };
  try {
    start_roots = roots(*p_.start, "Start", start_report_);
    goal_roots = roots(*p_.goal, "Goal", goal_report_);
  } catch (const AbortedDuringRoots& a) {
    if (a.role == "Start") start_roots = a.roots;
    else goal_roots = a.roots;
    tree_start_ = make_tree(start_roots);
    tree_goal_ = make_tree(goal_roots);
    const std::string role = a.role == "Start" ? "start" : "goal";
    return failure(Status::Aborted, "Aborted by user during " + role + " root collection", 0);
  }
  tree_start_ = make_tree(start_roots);
  tree_goal_ = make_tree(goal_roots);
  stats_.seconds_roots = std::chrono::duration<double>(Clock::now() - t_roots).count();

  t0_ = Clock::now();  // the deadline starts with the search, as in Python
  const auto deadline = t0_ + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(cfg_.timeout_seconds));
  // CBiRRT's P_sample (#196): a set that is not finite and can be sampled keeps adding roots during the search. A
  // finite set's members are all roots already, so its tree never tosses the coin.
  const bool start_grows = !p_.start->is_finite() && p_.start->sampler() != nullptr;
  const bool goal_grows = !p_.goal->is_finite() && p_.goal->sampler() != nullptr;
  const std::vector<Sample> start_seeds = start_grows ? p_.start->seeds() : std::vector<Sample>{};
  const std::vector<Sample> goal_seeds = goal_grows ? p_.goal->seeds() : std::vector<Sample>{};
  const SpaceSampler& free_sampler = p_.sampler ? *p_.sampler : static_cast<const SpaceSampler&>(*p_.space);
  const JointSpace& space = *p_.space;

  for (int iteration = 0; iteration < cfg_.max_iterations; ++iteration) {
    if (aborted()) return failure(Status::Aborted, "Aborted by user", iteration);
    if (Clock::now() > deadline) {
      return failure(Status::Timeout,
                     "Timeout after " + fmt(cfg_.timeout_seconds) + "s, " + std::to_string(iteration) + " iterations. Trees: start=" +
                         std::to_string(tree_start_->size()) + " nodes, goal=" + std::to_string(tree_goal_->size()) +
                         " nodes. Trees could not connect.",
                     iteration);
    }

    const bool a_is_start = iteration % 2 == 0;
    Tree& tree_a = a_is_start ? *tree_start_ : *tree_goal_;
    Tree& tree_b = a_is_start ? *tree_goal_ : *tree_start_;

    // Heads: this turn adds roots to tree_a from its own set, and that is the whole turn
    const bool grows = a_is_start ? start_grows : goal_grows;
    const double p_sample = a_is_start ? cfg_.start_sample_probability : cfg_.goal_sample_probability;
    if (grows && unit(rng_) < p_sample) {
      const StateSet& own = a_is_start ? *p_.start : *p_.goal;
      for (Sample& c : draw_roots(own, a_is_start ? start_seeds : goal_seeds, std::numeric_limits<int>::max(), nullptr)) {
        tree_a.add_root(std::move(c.q), std::move(c.source));
        ++stats_.search_roots;
      }
      continue;
    }

    // Tails: an ordinary turn toward a random configuration
    const Config q_sample = free_sampler.sample(rng_);
    const auto [grow_idx, grown] = grow(tree_a, q_sample, cfg_.extend_steps);
    (void)grown;
    const Config q_reached = tree_a.nodes()[static_cast<std::size_t>(grow_idx)].q;
    const auto [connect_idx, connected] = grow(tree_b, q_reached, cfg_.connect_steps);

    if (connected) {
      stats_.seconds_search = std::chrono::duration<double>(Clock::now() - t0_).count();
      std::vector<Config> path = extract_path(*tree_start_, *tree_goal_, a_is_start, grow_idx, connect_idx);
      const auto t_smooth = Clock::now();
      if (cfg_.smooth_path) {
        path = smooth(path);
        if (cfg_.shorten_endpoints) {
          if (auto shortened = shorten_endpoints(path)) path = smooth(*shortened);  // shortcut the unwound path too
        }
      }
      stats_.seconds_smoothing = std::chrono::duration<double>(Clock::now() - t_smooth).count();
      path = space.unwrap_path(path);

      PlanResult r;
      r.status = Status::Success;
      r.path = std::move(path);
      r.start_source = a_is_start ? tree_start_->root_source(grow_idx) : tree_start_->root_source(connect_idx);
      r.goal_source = a_is_start ? tree_goal_->root_source(connect_idx) : tree_goal_->root_source(grow_idx);
      r.iterations = iteration + 1;
      r.planning_seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
      r.tree_sizes = {tree_start_->size(), tree_goal_->size()};
      r.start_roots = start_report_;
      r.goal_roots = goal_report_;
      r.stats = stats_;
      if (opt_.keep_trees) {
        r.tree_start = tree_start_;
        r.tree_goal = tree_goal_;
      }
      return r;
    }
  }
  stats_.seconds_search = std::chrono::duration<double>(Clock::now() - t0_).count();
  return failure(Status::MaxIterations,
                 "Max iterations (" + std::to_string(cfg_.max_iterations) + ") reached. Trees: start=" +
                     std::to_string(tree_start_->size()) + " nodes, goal=" + std::to_string(tree_goal_->size()) +
                     " nodes. Trees could not connect.",
                 cfg_.max_iterations);
}

}  // namespace

Planner::Planner(PlannerConfig config) : config_(std::move(config)) { config_.validate(); }

std::optional<std::string> Planner::why_inadmissible(const PlanningProblem& problem, ConfigView q) {
  if (auto why = problem.space->why_invalid(q)) return "outside joint space (" + *why + ")";
  if (!problem.validator->is_valid(q)) return std::string("in collision");
  if (problem.path_constraint && !problem.path_constraint->contains(q)) return std::string("violates path constraints");
  return std::nullopt;
}

std::shared_ptr<DiscreteMotionValidator> Planner::default_motion_validator(const PlanningProblem& problem) const {
  return std::make_shared<DiscreteMotionValidator>(
      problem.space, [problem](ConfigView q) { return !why_inadmissible(problem, q).has_value(); }, config_.resolution());
}

PlanResult Planner::solve(const PlanningProblem& problem, const SolveOptions& options) const {
  config_.validate();
  if (!problem.space) throw std::invalid_argument("problem.space must not be null");
  if (!problem.start || !problem.goal) throw std::invalid_argument("problem.start and problem.goal must not be null");
  if (!problem.validator) throw std::invalid_argument("problem.validator must not be null");
  for (const auto& [name, s] : {std::pair{"start", problem.start.get()}, std::pair{"goal", problem.goal.get()}}) {
    if (!s->is_finite() && s->sampler() == nullptr) {
      throw UnsupportedCapability(std::string(name) + " set must be finite or sampleable: " + s->describe());
    }
  }
  Solve solve(config_, problem, options);
  return solve.run();
}

}  // namespace sscbirrt
