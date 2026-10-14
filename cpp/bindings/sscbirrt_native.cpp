// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// sscbirrt._native: the only target that sees Python. Exposes the native types and one solve;
// the GIL is released for the duration of solve and no Python object is touched after entry.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "sscbirrt/sscbirrt.hpp"
#include "sscbirrt/trajectory.hpp"
#if SSCBIRRT_HAS_SSIK
#include "sscbirrt/ssik/ssik_arm.hpp"
#include "ssik_cpp/pybind11/solver_data.hpp"
#include "sscbirrt/tsr/tsr.hpp"
#include "sscbirrt/tsr/tsr_set.hpp"
#endif

namespace py = pybind11;
using namespace sscbirrt;

namespace {

using Rows4 = std::array<std::array<double, 4>, 4>;
using Rows6x2 = std::array<std::array<double, 2>, 6>;

Transform from_rows(const Rows4& rows) { return Transform::from_rows(rows); }
sstsr::Transform pose_rows(const Rows4& rows) { return tsr::to_sstsr(from_rows(rows)); }
Rows4 to_rows(const Transform& T) {
  Rows4 out{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = T.at(r, c);
  }
  return out;
}

Metric space_metric(std::shared_ptr<const JointSpace> space) {
  return [space](ConfigView a, ConfigView b) { return space->distance(a, b); };
}

std::vector<JointBoxObstacles::Box> to_boxes(const std::vector<std::pair<std::vector<double>, std::vector<double>>>& in) {
  std::vector<JointBoxObstacles::Box> out;
  out.reserve(in.size());
  for (const auto& [lo, hi] : in) out.push_back({lo, hi});
  return out;
}

// Trajectory data crosses as plain tuples (#214): a ramp is (x0, v0, [(duration, acceleration), ...]), a piece is
// (validated, [ramp, ...]). sscbirrt._trajectory.from_native rebuilds the Python objects.
using RampData = std::tuple<double, double, std::vector<std::pair<double, double>>>;
using PieceData = std::pair<std::string, std::vector<RampData>>;
using AttemptData = std::tuple<int, double, double, std::optional<double>, std::string, bool, double>;

RampData ramp_data(const trajectory::Ramp& r) {
  std::vector<std::pair<double, double>> segs;
  segs.reserve(r.segments.size());
  for (const auto& s : r.segments) segs.emplace_back(s.duration, s.acceleration);
  return {r.x0, r.v0, std::move(segs)};
}

trajectory::Piece piece_from_data(const PieceData& data) {
  trajectory::Piece p{{}, data.first};
  for (const auto& [x0, v0, segs] : data.second) {
    trajectory::Ramp r{x0, v0, {}};
    for (const auto& [d, a] : segs) r.segments.push_back({d, a});
    p.ramps.push_back(std::move(r));
  }
  return p;
}

PieceData piece_data(const trajectory::Piece& p) {
  std::vector<RampData> ramps;
  ramps.reserve(p.ramps.size());
  for (const auto& r : p.ramps) ramps.push_back(ramp_data(r));
  return {p.validated, std::move(ramps)};
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "sscbirrt: the native C++20 core of sscbirrt (see docs/native-design.md)";

  // ----- exceptions ---------------------------------------------------------------------------
  static py::exception<UnsupportedCapability> exc_unsupported(m, "UnsupportedCapability", PyExc_TypeError);
  static py::exception<ContractError> exc_contract(m, "ContractError", PyExc_ValueError);
  static py::exception<NoRoots> exc_noroots(m, "NoRoots", PyExc_RuntimeError);
  static py::exception<MalformedTrajectory> exc_malformed(m, "MalformedTrajectory", PyExc_ValueError);
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const NoRoots& e) {
      const py::object type = exc_noroots;  // the exception type; calling it makes an instance
      py::object inst = type(e.what());
      inst.attr("role") = e.role;
      inst.attr("report") = py::cast(e.report);
      PyErr_SetObject(exc_noroots.ptr(), inst.ptr());
    }
  });

  py::class_<RootReport>(m, "RootReport")
      .def_readonly("explicit_candidates", &RootReport::explicit_candidates)
      .def_readonly("explicit_rejected", &RootReport::explicit_rejected)
      .def_readonly("draws", &RootReport::draws)
      .def_readonly("draws_empty", &RootReport::draws_empty)
      .def_readonly("outside_space", &RootReport::outside_space)
      .def_readonly("in_collision", &RootReport::in_collision)
      .def_readonly("constraint_violated", &RootReport::constraint_violated)
      .def_readonly("roots", &RootReport::roots)
      .def_readonly("details", &RootReport::details)
      .def("rejections", &RootReport::rejections)
      .def("only_collisions", &RootReport::only_collisions)
      .def("candidates", &RootReport::candidates)
      .def("summary", &RootReport::summary);

  // ----- space --------------------------------------------------------------------------------
  py::class_<SpaceSampler, std::shared_ptr<SpaceSampler>>(m, "SpaceSampler");
  py::class_<JointSpace, SpaceSampler, std::shared_ptr<JointSpace>>(m, "JointSpace")
      .def(py::init<std::vector<double>, std::vector<double>, std::vector<bool>>(), py::arg("lower"), py::arg("upper"),
           py::arg("angular") = std::vector<bool>{})
      .def_property_readonly("dof", &JointSpace::dof)
      .def_property_readonly("lower", &JointSpace::lower)
      .def_property_readonly("upper", &JointSpace::upper)
      .def_property_readonly("angular", &JointSpace::angular)
      .def("contains", [](const JointSpace& s, const Config& q) { return s.contains(q); })
      .def("why_invalid", [](const JointSpace& s, const Config& q) { return s.why_invalid(q); })
      .def("distance", [](const JointSpace& s, const Config& a, const Config& b) { return s.distance(a, b); })
      .def("direction", [](const JointSpace& s, const Config& a, const Config& b) { return s.direction(a, b); })
      .def("unwrap_path", &JointSpace::unwrap_path);

  // ----- sets ---------------------------------------------------------------------------------
  py::class_<StateSet, std::shared_ptr<StateSet>>(m, "StateSet")
      .def("contains", [](const StateSet& s, const Config& q) { return s.contains(q); })
      .def("is_finite", &StateSet::is_finite)
      .def("describe", &StateSet::describe)
      .def("supports_sampler", [](const StateSet& s) { return s.sampler() != nullptr; })
      .def("supports_projector", [](const StateSet& s) { return s.projector() != nullptr; })
      .def("supports_distance", [](const StateSet& s) { return s.distancer() != nullptr; })
      .def("supports_violation", [](const StateSet& s) { return s.violator() != nullptr; });

  py::class_<FiniteSet, StateSet, std::shared_ptr<FiniteSet>>(m, "FiniteSet", py::multiple_inheritance())
      .def(py::init([](std::vector<Config> configs, double tolerance) {
             return std::make_shared<FiniteSet>(std::move(configs), tolerance, Metric(euclidean));
           }),
           py::arg("configs"), py::arg("tolerance") = 1e-6, "Members compared under the Euclidean metric.")
      .def(py::init([](std::vector<Config> configs, double tolerance, std::shared_ptr<const JointSpace> space) {
             return std::make_shared<FiniteSet>(std::move(configs), tolerance, space_metric(std::move(space)));
           }),
           py::arg("configs"), py::arg("tolerance"), py::arg("space"), "Members compared under the space's metric.")
      .def_property_readonly("configs", &FiniteSet::configs)
      .def_property_readonly("tolerance", &FiniteSet::tolerance);

  py::class_<EmptySet, StateSet, std::shared_ptr<EmptySet>>(m, "EmptySet").def(py::init<>());

  py::class_<IntersectionProjection, std::shared_ptr<IntersectionProjection>>(m, "IntersectionProjection");
  py::class_<IntersectionSampling, std::shared_ptr<IntersectionSampling>>(m, "IntersectionSampling");
  py::class_<MostViolatedProjection, IntersectionProjection, std::shared_ptr<MostViolatedProjection>>(m, "MostViolatedProjection")
      .def(py::init<int, double>(), py::arg("max_iters") = 50, py::arg("progress_tolerance") = 1e-6);
  py::class_<RejectionSampling, IntersectionSampling, std::shared_ptr<RejectionSampling>>(m, "RejectionSampling")
      .def(py::init<int>(), py::arg("source") = 0);

  py::class_<AnyOf, StateSet, std::shared_ptr<AnyOf>>(m, "AnyOf", py::multiple_inheritance())
      .def(py::init([](std::vector<SetPtr> children, std::optional<std::vector<double>> weights,
                       std::shared_ptr<const JointSpace> space) {
             return std::make_shared<AnyOf>(std::move(children), std::move(weights),
                                            space ? space_metric(std::move(space)) : Metric(euclidean));
           }),
           py::arg("children"), py::arg("weights") = std::nullopt, py::arg("space") = nullptr,
           "space, if given, supplies the metric used to pick the nearest successful child projection.");

  py::class_<AllOf, StateSet, std::shared_ptr<AllOf>>(m, "AllOf", py::multiple_inheritance())
      .def(py::init<std::vector<SetPtr>, std::shared_ptr<const IntersectionProjection>,
                    std::shared_ptr<const IntersectionSampling>>(),
           py::arg("children"), py::arg("projection") = nullptr, py::arg("sampling") = nullptr);

  // ----- validity and motion ------------------------------------------------------------------
  py::class_<Contact>(m, "Contact")
      .def_readonly("kind", &Contact::kind)
      .def_readonly("body1", &Contact::body1)
      .def_readonly("body2", &Contact::body2)
      .def_readonly("geom1", &Contact::geom1)
      .def_readonly("geom2", &Contact::geom2)
      .def_readonly("distance", &Contact::distance);
  py::class_<StateValidator, std::shared_ptr<StateValidator>>(m, "StateValidator")
      .def("is_valid", [](const StateValidator& v, const Config& q) { return v.is_valid(q); })
      .def("contacts", [](const StateValidator& v, const Config& q) { return v.contacts(q); });
  py::class_<AcceptAll, StateValidator, std::shared_ptr<AcceptAll>>(m, "AcceptAll").def(py::init<>());
  py::class_<JointBoxObstacles, StateValidator, std::shared_ptr<JointBoxObstacles>>(m, "JointBoxObstacles")
      .def(py::init([](const std::vector<std::pair<std::vector<double>, std::vector<double>>>& boxes) {
             return std::make_shared<JointBoxObstacles>(to_boxes(boxes));
           }),
           py::arg("boxes"), "Open axis-aligned boxes as (lo, hi) pairs; infinite bounds allowed.");
  py::class_<MotionValidator, std::shared_ptr<MotionValidator>>(m, "MotionValidator");

  // ----- validation (#230) ---------------------------------------------------------------------
  m.attr("__version__") = version();
  m.def("spread_order", &spread_order, py::arg("n"), "Van der Corput visiting order of n states (sscbirrt.validation).");
  py::class_<ValidationPolicy>(m, "ValidationPolicy")
      .def(py::init([](double resolution) {
             ValidationPolicy p{resolution};
             p.validate();
             return p;
           }),
           py::arg("resolution") = 0.05)
      .def_readonly("resolution", &ValidationPolicy::resolution)
      .def_property_readonly("parameters", &ValidationPolicy::parameters)
      .def_property_readonly("summary", &ValidationPolicy::summary);
  py::class_<InvalidWitness>(m, "InvalidWitness")
      .def_property_readonly("failure", [](const InvalidWitness& w) { return std::string(failure_name(w.failure)); })
      .def_readonly("configuration", &InvalidWitness::configuration)
      .def_readonly("segment", &InvalidWitness::segment)
      .def_readonly("fraction", &InvalidWitness::fraction)
      .def_readonly("time_s", &InvalidWitness::time_s)
      .def_readonly("contacts", &InvalidWitness::contacts)
      .def_readonly("constraint", &InvalidWitness::constraint)
      .def_readonly("detail", &InvalidWitness::detail);
  py::class_<ValidationReport>(m, "ValidationReport")
      .def_readonly("subject", &ValidationReport::subject)
      .def_readonly("valid", &ValidationReport::valid)
      .def_readonly("intervals", &ValidationReport::intervals)
      .def_readonly("checks", &ValidationReport::checks)
      .def_readonly("policy", &ValidationReport::policy)
      .def_readonly("implementation", &ValidationReport::implementation)
      .def_readonly("witness", &ValidationReport::witness)
      .def_readonly("duration_s", &ValidationReport::duration_s);
  py::class_<Validator>(m, "Validator")
      .def(py::init<std::shared_ptr<const JointSpace>, std::shared_ptr<const StateValidator>,
                    std::shared_ptr<const StateSet>, ValidationPolicy>(),
           py::arg("space"), py::arg("validator"), py::arg("path_constraint") = nullptr,
           py::arg("policy") = ValidationPolicy{},
           "Not thread-safe: the validator's components are used without locks (one Validator per thread).")
      .def("validate_state", [](const Validator& v, const Config& q) { return v.validate_state(q); },
           py::call_guard<py::gil_scoped_release>())
      .def("validate_motion", [](const Validator& v, const Config& a, const Config& b) { return v.validate_motion(a, b); },
           py::call_guard<py::gil_scoped_release>())
      .def("validate_path", &Validator::validate_path, py::call_guard<py::gil_scoped_release>())
      .def(
          "validate_trajectory",
          [](const Validator& v, const std::vector<PieceData>& pieces, const std::vector<double>& vmax,
             const std::vector<double>& amax, std::optional<std::vector<Config>> path) {
            trajectory::Trajectory traj{{}, vmax, amax, path.value_or(std::vector<Config>{})};
            for (const auto& p : pieces) traj.pieces.push_back(piece_from_data(p));
            return v.validate_trajectory(traj);
          },
          py::arg("pieces"), py::arg("vmax"), py::arg("amax"), py::arg("path") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(), "Pieces as smooth_trajectory returns them; path: the source path.")
      .def_property_readonly("policy", &Validator::policy);

  // ----- problem, config, cancellation --------------------------------------------------------
  py::class_<PlanningProblem>(m, "PlanningProblem")
      .def(py::init<>())
      .def_readwrite("space", &PlanningProblem::space)
      .def_readwrite("start", &PlanningProblem::start)
      .def_readwrite("goal", &PlanningProblem::goal)
      .def_readwrite("validator", &PlanningProblem::validator)
      .def_readwrite("path_constraint", &PlanningProblem::path_constraint)
      .def_readwrite("motion_validator", &PlanningProblem::motion_validator)
      .def_readwrite("sampler", &PlanningProblem::sampler);

  py::class_<PlannerConfig>(m, "PlannerConfig")
      .def(py::init<>())
      .def_readwrite("timeout_seconds", &PlannerConfig::timeout_seconds)
      .def_readwrite("max_iterations", &PlannerConfig::max_iterations)
      .def_readwrite("connection_tolerance", &PlannerConfig::connection_tolerance)
      .def_readwrite("edge_resolution", &PlannerConfig::edge_resolution)
      .def_readwrite("progress_tolerance", &PlannerConfig::progress_tolerance)
      .def_readwrite("step_size", &PlannerConfig::step_size)
      .def_readwrite("start_sample_probability", &PlannerConfig::start_sample_probability)
      .def_readwrite("goal_sample_probability", &PlannerConfig::goal_sample_probability)
      .def_readwrite("extend_steps", &PlannerConfig::extend_steps)
      .def_readwrite("connect_steps", &PlannerConfig::connect_steps)
      .def_readwrite("sample_draws", &PlannerConfig::sample_draws)
      .def_readwrite("num_tree_roots", &PlannerConfig::num_tree_roots)
      .def_readwrite("max_per_draw", &PlannerConfig::max_per_draw)
      .def_readwrite("smooth_path", &PlannerConfig::smooth_path)
      .def_readwrite("smoothing_iterations", &PlannerConfig::smoothing_iterations)
      .def_readwrite("smoothing_patience", &PlannerConfig::smoothing_patience)
      .def_readwrite("shorten_endpoints", &PlannerConfig::shorten_endpoints)
      .def("validate", &PlannerConfig::validate);

  py::class_<CancellationToken, std::shared_ptr<CancellationToken>>(m, "CancellationToken")
      .def(py::init<>())
      .def("cancel", &CancellationToken::cancel)
      .def("cancelled", &CancellationToken::cancelled);

  // ----- result -------------------------------------------------------------------------------
  py::class_<Tree, std::shared_ptr<Tree>>(m, "Tree")
      .def("size", &Tree::size)
      .def("nodes", [](const Tree& t) {
        std::vector<std::tuple<Config, int, Provenance>> out;
        out.reserve(t.nodes().size());
        for (const Node& n : t.nodes()) out.emplace_back(n.q, n.parent, n.source);
        return out;
      });

  py::class_<SolveStats>(m, "SolveStats")
      .def_readonly("state_checks", &SolveStats::state_checks)
      .def_readonly("edge_checks", &SolveStats::edge_checks)
      .def_readonly("set_samples", &SolveStats::set_samples)
      .def_readonly("set_projections", &SolveStats::set_projections)
      .def_readonly("search_roots", &SolveStats::search_roots)
      .def_readonly("reused_verdicts", &SolveStats::reused_verdicts)
      .def_readonly("seconds_state_checks", &SolveStats::seconds_state_checks)
      .def_readonly("seconds_edge_checks", &SolveStats::seconds_edge_checks)
      .def_readonly("seconds_set_samples", &SolveStats::seconds_set_samples)
      .def_readonly("seconds_set_projections", &SolveStats::seconds_set_projections)
      .def_readonly("seconds_roots", &SolveStats::seconds_roots)
      .def_readonly("seconds_search", &SolveStats::seconds_search)
      .def_readonly("seconds_smoothing", &SolveStats::seconds_smoothing);

  py::class_<PlanResult>(m, "PlanResult")
      .def_property_readonly("status", [](const PlanResult& r) { return std::string(status_name(r.status)); })
      .def_property_readonly("success", &PlanResult::success)
      .def_readonly("reason", &PlanResult::reason)
      .def_readonly("path", &PlanResult::path)
      .def_readonly("start_source", &PlanResult::start_source)
      .def_readonly("goal_source", &PlanResult::goal_source)
      .def_readonly("iterations", &PlanResult::iterations)
      .def_readonly("planning_seconds", &PlanResult::planning_seconds)
      .def_readonly("tree_sizes", &PlanResult::tree_sizes)
      .def_readonly("start_roots", &PlanResult::start_roots)
      .def_readonly("goal_roots", &PlanResult::goal_roots)
      .def_readonly("stats", &PlanResult::stats)
      .def_readonly("tree_start", &PlanResult::tree_start)
      .def_readonly("tree_goal", &PlanResult::tree_goal)
      .def_readonly("validation", &PlanResult::validation);

  // ----- planner ------------------------------------------------------------------------------
  py::class_<Planner>(m, "Planner")
      .def(py::init<PlannerConfig>(), py::arg("config") = PlannerConfig{})
      .def_property_readonly("config", &Planner::config)
      .def(
          "solve",
          [](const Planner& planner, const PlanningProblem& problem, std::optional<std::uint64_t> seed,
             std::shared_ptr<const CancellationToken> cancel, bool keep_trees) {
            SolveOptions o;
            o.seed = seed;
            o.cancel = std::move(cancel);
            o.keep_trees = keep_trees;
            return planner.solve(problem, o);
          },
          py::arg("problem"), py::arg("seed") = std::nullopt, py::arg("cancel") = nullptr, py::arg("keep_trees") = true,
          py::call_guard<py::gil_scoped_release>(),
          "Solve with the GIL released. No Python object is touched after entry.");

  m.def("why_inadmissible", [](const PlanningProblem& p, const Config& q) { return Planner::why_inadmissible(p, q); });

  // ----- trajectory smoothing (#214, not yet public API) ---------------------------------------
  m.def("ramp_min_time", [](double x0, double v0, double x1, double v1, double vmax, double amax) {
    auto r = trajectory::min_time(x0, v0, x1, v1, vmax, amax);
    return r ? std::optional<RampData>(ramp_data(*r)) : std::nullopt;
  });
  m.def("ramp_fixed_time", [](double x0, double v0, double x1, double v1, double T, double vmax, double amax) {
    auto r = trajectory::fixed_time(x0, v0, x1, v1, T, vmax, amax);
    return r ? std::optional<RampData>(ramp_data(*r)) : std::nullopt;
  });
  m.def("synchronize", [](const Config& q0, const Config& v0, const Config& q1, const Config& v1,
                          const std::vector<double>& vmax, const std::vector<double>& amax) {
    auto p = trajectory::synchronize(q0, v0, q1, v1, vmax, amax);
    return p ? std::optional<PieceData>(piece_data(*p)) : std::nullopt;
  });
  m.def("coalesce", &trajectory::coalesce, py::arg("path"), py::arg("tolerance") = 1e-9);
  m.def(
      "smooth_trajectory",
      [](const Planner& planner, const PlanningProblem& problem, const std::vector<Config>& path,
         const std::vector<double>& vmax, const std::vector<double>& amax, int iterations, std::uint64_t seed,
         std::optional<int> convergence_window, double convergence_tolerance, std::optional<int> patience,
         std::optional<double> time_budget, double min_interval, std::shared_ptr<const CancellationToken> cancel) {
        trajectory::ShortcutOptions o;
        o.cancel = std::move(cancel);
        o.iterations = iterations;
        o.seed = seed;
        o.convergence_window = convergence_window;
        o.convergence_tolerance = convergence_tolerance;
        o.patience = patience;
        o.time_budget = time_budget;
        o.min_interval = min_interval;
        trajectory::ShortcutReport rep;
        const trajectory::Trajectory traj = trajectory::smooth_path_in_time(planner, problem, path, vmax, amax, o, &rep);
        std::vector<PieceData> pieces;
        pieces.reserve(traj.pieces.size());
        for (const auto& p : traj.pieces) pieces.push_back(piece_data(p));
        std::vector<AttemptData> attempts;
        attempts.reserve(rep.attempts.size());
        for (const auto& a : rep.attempts) attempts.emplace_back(a.iteration, a.t1, a.t2, a.gain, a.rejected, a.accepted, a.elapsed);
        return std::make_tuple(std::move(pieces), std::move(attempts), rep.accepted, rep.seconds, rep.stopped);
      },
      py::arg("planner"), py::arg("problem"), py::arg("path"), py::arg("vmax"), py::arg("amax"), py::arg("iterations"),
      py::arg("seed"), py::arg("convergence_window"), py::arg("convergence_tolerance"), py::arg("patience"),
      py::arg("time_budget"), py::arg("min_interval"), py::arg("cancel") = nullptr,
      py::call_guard<py::gil_scoped_release>(),
      "Time-optimized shortcuts on a planned path, with the GIL released: "
      "(pieces, attempts, accepted, seconds, stopped).");
  m.def("shortcut_defaults", [] {
    const trajectory::ShortcutOptions o;
    return py::dict(py::arg("iterations") = o.iterations, py::arg("convergence_window") = o.convergence_window,
                    py::arg("convergence_tolerance") = o.convergence_tolerance, py::arg("patience") = o.patience,
                    py::arg("time_budget") = o.time_budget, py::arg("seed") = o.seed,
                    py::arg("min_interval") = o.min_interval);
  }, "The native ShortcutOptions defaults, for the test that holds them equal to Python's.");

  // ----- pose regions (v1.6.0) ------------------------------------------------------------------
  py::class_<tsr::TSR>(m, "TSR", "sstsr's own C++ TSR (sstsr >= 3.3); frames as 4x4 nested lists, Bw as 6x2.")
      .def(py::init([](const Rows4& T0_w, const Rows4& Tw_e, const Rows6x2& Bw) {
             return tsr::TSR(pose_rows(T0_w), pose_rows(Tw_e), tsr::Bounds6{Bw});
           }),
           py::arg("T0_w"), py::arg("Tw_e"), py::arg("Bw"))
      .def("contains", [](const tsr::TSR& t, const Rows4& T) { return t.contains(pose_rows(T)); })
      .def("distance", [](const tsr::TSR& t, const Rows4& T) { return t.distance(pose_rows(T)); })
      .def("distance_bwopt", [](const tsr::TSR& t, const Rows4& T) { return t.distance_bwopt(pose_rows(T)); })
      .def("closest_transform",
           [](const tsr::TSR& t, const Rows4& T) {
             const auto [d, C] = t.closest_transform(pose_rows(T));
             return std::make_pair(d, to_rows(tsr::from_sstsr(C)));
           })
      .def("to_xyzrpy", [](const tsr::TSR& t, const Rows4& T) { return t.to_xyzrpy(pose_rows(T)); })
      .def(
          "sample",
          [](const tsr::TSR& t, std::uint64_t seed, int count) {
            Rng rng(seed);
            std::vector<Rows4> out;
            for (int i = 0; i < count; ++i) out.push_back(to_rows(tsr::from_sstsr(t.sample(rng))));
            return out;
          },
          py::arg("seed"), py::arg("count") = 1, "Draw `count` poses from one RNG seeded with `seed`.")
      .def("sample_xyzrpy", [](const tsr::TSR& t, std::uint64_t seed) { Rng rng(seed); return t.sample_xyzrpy(rng); })
      .def("volume", &tsr::TSR::volume)
      .def("continuous_bounds", [](const tsr::TSR& t) { return t.continuous_bounds().rows; });
  py::class_<tsr::TSRChain>(m, "TSRChain",
                            "sstsr's own C++ TSRChain (sstsr >= 3.3): TSRs composed in series. A later link's T0_w must be "
                            "the identity; its offset belongs in the previous link's Tw_e.")
      .def(py::init([](const std::vector<tsr::TSR>& links) { return tsr::TSRChain(links); }), py::arg("tsrs"))
      .def("__len__", &tsr::TSRChain::size)
      .def("distance", [](const tsr::TSRChain& c, const Rows4& T) { return c.distance(pose_rows(T)).first; },
           "The residual of a bounded inverse solve: an upper bound, 0 when a witness is found.")
      .def("closest_transform",
           [](const tsr::TSRChain& c, const Rows4& T) {
             const auto [d, C] = c.closest_transform(pose_rows(T));
             return std::make_pair(d, to_rows(tsr::from_sstsr(C)));
           })
      .def(
          "sample",
          [](const tsr::TSRChain& c, std::uint64_t seed, int count) {
            Rng rng(seed);
            std::vector<Rows4> out;
            for (int i = 0; i < count; ++i) out.push_back(to_rows(tsr::from_sstsr(c.sample(rng))));
            return out;
          },
          py::arg("seed"), py::arg("count") = 1, "Draw `count` poses from one RNG seeded with `seed`.");
  py::class_<ForwardKinematics, std::shared_ptr<ForwardKinematics>>(m, "ForwardKinematics")
      .def_property_readonly("dof", &ForwardKinematics::dof)
      .def("fk", [](const ForwardKinematics& f, const Config& q) { return to_rows(f.fk(q)); });
  py::class_<IKSolver, std::shared_ptr<IKSolver>>(m, "IKSolver")
      .def_property_readonly("dof", &IKSolver::dof)
      .def("solve", [](const IKSolver& s, const Rows4& T, std::optional<Config> seed) {
             return s.solve(from_rows(T), seed ? ConfigView(*seed) : ConfigView{});
           },
           py::arg("pose"), py::arg("seed") = std::nullopt);

  py::class_<tsr::TSRConfigurationSet, StateSet, std::shared_ptr<tsr::TSRConfigurationSet>>(m, "TSRConfigurationSet",
                                                                                            py::multiple_inheritance())
      .def(py::init([](const tsr::TSRChain& region, std::shared_ptr<const ForwardKinematics> fk,
                       std::shared_ptr<const IKSolver> ik, std::shared_ptr<const JointSpace> space, double tolerance,
                       int max_projection_iters, double progress_tolerance) {
             return std::make_shared<tsr::TSRConfigurationSet>(tsr::Region(region), std::move(fk), std::move(ik), std::move(space),
                                                                tolerance, max_projection_iters, progress_tolerance);
           }),
           py::arg("region"), py::arg("fk"), py::arg("ik"), py::arg("space"), py::arg("tolerance") = 1e-3,
           py::arg("max_projection_iters") = 50, py::arg("progress_tolerance") = 1e-6)
      .def(py::init([](const tsr::TSR& region, std::shared_ptr<const ForwardKinematics> fk, std::shared_ptr<const IKSolver> ik,
                       std::shared_ptr<const JointSpace> space, double tolerance, int max_projection_iters,
                       double progress_tolerance) {
             return std::make_shared<tsr::TSRConfigurationSet>(tsr::Region(region), std::move(fk), std::move(ik), std::move(space), tolerance,
                                                                max_projection_iters, progress_tolerance);
           }),
           py::arg("region"), py::arg("fk"), py::arg("ik"), py::arg("space"), py::arg("tolerance") = 1e-3,
           py::arg("max_projection_iters") = 50, py::arg("progress_tolerance") = 1e-6)
      .def("distance", [](const tsr::TSRConfigurationSet& s, const Config& q) { return s.distance(q); })
      .def("project", [](const tsr::TSRConfigurationSet& s, const Config& q) { return s.project(q, q); })
      .def("sample", [](const tsr::TSRConfigurationSet& s, std::uint64_t seed) {
        Rng rng(seed);
        std::vector<Config> out;
        for (Sample& smp : s.sample(rng)) out.push_back(std::move(smp.q));
        return out;
      });
  m.def("tsr_weights", [](const std::vector<std::shared_ptr<const tsr::TSRConfigurationSet>>& sets) { return tsr::tsr_weights(sets); });

#if SSCBIRRT_HAS_SSIK
  m.def("has_ssik", [] { return true; });
  m.def("ssik_unavailable_reason", [] { return std::string(); });
  py::class_<sscbirrt::ssik::SSIKArm, ForwardKinematics, IKSolver, std::shared_ptr<sscbirrt::ssik::SSIKArm>>(m, "SSIKArm", py::multiple_inheritance())
      .def(py::init([](py::handle data, const Rows4& T_base, const Rows4& T_ee) {
             sscbirrt::ssik::ArmSpec spec{::ssik::pybind11_interop::solver_data_from_python(data), from_rows(T_base), from_rows(T_ee)};
             return std::make_shared<sscbirrt::ssik::SSIKArm>(std::move(spec));
           }),
           py::arg("data"), py::arg("T_base"), py::arg("T_ee"),
           "An arm from ssik.cpp.solver_data(manipulator) (ssik >= 8.2), plus the SSIKSolver frame offsets.")
      .def_property_readonly("family", [](const sscbirrt::ssik::SSIKArm& a) { return a.family(); });
#else
  m.def("has_ssik", [] { return false; });
  m.def("ssik_unavailable_reason", [] {
    return std::string("sscbirrt._native was built without SSIK support (ssik_cpp or Eigen3 not found at build time)");
  });
#endif

  m.def("unit_draws", [](std::uint64_t seed, int n) { Rng rng(seed); std::vector<double> out; for (int i = 0; i < n; ++i) out.push_back(unit(rng)); return out; });
}
