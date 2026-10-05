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
#if SSCBIRRT_HAS_SSIK
#include "sscbirrt/ssik/ssik_arm.hpp"
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

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "sscbirrt: the native C++20 core of sscbirrt (see docs/native-design.md)";

  // ----- exceptions ---------------------------------------------------------------------------
  static py::exception<UnsupportedCapability> exc_unsupported(m, "UnsupportedCapability", PyExc_TypeError);
  static py::exception<ContractError> exc_contract(m, "ContractError", PyExc_ValueError);
  static py::exception<NoRoots> exc_noroots(m, "NoRoots", PyExc_RuntimeError);
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
  py::class_<StateValidator, std::shared_ptr<StateValidator>>(m, "StateValidator")
      .def("is_valid", [](const StateValidator& v, const Config& q) { return v.is_valid(q); });
  py::class_<AcceptAll, StateValidator, std::shared_ptr<AcceptAll>>(m, "AcceptAll").def(py::init<>());
  py::class_<JointBoxObstacles, StateValidator, std::shared_ptr<JointBoxObstacles>>(m, "JointBoxObstacles")
      .def(py::init([](const std::vector<std::pair<std::vector<double>, std::vector<double>>>& boxes) {
             return std::make_shared<JointBoxObstacles>(to_boxes(boxes));
           }),
           py::arg("boxes"), "Open axis-aligned boxes as (lo, hi) pairs; infinite bounds allowed.");
  py::class_<MotionValidator, std::shared_ptr<MotionValidator>>(m, "MotionValidator");

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
      .def_readonly("tree_goal", &PlanResult::tree_goal);

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
  py::class_<ssik::SSIKArm, ForwardKinematics, IKSolver, std::shared_ptr<ssik::SSIKArm>>(m, "SSIKArm", py::multiple_inheritance())
      .def(py::init([](const std::string& solver, const std::vector<std::array<double, 3>>& axis,
                       const std::vector<std::array<double, 16>>& t_left, const std::vector<std::array<double, 16>>& t_right,
                       const std::vector<int>& joint_type, const std::vector<double>& lo, const std::vector<double>& hi,
                       const std::vector<bool>& present, const Rows4& T_base, const Rows4& T_ee) {
             if (axis.size() != 6 || t_left.size() != 6 || t_right.size() != 6 || joint_type.size() != 6 || lo.size() != 6 ||
                 hi.size() != 6 || present.size() != 6) {
               throw std::invalid_argument("SSIKArm expects six joints");
             }
             ssik::ArmSpec spec;
             spec.solver_name = solver;
             for (std::size_t i = 0; i < 6; ++i) {
               spec.axis[i] = axis[i];
               spec.t_left[i] = t_left[i];
               spec.t_right[i] = t_right[i];
               spec.revolute[i] = joint_type[i] == 0;
               spec.lo[i] = lo[i];
               spec.hi[i] = hi[i];
               spec.present[i] = present[i];
             }
             spec.T_base = from_rows(T_base);
             spec.T_ee = from_rows(T_ee);
             return std::make_shared<ssik::SSIKArm>(std::move(spec));
           }),
           py::arg("solver"), py::arg("axis"), py::arg("t_left"), py::arg("t_right"), py::arg("joint_type"), py::arg("lo"),
           py::arg("hi"), py::arg("present"), py::arg("T_base"), py::arg("T_ee"),
           "Constants as ssik.cpp.joint_data renders them (t_left/t_right as flat row-major 16-vectors), plus the "
           "SSIKSolver frame offsets.")
      .def_property_readonly("family", [](const ssik::SSIKArm& a) { return std::string(ssik::family_name(a.family())); });
#else
  m.def("has_ssik", [] { return false; });
  m.def("ssik_unavailable_reason", [] {
    return std::string("sscbirrt._native was built without SSIK support (ssik_cpp or Eigen3 not found at build time)");
  });
#endif

  m.def("unit_draws", [](std::uint64_t seed, int n) { Rng rng(seed); std::vector<double> out; for (int i = 0; i < n; ++i) out.push_back(unit(rng)); return out; });
}
