// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// sscbirrt._native_mujoco: the owned MuJoCo scene. A separate module from _native so that importing
// sscbirrt never loads MuJoCo; this module imports _native first so the shared base types exist.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "sscbirrt/mujoco/scene.hpp"
#include "sscbirrt/mujoco/snapshot.hpp"
#include "sscbirrt/mujoco/validator.hpp"

namespace py = pybind11;
using namespace sscbirrt;
using sscbirrt::mujoco::Attachment;
using sscbirrt::mujoco::Scene;
using sscbirrt::mujoco::SceneProvenance;
using sscbirrt::mujoco::Snapshot;
using sscbirrt::mujoco::InvalidContact;
using sscbirrt::mujoco::SceneValidator;

namespace {
using Rows4 = std::array<std::array<double, 4>, 4>;
Rows4 to_rows(const Transform& T) {
  Rows4 out{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) out[static_cast<std::size_t>(r)][static_cast<std::size_t>(c)] = T.at(r, c);
  }
  return out;
}
}  // namespace

PYBIND11_MODULE(_native_mujoco, m) {
  py::module_::import("sscbirrt._native");
  m.doc() = "sscbirrt::mujoco: an owned MuJoCo scene and immutable snapshots (docs/native-design.md, v1.7.0)";

  m.def("compiled_mujoco_version", &mujoco::compiled_mujoco_version_string, "The MuJoCo this module was built against, e.g. '3.14.0'.");
  m.def("loaded_mujoco_version", [] {
    const int v = mujoco::loaded_mujoco_version();
    return std::to_string(v / 1000000) + "." + std::to_string((v / 1000) % 1000) + "." + std::to_string(v % 1000);
  });
  m.attr("MUJOCO_LIBRARY") = std::string(SSCBIRRT_MUJOCO_LIBNAME);

  py::class_<SceneProvenance>(m, "SceneProvenance")
      .def_readonly("mujoco_version", &SceneProvenance::mujoco_version)
      .def_readonly("model_signature", &SceneProvenance::model_signature)
      .def_readonly("mjb_sha256", &SceneProvenance::mjb_sha256);

  py::class_<Scene, std::shared_ptr<Scene>>(m, "Scene")
      .def(py::init([](const py::bytes& mjb, std::vector<std::string> joints, std::vector<std::string> extra_arm_bodies,
                       std::uint64_t source_signature) {
             const std::string_view sv = mjb;
             return std::make_shared<Scene>(std::span<const std::byte>(reinterpret_cast<const std::byte*>(sv.data()), sv.size()),
                                            std::move(joints), std::move(extra_arm_bodies), source_signature);
           }),
           py::arg("mjb"), py::arg("controlled_joints"), py::arg("extra_arm_bodies") = std::vector<std::string>{},
           py::arg("source_signature") = 0)
      .def_property_readonly("dof", &Scene::dof)
      .def_property_readonly("qpos_addresses", &Scene::qpos_addresses)
      .def_property_readonly("joint_ids", &Scene::joint_ids)
      .def_property_readonly("lower", &Scene::lower)
      .def_property_readonly("upper", &Scene::upper)
      .def_property_readonly("arm_bodies", &Scene::arm_bodies)
      .def_property_readonly("nq", &Scene::nq)
      .def_property_readonly("nbody", &Scene::nbody)
      .def_property_readonly("nmocap", &Scene::nmocap)
      .def_property_readonly("ngeom", &Scene::ngeom)
      .def_property_readonly("provenance", &Scene::provenance)
      .def("body_id", &Scene::body_id)
      .def("site_id", &Scene::site_id)
      .def("geom_id", &Scene::geom_id)
      .def("body_name", &Scene::body_name)
      .def("subtree", &Scene::subtree)
      .def("body_has_free_joint", &Scene::body_has_free_joint)
      .def("is_arm_body", &Scene::is_arm_body);

  py::class_<Attachment>(m, "Attachment")
      .def(py::init([](int object_body, int gripper_body, const Rows4& T, std::vector<int> allowed) {
             Attachment a;
             a.object_body = object_body;
             a.gripper_body = gripper_body;
             a.T_gripper_object = Transform::from_rows(T);
             a.allowed_bodies = std::move(allowed);
             return a;
           }),
           py::arg("object_body"), py::arg("gripper_body"), py::arg("T_gripper_object"), py::arg("allowed_bodies"))
      .def_readonly("object_body", &Attachment::object_body)
      .def_readonly("gripper_body", &Attachment::gripper_body)
      .def_property_readonly("T_gripper_object", [](const Attachment& a) { return to_rows(a.T_gripper_object); })
      .def_readonly("allowed_bodies", &Attachment::allowed_bodies);

  py::class_<Snapshot>(m, "Snapshot")
      .def(py::init([](const Scene& scene, std::vector<double> qpos, std::vector<double> mocap_pos, std::vector<double> mocap_quat,
                       std::vector<Attachment> attachments) {
             Snapshot s;
             s.qpos = std::move(qpos);
             s.mocap_pos = std::move(mocap_pos);
             s.mocap_quat = std::move(mocap_quat);
             s.attachments = std::move(attachments);
             s.validate(scene);  // throws std::invalid_argument -> ValueError
             return s;
           }),
           py::arg("scene"), py::arg("qpos"), py::arg("mocap_pos"), py::arg("mocap_quat"), py::arg("attachments"),
           "A validated, immutable snapshot; construction copies every array.")
      .def_readonly("qpos", &Snapshot::qpos)
      .def_readonly("mocap_pos", &Snapshot::mocap_pos)
      .def_readonly("mocap_quat", &Snapshot::mocap_quat)
      .def_readonly("attachments", &Snapshot::attachments)
      .def_readonly("sha256", &Snapshot::sha256);

  py::class_<InvalidContact>(m, "InvalidContact")
      .def_readonly("body1", &InvalidContact::body1)
      .def_readonly("body2", &InvalidContact::body2)
      .def_readonly("geom1", &InvalidContact::geom1)
      .def_readonly("geom2", &InvalidContact::geom2)
      .def_readonly("dist", &InvalidContact::dist)
      .def_property_readonly("kind", [](const InvalidContact& c) {
        return std::string(c.kind == InvalidContact::Kind::SelfCollision ? "self_collision" : "robot_environment");
      });

  // Derives from the StateValidator registered by sscbirrt._native, so a native PlanningProblem accepts it.
  py::class_<SceneValidator, StateValidator, std::shared_ptr<SceneValidator>>(m, "SceneValidator")
      .def(py::init([](std::shared_ptr<const Scene> scene, const Snapshot& snapshot) { return std::make_shared<SceneValidator>(std::move(scene), snapshot); }),
           py::arg("scene"), py::arg("snapshot"), "One private mjData per validator; use one validator per solve.")
      .def("is_valid", [](const SceneValidator& v, const Config& q) { return v.is_valid(q); })
      .def("full_turn_invariant", &SceneValidator::full_turn_invariant)
      .def("invalid_contacts", [](const SceneValidator& v, const Config& q) { return v.invalid_contacts(q); })
      .def_property_readonly("snapshot", &SceneValidator::snapshot);
}
