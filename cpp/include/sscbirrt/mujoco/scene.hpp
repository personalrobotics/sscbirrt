// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "sscbirrt/transform.hpp"
#include "sscbirrt/types.hpp"

struct mjModel_;  // MuJoCo's model; this header does not include MuJoCo so consumers of the type need not

namespace sscbirrt::mujoco {

struct SceneProvenance {
  std::string mujoco_version;     // mj_versionString() of the loaded library
  std::uint64_t model_signature;  // the compiled model's signature as the caller passed it (MJB does not carry it)
  std::string mjb_sha256;         // of the bytes the scene was loaded from: the scene's identity
};

// The MuJoCo version this library was compiled against (mjVERSION_HEADER, e.g. 3014000 for 3.14.0) and the
// one it loaded at runtime (mj_version()). They must agree; Scene refuses to construct otherwise.
int compiled_mujoco_version();
int loaded_mujoco_version();
std::string compiled_mujoco_version_string();  // "3.14.0"

// An mjModel the planner owns, loaded from the MJB bytes of a compiled model. Read-only after construction
// and shared by every validator and solve; the model that produced the bytes may be destroyed or changed.
class Scene {
 public:
  Scene(std::span<const std::byte> mjb, std::vector<std::string> controlled_joints,
        std::vector<std::string> extra_arm_bodies = {},
        std::uint64_t source_signature = 0);  // throws std::invalid_argument, see docs/native-design.md
  ~Scene();
  Scene(const Scene&) = delete;
  Scene& operator=(const Scene&) = delete;

  int dof() const { return static_cast<int>(qpos_adr_.size()); }
  const std::vector<int>& qpos_addresses() const { return qpos_adr_; }
  const std::vector<int>& joint_ids() const { return joint_ids_; }
  const std::vector<double>& lower() const { return lower_; }  // ±infinity where the joint is unlimited (#107)
  const std::vector<double>& upper() const { return upper_; }
  const std::vector<int>& arm_bodies() const { return arm_bodies_; }  // sorted, unique
  bool is_arm_body(int body) const;

  int body_id(const std::string& name) const;   // -1 if absent
  int site_id(const std::string& name) const;
  int geom_id(const std::string& name) const;
  std::string body_name(int body) const;
  std::vector<int> subtree(int body) const;     // body and every descendant, sorted
  bool body_has_free_joint(int body) const;     // its first joint is a free joint

  int nq() const;
  int nbody() const;
  int nmocap() const;
  int ngeom() const;
  const SceneProvenance& provenance() const { return provenance_; }
  std::string current_mjb_sha256() const;  // of the model as it is now: equals provenance().mjb_sha256 unless written
  const ::mjModel_* model() const { return model_; }

 private:
  ::mjModel_* model_ = nullptr;
  std::vector<int> joint_ids_, qpos_adr_, arm_bodies_;
  std::vector<double> lower_, upper_;
  SceneProvenance provenance_;
};

}  // namespace sscbirrt::mujoco
