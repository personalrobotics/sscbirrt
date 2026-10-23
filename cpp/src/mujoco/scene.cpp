// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// The only translation units that include MuJoCo are under src/mujoco/.
#include "sscbirrt/mujoco/scene.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "sscbirrt/detail/sha256.hpp"

namespace sscbirrt::mujoco {

namespace {

// MuJoCo reports fatal errors through a global handler that by default prints and exits. When nothing else
// has installed one (a standalone C++ consumer), install one that throws so a bad buffer is an exception.
// The Python bindings install their own throwing handler; that one is left alone.
[[noreturn]] void throwing_error_handler(const char* msg) { throw std::invalid_argument(std::string("MuJoCo: ") + (msg ? msg : "error")); }

void ensure_error_handler() {
  if (mju_user_error == nullptr) mju_user_error = throwing_error_handler;
}

std::string version_string(int v) {
  return std::to_string(v / 1000000) + "." + std::to_string((v / 1000) % 1000) + "." + std::to_string(v % 1000);
}

}  // namespace

int compiled_mujoco_version() { return mjVERSION_HEADER; }
int loaded_mujoco_version() { return mj_version(); }
std::string compiled_mujoco_version_string() { return version_string(mjVERSION_HEADER); }

Scene::Scene(std::span<const std::byte> mjb, std::vector<std::string> controlled_joints, std::vector<std::string> extra_arm_bodies,
             std::uint64_t source_signature) {
  if (compiled_mujoco_version() != loaded_mujoco_version()) {
    throw std::invalid_argument("MuJoCo version mismatch: sscbirrt::mujoco was compiled against " + compiled_mujoco_version_string() +
                                " but loaded " + version_string(loaded_mujoco_version()));
  }
  if (mjb.empty()) throw std::invalid_argument("MJB buffer is empty");
  if (controlled_joints.empty()) throw std::invalid_argument("at least one controlled joint is required");

  ensure_error_handler();
  try {
    model_ = mj_loadModelBuffer(mjb.data(), static_cast<int>(mjb.size()));
  } catch (...) {
    model_ = nullptr;
  }
  if (model_ == nullptr) {
    throw std::invalid_argument("MJB buffer could not be loaded: not an MJB of MuJoCo " + compiled_mujoco_version_string() +
                                ", or truncated");
  }

  try {
    std::set<std::string> seen;
    for (const std::string& name : controlled_joints) {
      if (!seen.insert(name).second) throw std::invalid_argument("duplicate controlled joint '" + name + "'");
      const int id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
      if (id < 0) throw std::invalid_argument("joint '" + name + "' not found in model");
      const int type = model_->jnt_type[id];
      if (type != mjJNT_HINGE && type != mjJNT_SLIDE) {
        throw std::invalid_argument("controlled joint '" + name + "' is a " + (type == mjJNT_FREE ? "free" : "ball") +
                                    " joint; native planning controls hinge and slide joints");
      }
      joint_ids_.push_back(id);
      qpos_adr_.push_back(model_->jnt_qposadr[id]);
      if (model_->jnt_limited[id]) {
        lower_.push_back(model_->jnt_range[2 * id]);
        upper_.push_back(model_->jnt_range[2 * id + 1]);
      } else {
        lower_.push_back(-std::numeric_limits<double>::infinity());
        upper_.push_back(std::numeric_limits<double>::infinity());
      }
    }

    std::set<int> arm;
    for (int id : joint_ids_) {
      for (int b : subtree(model_->jnt_bodyid[id])) arm.insert(b);
    }
    for (const std::string& name : extra_arm_bodies) {
      const int b = body_id(name);
      if (b < 0) throw std::invalid_argument("extra arm body '" + name + "' not found in model");
      for (int x : subtree(b)) arm.insert(x);
    }
    arm_bodies_.assign(arm.begin(), arm.end());

    provenance_.mujoco_version = mj_versionString();
    provenance_.model_signature = source_signature;  // mj_saveModel does not serialize the signature (it reads back 0)
    provenance_.mjb_sha256 = detail::Sha256::hex(mjb);
  } catch (...) {
    mj_deleteModel(model_);
    model_ = nullptr;
    throw;
  }
}

Scene::~Scene() {
  if (model_) mj_deleteModel(model_);
}

bool Scene::is_arm_body(int body) const { return std::binary_search(arm_bodies_.begin(), arm_bodies_.end(), body); }

int Scene::body_id(const std::string& name) const { return mj_name2id(model_, mjOBJ_BODY, name.c_str()); }
int Scene::site_id(const std::string& name) const { return mj_name2id(model_, mjOBJ_SITE, name.c_str()); }
int Scene::geom_id(const std::string& name) const { return mj_name2id(model_, mjOBJ_GEOM, name.c_str()); }

std::string Scene::body_name(int body) const {
  if (body < 0 || body >= model_->nbody) return "";
  const char* n = mj_id2name(model_, mjOBJ_BODY, body);
  return n ? std::string(n) : std::string();
}

std::vector<int> Scene::subtree(int body) const {
  std::vector<int> out;
  if (body < 0 || body >= model_->nbody) return out;
  std::vector<int> stack{body};
  while (!stack.empty()) {
    const int b = stack.back();
    stack.pop_back();
    out.push_back(b);
    for (int i = b + 1; i < model_->nbody; ++i) {  // children have larger ids than their parent in MuJoCo
      if (model_->body_parentid[i] == b) stack.push_back(i);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

bool Scene::body_has_free_joint(int body) const {
  if (body < 0 || body >= model_->nbody || model_->body_jntnum[body] < 1) return false;
  return model_->jnt_type[model_->body_jntadr[body]] == mjJNT_FREE;
}

int Scene::nq() const { return model_->nq; }
int Scene::nbody() const { return model_->nbody; }

std::string Scene::current_mjb_sha256() const {
  std::vector<std::byte> buf(static_cast<std::size_t>(mj_sizeModel(model_)));
  mj_saveModel(model_, nullptr, buf.data(), static_cast<int>(buf.size()));
  return detail::Sha256::hex(buf);
}
int Scene::nmocap() const { return model_->nmocap; }
int Scene::ngeom() const { return model_->ngeom; }

}  // namespace sscbirrt::mujoco
