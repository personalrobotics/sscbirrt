// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/mujoco/validator.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace sscbirrt::mujoco {

namespace {
std::string geom_label(const mjModel* m, int g) {
  const char* name = mj_id2name(m, mjOBJ_GEOM, g);
  return name ? "'" + std::string(name) + "'" : std::to_string(g);
}
}  // namespace

SceneValidator::SceneValidator(std::shared_ptr<const Scene> scene, Snapshot snapshot, CollisionOptions options)
    : scene_(std::move(scene)), snapshot_(std::move(snapshot)), data_(nullptr), clearance_(options.clearance) {
  if (!scene_) throw std::invalid_argument("SceneValidator needs a scene");
  if (!std::isfinite(clearance_) || clearance_ < 0.0) {
    throw std::invalid_argument("clearance must be a finite number of metres >= 0, got " + std::to_string(clearance_));
  }
  snapshot_.validate(*scene_);
  attachment_of_body_.assign(static_cast<std::size_t>(scene_->nbody()), -1);
  for (std::size_t i = 0; i < snapshot_.attachments.size(); ++i) {
    for (int b : scene_->subtree(snapshot_.attachments[i].object_body)) attachment_of_body_[static_cast<std::size_t>(b)] = static_cast<int>(i);
  }
  const mjModel* m = scene_->model();
  auto moving = [&](int b) { return b >= 0 && (scene_->is_arm_body(b) || attachment_of_body_[static_cast<std::size_t>(b)] >= 0); };
  if (clearance_ > 0.0) {
    if (m->opt.enableflags & mjENBL_OVERRIDE) {
      throw std::invalid_argument(
          "clearance > 0 needs the model's own contact margins, but the model enables MuJoCo's margin override "
          "(<option><flag override=\"enable\"/>), which replaces every margin with opt.o_margin");
    }
    for (int f = 0; f < m->nflex; ++f) {
      const int ct = m->flex_contype[f], ca = m->flex_conaffinity[f];
      if (!ct && !ca) continue;
      for (int g = 0; g < m->ngeom; ++g) {
        if (moving(m->geom_bodyid[g]) && ((ct & m->geom_conaffinity[g]) || (m->geom_contype[g] & ca))) {
          const char* name = mj_id2name(m, mjOBJ_FLEX, f);
          throw std::invalid_argument("clearance > 0 is not implemented for flex contacts: flex " +
                                      (name ? "'" + std::string(name) + "'" : std::to_string(f)) +
                                      " can collide with moving geom " + geom_label(m, g));
        }
      }
    }
  }
  if (!options.filter_passive_pairs && clearance_ == 0.0) {
    data_ = mj_makeData(m);
    if (!data_) throw std::runtime_error("mj_makeData failed");
    return;  // the full mj_collision on the scene's model as authored: the reference path
  }
  collision_view_ = std::make_unique<mjModel>(*m);  // the struct only; every array is the scene's, read-only
  if (options.filter_passive_pairs) {
    // mj_collision reads body_contype/body_conaffinity only to admit a body to the broadphase and to prefilter a
    // body pair; each geom pair is then filtered by its own geom masks. A body that could collide keeps contype ~0;
    // a moving one also keeps conaffinity ~0, a passive one gets 0: passive against passive fails the body-level
    // bitmask, anything against a moving body passes it and MuJoCo's geom-level filter decides as before.
    body_contype_.assign(m->body_contype, m->body_contype + m->nbody);
    body_conaffinity_.assign(m->body_conaffinity, m->body_conaffinity + m->nbody);
    for (int b = 0; b < m->nbody; ++b) {
      const auto i = static_cast<std::size_t>(b);
      if (!body_contype_[i] && !body_conaffinity_[i]) continue;  // collides with nothing, as before
      body_contype_[i] = ~0;
      body_conaffinity_[i] = moving(b) ? ~0 : 0;
    }
    collision_view_->body_contype = body_contype_.data();
    collision_view_->body_conaffinity = body_conaffinity_.data();
  }
  if (clearance_ > 0.0) {
    // In MuJoCo 3.14 margins are read only by the collision driver: a geom pair's margin is the sum of its geoms'
    // (or its pair_margin when predefined), the broadphase inflates each geom by margin + gap, and the BVH midphase
    // prunes by the sum of the bodies' body_margin, the maximum over a body's geoms of margin + gap.
    geom_margin_.assign(m->geom_margin, m->geom_margin + m->ngeom);
    body_margin_.assign(m->body_margin, m->body_margin + m->nbody);
    for (int g = 0; g < m->ngeom; ++g) {
      const int b = m->geom_bodyid[g];
      if (moving(b)) continue;  // the moving set keeps its authored margins
      auto& margin = geom_margin_[static_cast<std::size_t>(g)];
      margin = std::max(margin, clearance_);
      auto& bm = body_margin_[static_cast<std::size_t>(b)];
      bm = std::max(bm, margin + m->geom_gap[g]);
    }
    collision_view_->geom_margin = geom_margin_.data();
    collision_view_->body_margin = body_margin_.data();
    if (m->npair > 0) {
      pair_margin_.assign(m->pair_margin, m->pair_margin + m->npair);
      for (int p = 0; p < m->npair; ++p) {
        if (moving(m->geom_bodyid[m->pair_geom1[p]]) != moving(m->geom_bodyid[m->pair_geom2[p]])) {
          auto& pm = pair_margin_[static_cast<std::size_t>(p)];
          pm = std::max(pm, clearance_);
        }
      }
      collision_view_->pair_margin = pair_margin_.data();
    }
  }
  data_ = mj_makeData(m);
  if (!data_) throw std::runtime_error("mj_makeData failed");
}

SceneValidator::~SceneValidator() {
  if (data_) mj_deleteData(data_);
}

void SceneValidator::pose(ConfigView q) const {
  const mjModel* m = scene_->model();
  if (q.size() != static_cast<std::size_t>(scene_->dof())) {
    throw std::invalid_argument("SceneValidator: configuration has " + std::to_string(q.size()) + " values for " + std::to_string(scene_->dof()) + " controlled joints");
  }
  // 1. restore the snapshot
  std::memcpy(data_->qpos, snapshot_.qpos.data(), sizeof(double) * snapshot_.qpos.size());
  if (m->nmocap > 0) {
    std::memcpy(data_->mocap_pos, snapshot_.mocap_pos.data(), sizeof(double) * snapshot_.mocap_pos.size());
    std::memcpy(data_->mocap_quat, snapshot_.mocap_quat.data(), sizeof(double) * snapshot_.mocap_quat.size());
  }
  // 2. write the controlled joints
  const std::vector<int>& adr = scene_->qpos_addresses();
  for (std::size_t i = 0; i < adr.size(); ++i) data_->qpos[adr[i]] = q[i];
  // 3. kinematics: body and geom poses from qpos and mocap; no dynamics
  mj_kinematics(m, data_);
  // 4. attached free bodies ride on their gripper body
  for (const Attachment& a : snapshot_.attachments) {
    Transform Tg = Transform::identity();
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) Tg.at(r, c) = data_->xmat[9 * a.gripper_body + 3 * r + c];
      Tg.at(r, 3) = data_->xpos[3 * a.gripper_body + r];
    }
    const Transform To = Tg * a.T_gripper_object;
    const int jnt = m->body_jntadr[a.object_body];
    const int qadr = m->jnt_qposadr[jnt];
    mjtNum mat[9];
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) mat[3 * r + c] = To.at(r, c);
      data_->qpos[qadr + r] = To.at(r, 3);
    }
    mjtNum quat[4];
    mju_mat2Quat(quat, mat);
    for (int k = 0; k < 4; ++k) data_->qpos[qadr + 3 + k] = quat[k];
  }
  // 5. kinematics again only if attachments moved anything
  if (!snapshot_.attachments.empty()) mj_kinematics(m, data_);
  // 6. collision, of the pairs that involve the arm or a held object unless the validator is unfiltered
  mj_collision(collision_view_ ? collision_view_.get() : m, data_);
}

std::vector<InvalidContact> SceneValidator::invalid_contacts(ConfigView q) const {
  pose(q);
  const mjModel* m = scene_->model();
  std::vector<InvalidContact> out;
  for (int i = 0; i < data_->ncon; ++i) {
    const mjContact& c = data_->contact[i];
    const int g1 = c.geom[0], g2 = c.geom[1];
    const int b1 = g1 >= 0 ? m->geom_bodyid[g1] : -1;  // -1: a flex element; never part of the robot
    const int b2 = g2 >= 0 ? m->geom_bodyid[g2] : -1;
    const bool arm1 = b1 >= 0 && scene_->is_arm_body(b1), arm2 = b2 >= 0 && scene_->is_arm_body(b2);
    const int att1 = b1 >= 0 ? attachment_of_body_[static_cast<std::size_t>(b1)] : -1;
    const int att2 = b2 >= 0 ? attachment_of_body_[static_cast<std::size_t>(b2)] : -1;
    const bool robot1 = arm1 || att1 >= 0, robot2 = arm2 || att2 >= 0;
    if (!robot1 && !robot2) continue;  // environment against environment: ignored
    // 7. a held object touching a body its attachment allows (the grasp, a support, a container) is not a collision:
    // exactly one side is held, and the other side is that exact body, arm or environment. Two held objects are
    // always checked.
    auto allowed = [&](int att, int other) {
      const std::vector<int>& ok = snapshot_.attachments[static_cast<std::size_t>(att)].allowed_bodies;
      return std::binary_search(ok.begin(), ok.end(), other);
    };
    if ((att1 >= 0 && att2 < 0 && allowed(att1, b2)) || (att2 >= 0 && att1 < 0 && allowed(att2, b1))) continue;
    out.push_back({b1, b2, g1, g2, c.dist,
                   robot1 && robot2 ? InvalidContact::Kind::SelfCollision : InvalidContact::Kind::RobotEnvironment});
  }
  return out;
}

bool SceneValidator::is_valid(ConfigView q) const { return invalid_contacts(q).empty(); }

std::vector<int> SceneValidator::moving_bodies() const {
  std::vector<int> out;
  for (int b = 0; b < scene_->nbody(); ++b) {
    if (scene_->is_arm_body(b) || attachment_of_body_[static_cast<std::size_t>(b)] >= 0) out.push_back(b);
  }
  return out;
}

std::vector<double> SceneValidator::geom_margins() const {
  const mjModel* m = collision_view_ ? collision_view_.get() : scene_->model();
  return {m->geom_margin, m->geom_margin + m->ngeom};
}

std::vector<double> SceneValidator::pair_margins() const {
  const mjModel* m = collision_view_ ? collision_view_.get() : scene_->model();
  return {m->pair_margin, m->pair_margin + m->npair};
}

int SceneValidator::raw_contact_count(ConfigView q) const {
  pose(q);
  return data_->ncon;
}

std::vector<Contact> SceneValidator::contacts(ConfigView q) const {
  std::vector<Contact> out;
  for (const InvalidContact& c : invalid_contacts(q)) {
    out.push_back({c.kind == InvalidContact::Kind::SelfCollision ? "self_collision" : "robot_environment",
                   scene_->body_name(c.body1), scene_->body_name(c.body2), c.geom1, c.geom2, c.dist});
  }
  return out;
}

bool SceneValidator::full_turn_invariant() const {
  const ::mjModel* m = scene_->model();
  return std::all_of(scene_->joint_ids().begin(), scene_->joint_ids().end(),
                     [m](int id) { return m->jnt_type[id] == mjJNT_HINGE; });
}

}  // namespace sscbirrt::mujoco
