// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/mujoco/validator.hpp"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace sscbirrt::mujoco {

SceneValidator::SceneValidator(std::shared_ptr<const Scene> scene, Snapshot snapshot)
    : scene_(std::move(scene)), snapshot_(std::move(snapshot)), data_(nullptr) {
  if (!scene_) throw std::invalid_argument("SceneValidator needs a scene");
  snapshot_.validate(*scene_);
  attachment_of_body_.assign(static_cast<std::size_t>(scene_->nbody()), -1);
  for (std::size_t i = 0; i < snapshot_.attachments.size(); ++i) {
    for (int b : scene_->subtree(snapshot_.attachments[i].object_body)) attachment_of_body_[static_cast<std::size_t>(b)] = static_cast<int>(i);
  }
  data_ = mj_makeData(scene_->model());
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
  // 6. collision
  mj_collision(m, data_);
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
    if (robot1 && robot2) {
      // 7. a grasped object touching a body its attachment allows is the grasp itself, not a collision
      auto allowed = [&](int att, int other) {
        if (att < 0) return false;
        const std::vector<int>& ok = snapshot_.attachments[static_cast<std::size_t>(att)].allowed_bodies;
        return std::binary_search(ok.begin(), ok.end(), other);
      };
      if (allowed(att1, b2) || allowed(att2, b1)) continue;
      out.push_back({b1, b2, g1, g2, c.dist, InvalidContact::Kind::SelfCollision});
      continue;
    }
    out.push_back({b1, b2, g1, g2, c.dist, InvalidContact::Kind::RobotEnvironment});
  }
  return out;
}

bool SceneValidator::is_valid(ConfigView q) const { return invalid_contacts(q).empty(); }

bool SceneValidator::full_turn_invariant() const {
  const ::mjModel* m = scene_->model();
  return std::all_of(scene_->joint_ids().begin(), scene_->joint_ids().end(),
                     [m](int id) { return m->jnt_type[id] == mjJNT_HINGE; });
}

}  // namespace sscbirrt::mujoco
