// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <vector>

#include "sscbirrt/mujoco/scene.hpp"
#include "sscbirrt/mujoco/snapshot.hpp"
#include "sscbirrt/validity.hpp"

struct mjData_;
struct mjModel_;

namespace sscbirrt::mujoco {

struct InvalidContact {
  enum class Kind { SelfCollision, RobotEnvironment };
  int body1, body2;  // bodies of the two geoms
  int geom1, geom2;
  double dist;       // MuJoCo's contact distance; negative is penetration
  Kind kind;
};

// mj_manipulator's snapshot-mode CollisionChecker, step for step (docs/native-design.md, v1.7.0):
// restore the snapshot, write the controlled joints, kinematics, move attached free bodies with their
// gripper, kinematics again, collision, then classify every contact by body. Every contact MuJoCo
// generates counts, margin-inflated ones included. One instance owns one mjData: one validator per solve.
//
// Collision skips the pairs whose contacts the classification discards (#254): two bodies that are neither arm
// bodies nor in a held object's subtree. A private view of the scene's mjModel (the struct copied, its arrays
// shared) blocks those pairs with MuJoCo's body-level masks, which mj_collision uses only to prefilter body pairs;
// every remaining geom pair passes MuJoCo's own geom-level filter, so the contacts that count, and their order, are
// those of a full mj_collision. filter_passive_pairs = false runs the full mj_collision (the reference path).
//
// A clearance (#256) keeps the moving set (arm bodies and held objects' subtrees) at least that far from the
// environment (every other body). The view gives each environment geom margin max(authored, clearance) and raises
// body_margin (MuJoCo prunes BVH nodes by it) and the pair_margin of every predefined moving-environment <pair> to
// match; moving geoms keep their authored margins. MuJoCo adds the two geom margins of a pair, so the separation
// kept is at least the clearance, exactly the clearance where the moving geom's authored margin is 0. Clearance 0
// leaves every margin as authored. The scene is never written: each validator owns its margins.
struct CollisionOptions {
  double clearance = 0.0;           // metres, finite and >= 0
  bool filter_passive_pairs = true;
};

class SceneValidator final : public StateValidator {
 public:
  // Validates the snapshot; throws std::invalid_argument for a bad clearance, and for clearance > 0 on a model whose
  // margins MuJoCo overrides (mjENBL_OVERRIDE) or with a collision-enabled flex that can touch the moving set.
  SceneValidator(std::shared_ptr<const Scene> scene, Snapshot snapshot, CollisionOptions options = {});
  ~SceneValidator() override;
  SceneValidator(const SceneValidator&) = delete;
  SceneValidator& operator=(const SceneValidator&) = delete;

  bool is_valid(ConfigView q) const override;
  bool full_turn_invariant() const override;  // every controlled joint is a hinge
  std::vector<InvalidContact> invalid_contacts(ConfigView q) const;
  std::vector<Contact> contacts(ConfigView q) const override;  // invalid_contacts with body names
  int raw_contact_count(ConfigView q) const;  // contacts mj_collision generates at q, before classification
  const Scene& scene() const { return *scene_; }
  const Snapshot& snapshot() const { return snapshot_; }
  double clearance() const { return clearance_; }
  std::vector<int> moving_bodies() const;      // the arm bodies and the held objects' subtrees, ascending
  std::vector<double> geom_margins() const;    // the margins this validator collides with, per geom
  std::vector<double> pair_margins() const;    // and per predefined <pair>

 private:
  void pose(ConfigView q) const;  // steps 1 to 6 into data_
  std::shared_ptr<const Scene> scene_;
  Snapshot snapshot_;
  ::mjData_* data_;                       // private; mutated by the const queries, never shared
  std::vector<int> attachment_of_body_;   // per body: index of the attachment whose object subtree holds it, or -1
  std::unique_ptr<::mjModel_> collision_view_;     // the scene's model with the masks below; null when unfiltered
  std::vector<int> body_contype_, body_conaffinity_;  // moving bodies (~0, ~0), passive bodies (~0, 0)
  std::vector<double> geom_margin_, body_margin_, pair_margin_;  // the clearance's margins; empty at clearance 0
  double clearance_ = 0.0;
};

}  // namespace sscbirrt::mujoco
