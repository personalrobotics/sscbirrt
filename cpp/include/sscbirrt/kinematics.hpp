// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <vector>

#include "sscbirrt/transform.hpp"
#include "sscbirrt/types.hpp"

namespace sscbirrt {

// RobotModel.forward_kinematics: the end-effector pose in the planner's frames. Joint limits are the
// space's, not the model's, as in Python.
class ForwardKinematics {
 public:
  virtual ~ForwardKinematics() = default;
  virtual int dof() const = 0;
  virtual Transform fk(ConfigView q) const = 0;
};

// IKSolver.solve: every solution, unfiltered (no collision, no cap). seed may be empty.
class IKSolver {
 public:
  virtual ~IKSolver() = default;
  virtual int dof() const = 0;
  virtual std::vector<Config> solve(const Transform& pose, ConfigView seed) const = 0;
  // Joints on which q and q + 2*pi*k place the links identically, as declared by the solver (#200). Empty: none
  // declared, so solutions are never treated as the same physical configuration.
  virtual std::vector<bool> revolute_joints() const { return {}; }
};

}  // namespace sscbirrt
