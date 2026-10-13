// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "sscbirrt/kinematics.hpp"
#include "sscbirrt/transform.hpp"
#include "ssik_cpp/solver_data.hpp"  // Eigen-free: the arm's complete solver input as named arrays

namespace sscbirrt::ssik {

// An arm as ssik.cpp.solver_data(manipulator) renders it (ssik >= 8.2, ssik#707), plus the SSIKSolver adapter's frame
// offsets. Every family ssik solves natively is supported (ssik::native_families(): the UR three-parallel arms, general
// 6R arms such as Kinova's, and the 7R SRS and spherical-shoulder arms such as the iiwa and the Panda). ssik verifies
// its own native solvers against its Python, so sscbirrt keeps no family allowlist of its own (#220).
struct ArmSpec {
  ::ssik::SolverData data;
  Transform T_base = Transform::identity();  // SSIKSolver.T_base: SSIK base -> robot base
  Transform T_ee = Transform::identity();    // SSIKSolver.T_ee: SSIK ee -> robot ee
};

// SSIK through ssik::Solver: SSIKSolver.solve and .fk in C++.
//   fk(q)    = T_base * ssik fk(q) * T_ee
//   solve(T) = every solution of the family solver for inv(T_base) * T * inv(T_ee), with the Python adapter's
//              parameters: respect_limits, enumerate_windings, no cap, seed iff given. Exactly ssik's
//              solve(native=True).
class SSIKArm final : public ForwardKinematics, public IKSolver {
 public:
  explicit SSIKArm(ArmSpec spec);  // throws ssik::SolverDataError (an std::invalid_argument) for malformed or
                                   // unsupported data, std::invalid_argument for a malformed frame
  ~SSIKArm() override;
  SSIKArm(const SSIKArm&) = delete;
  SSIKArm& operator=(const SSIKArm&) = delete;

  int dof() const override;
  Transform fk(ConfigView q) const override;
  std::vector<Config> solve(const Transform& pose, ConfigView seed) const override;
  std::vector<bool> revolute_joints() const override { return revolute_; }

  const std::string& family() const;  // the dispatched solver, e.g. "seven_r.spherical_shoulder"
  const ArmSpec& spec() const { return spec_; }

 private:
  struct Impl;
  ArmSpec spec_;
  std::vector<bool> revolute_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sscbirrt::ssik
