// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sscbirrt/kinematics.hpp"
#include "sscbirrt/transform.hpp"

namespace sscbirrt::ssik {

// The verified-family allowlist. A family enters when the C++ adapter has been checked against
// Python's SSIKSolver on that family's robot (docs/native-design.md, v1.6.0 addendum).
enum class Family { ThreeParallel };
std::optional<Family> family_from_solver_name(const std::string& solver_name);  // "ikgeo.three_parallel"
const char* family_name(Family f);

// Rendered from ssik.cpp.joint_data(manipulator): JointConsts<6> and JointLimits<6> as plain arrays,
// plus the SSIKSolver adapter's frame offsets.
struct ArmSpec {
  std::string solver_name;                              // e.g. "ikgeo.three_parallel"
  std::array<std::array<double, 3>, 6> axis{};
  std::array<std::array<double, 16>, 6> t_left{};      // row-major 4x4
  std::array<std::array<double, 16>, 6> t_right{};
  std::array<bool, 6> revolute{};
  std::array<double, 6> lo{}, hi{};
  std::array<bool, 6> present{};                        // false: the joint has no limits
  Transform T_base = Transform::identity();             // SSIKSolver.T_base: SSIK base -> robot base
  Transform T_ee = Transform::identity();               // SSIKSolver.T_ee: SSIK ee -> robot ee
};

// SSIK through ssik's header-only family solvers: SSIKSolver.solve and .fk in C++.
//   fk(q)    = T_base * ssik::fk(consts, q) * T_ee
//   solve(T) = every solution of the family solver for inv(T_base) * T * inv(T_ee), with the Python
//              adapter's parameters: respect_limits, enumerate_windings, no cap, seed iff given.
class SSIKArm final : public ForwardKinematics, public IKSolver {
 public:
  explicit SSIKArm(ArmSpec spec);  // throws std::invalid_argument for an unverified family or a malformed spec
  ~SSIKArm() override;
  SSIKArm(const SSIKArm&) = delete;
  SSIKArm& operator=(const SSIKArm&) = delete;

  int dof() const override { return 6; }
  Transform fk(ConfigView q) const override;
  std::vector<Config> solve(const Transform& pose, ConfigView seed) const override;
  std::vector<bool> revolute_joints() const override { return std::vector<bool>(6, true); }  // ssik arms are all revolute

  Family family() const { return family_; }
  const ArmSpec& spec() const { return spec_; }

 private:
  struct Impl;
  ArmSpec spec_;
  Family family_;
  Impl* impl_;
};

}  // namespace sscbirrt::ssik
