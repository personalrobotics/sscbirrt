// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
//
// The only translation unit that sees Eigen and ssik_cpp's solvers.
#include "sscbirrt/ssik/ssik_arm.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include "ssik_cpp/solve.hpp"

namespace sscbirrt::ssik {

namespace {

Eigen::Matrix4d to_eigen(const Transform& T) {
  Eigen::Matrix4d out;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) out(r, c) = T.at(r, c);
  }
  return out;
}

Transform from_eigen(const Eigen::Matrix4d& M) {
  Transform T;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) T.at(r, c) = M(r, c);
  }
  return T;
}

template <int N>
void solve_n(const ::ssik::Solver& s, const Eigen::Matrix4d& T, ConfigView seed, std::vector<Config>& out) {
  ::ssik::ArtifactParams<N> p;  // the Python solve() defaults: respect_limits, enumerate_windings, no cap
  if (!seed.empty()) {
    p.has_seed = true;
    std::copy_n(seed.begin(), N, p.q_seed.begin());
  }
  for (const auto& sol : s.solve<N>(T, p)) out.emplace_back(sol.q.begin(), sol.q.end());
}

template <int N>
Eigen::Matrix4d fk_n(const ::ssik::Solver& s, ConfigView q) {
  std::array<double, N> qa{};
  std::copy_n(q.begin(), N, qa.begin());
  return s.fk<N>(qa);
}

}  // namespace

struct SSIKArm::Impl {
  ::ssik::Solver solver;
};

SSIKArm::SSIKArm(ArmSpec spec) : spec_(std::move(spec)) {
  if (auto why = why_not_frame(spec_.T_base, "T_base")) throw std::invalid_argument(*why);
  if (auto why = why_not_frame(spec_.T_ee, "T_ee")) throw std::invalid_argument(*why);
  impl_ = std::make_unique<Impl>(Impl{::ssik::Solver(spec_.data)});  // validates the data against ssik's schema
  // ssik's joint_type: 0 revolute, otherwise prismatic. Root collection shares a verdict across full turns of
  // revolute joints only (#200).
  const auto& types = spec_.data.arrays.at("joint_type").i32;
  revolute_.assign(types.size(), false);
  for (std::size_t j = 0; j < types.size(); ++j) revolute_[j] = types[j] == 0;
}

SSIKArm::~SSIKArm() = default;

int SSIKArm::dof() const { return impl_->solver.dof(); }

const std::string& SSIKArm::family() const { return impl_->solver.family(); }

Transform SSIKArm::fk(ConfigView q) const {
  if (static_cast<int>(q.size()) != dof()) {
    throw std::invalid_argument("SSIKArm::fk expects " + std::to_string(dof()) + " joint values");
  }
  const Eigen::Matrix4d T_arm = dof() == 6 ? fk_n<6>(impl_->solver, q) : fk_n<7>(impl_->solver, q);
  return from_eigen(to_eigen(spec_.T_base) * T_arm * to_eigen(spec_.T_ee));
}

std::vector<Config> SSIKArm::solve(const Transform& pose, ConfigView seed) const {
  if (!seed.empty() && static_cast<int>(seed.size()) != dof()) {
    throw std::invalid_argument("SSIKArm::solve seed must have " + std::to_string(dof()) + " values");
  }
  const Eigen::Matrix4d target = to_eigen(spec_.T_base).inverse() * to_eigen(pose) * to_eigen(spec_.T_ee).inverse();
  std::vector<Config> out;
  if (dof() == 6) {
    solve_n<6>(impl_->solver, target, seed, out);
  } else {
    solve_n<7>(impl_->solver, target, seed, out);
  }
  return out;
}

}  // namespace sscbirrt::ssik
