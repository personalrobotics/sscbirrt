// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <optional>

namespace sscbirrt {

// Mirrors CBiRRTConfig. Set-owned tolerances (membership, projection) live on the sets and
// strategies; angular joints live on JointSpace; abort_fn is a CancellationToken.
struct PlannerConfig {
  // Termination
  double timeout_seconds = 30.0;
  int max_iterations = 100000;

  // Tolerances
  double connection_tolerance = 1e-3;     // growth has reached its target within this joint-space distance
  std::optional<double> edge_resolution = 0.05;  // spacing of collision checks along an edge (#204); nullopt: step_size
  double progress_tolerance = 1e-6;       // growth stops when the distance to target shrinks by less than this

  // Growth
  double step_size = 0.1;
  // CBiRRT's P_sample (#196): on a tree's turn, the probability that the turn adds roots drawn from that tree's own
  // set instead of extending toward a random configuration. A finite set never does: its members are all roots.
  double start_sample_probability = 0.1;
  double goal_sample_probability = 0.1;
  std::optional<int> extend_steps;   // nullopt: connect until blocked
  std::optional<int> connect_steps;

  // Roots (Python: tsr_samples, num_tree_roots, max_ik_per_pose)
  int sample_draws = 100;
  int num_tree_roots = 100;
  int max_per_draw = 3;

  // Smoothing
  bool smooth_path = true;
  int smoothing_iterations = 50;
  int smoothing_patience = 15;

  // Throws std::invalid_argument("<field> must be <requirement>, got <value>") with CBiRRTConfig's ranges.
  void validate() const;
  double resolution() const { return edge_resolution.value_or(step_size); }
};

}  // namespace sscbirrt
