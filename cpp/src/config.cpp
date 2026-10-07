// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#include "sscbirrt/config.hpp"

#include <sstream>
#include <stdexcept>
#include <string>

namespace sscbirrt {

namespace {
template <class T>
void check(const char* name, bool ok, const char* requirement, T value) {
  if (ok) return;
  std::ostringstream os;
  os << name << " must be " << requirement << ", got " << value;
  throw std::invalid_argument(os.str());
}
}  // namespace

void PlannerConfig::validate() const {
  check("timeout", timeout_seconds > 0, "positive", timeout_seconds);
  check("step_size", step_size > 0, "positive", step_size);
  check("progress_tolerance", progress_tolerance > 0, "positive", progress_tolerance);
  check("connection_tolerance", connection_tolerance >= 0, "nonnegative", connection_tolerance);
  check("max_iterations", max_iterations >= 1, "at least 1", max_iterations);
  check("tsr_samples", sample_draws >= 1, "at least 1", sample_draws);
  check("num_tree_roots", num_tree_roots >= 1, "at least 1", num_tree_roots);
  if (max_per_draw) check("max_per_draw", *max_per_draw >= 1, "None or at least 1", *max_per_draw);
  check("smoothing_iterations", smoothing_iterations >= 0, "nonnegative", smoothing_iterations);
  check("smoothing_patience", smoothing_patience >= 0, "nonnegative", smoothing_patience);
  if (edge_resolution) check("edge_resolution", *edge_resolution > 0, "None or positive", *edge_resolution);
  if (extend_steps) check("extend_steps", *extend_steps >= 1, "None or at least 1", *extend_steps);
  if (connect_steps) check("connect_steps", *connect_steps >= 1, "None or at least 1", *connect_steps);
  check("start_sample_probability", start_sample_probability >= 0.0 && start_sample_probability <= 1.0, "within [0, 1]",
        start_sample_probability);
  check("goal_sample_probability", goal_sample_probability >= 0.0 && goal_sample_probability <= 1.0, "within [0, 1]",
        goal_sample_probability);
}

}  // namespace sscbirrt
