// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include <sstsr/rng.hpp>
#include <sstsr/transform.hpp>
#include <sstsr/tsr.hpp>
#include <sstsr/tsr_chain.hpp>

#include "sscbirrt/transform.hpp"
#include "sscbirrt/types.hpp"

namespace sscbirrt::tsr {

// The pose regions are sstsr's own C++ (sstsr >= 3.3, #184): one implementation of the TSR and TSR-chain rules,
// held to sstsr's Python by sstsr's conformance corpus. This header is the boundary to it.
using sstsr::Bounds6;
using sstsr::TSR;
using sstsr::TSRChain;

// The core keeps its own Transform so that it depends on the standard library alone; the two are the same row-major
// 4x4 of doubles, so crossing the boundary is a copy.
inline sstsr::Transform to_sstsr(const Transform& T) {
  sstsr::Transform out;
  out.m = T.m;
  return out;
}

inline Transform from_sstsr(const sstsr::Transform& T) {
  Transform out;
  out.m = T.m;
  return out;
}

// The planner's generator is passed straight to sstsr's sampling, so a seeded solve stays repeatable.
static_assert(std::is_same_v<Rng, sstsr::Rng>, "sscbirrt and sstsr must draw from the same engine");

// A pose region: a single TSR, or a chain of TSRs composed in series (a handle on a hinged door). A chain's distance
// is the residual of a bounded inverse solve: an upper bound, exact on sstsr's exact paths, so a configuration it
// rejects may still be in the region (sstsr #85). The same holds in sscbirrt's Python reference.
using Region = std::variant<TSR, TSRChain>;

double region_distance(const Region& region, const Transform& T);
std::pair<double, Transform> region_closest_transform(const Region& region, const Transform& T);
Transform region_sample(const Region& region, Rng& rng);
double region_volume(const Region& region);  // sstsr's _interval_sum; a chain's is the sum over its links

}  // namespace sscbirrt::tsr
