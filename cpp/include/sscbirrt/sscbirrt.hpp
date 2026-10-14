// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Siddhartha Srinivasa
#pragma once

// The core: standard library only. The pose regions (sscbirrt/tsr/*.hpp, target sscbirrt::tsr) are built on
// sstsr's C++ core and are included on their own, so a core-only consumer never needs sstsr.

#include "sscbirrt/cancel.hpp"
#include "sscbirrt/config.hpp"
#include "sscbirrt/errors.hpp"
#include "sscbirrt/kinematics.hpp"
#include "sscbirrt/motion.hpp"
#include "sscbirrt/planner.hpp"
#include "sscbirrt/problem.hpp"
#include "sscbirrt/result.hpp"
#include "sscbirrt/sets.hpp"
#include "sscbirrt/space.hpp"
#include "sscbirrt/transform.hpp"
#include "sscbirrt/types.hpp"
#include "sscbirrt/validation.hpp"
#include "sscbirrt/validity.hpp"
