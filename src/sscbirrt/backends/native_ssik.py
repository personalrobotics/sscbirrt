# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Render an ``ssik.Manipulator`` into the native SSIK adapter (docs/native-design.md, v1.6.0 addendum)."""

from __future__ import annotations

import numpy as np

from sscbirrt import _native


def unsupported_reason(manipulator) -> str | None:
    """Why this manipulator cannot be lifted natively, or None. Every family ssik solves natively is supported
    (``ssik.cpp.NATIVE_FAMILIES``); ssik verifies those solvers against its Python, so sscbirrt adds no allowlist."""
    if not _native.has_ssik():
        return _native.ssik_unavailable_reason()
    import ssik.cpp

    name = getattr(manipulator, "solver_name", None)
    if name not in ssik.cpp.NATIVE_FAMILIES:
        return f"SSIK family {name} has no native solver in ssik {ssik.__version__}"
    return None


def arm_from_manipulator(manipulator, *, T_base: np.ndarray | None = None, T_ee: np.ndarray | None = None):
    """``ssik.cpp.solver_data(manipulator)`` handed to ``_native.SSIKArm`` with the adapter's frame offsets."""
    import ssik.cpp

    why = unsupported_reason(manipulator)
    if why is not None:
        raise ValueError(why)
    data = ssik.cpp.solver_data(manipulator)  # ssik.cpp.UnsupportedFamilyError (a ValueError) if it has none
    eye = np.eye(4)
    return _native.SSIKArm(
        data,
        (eye if T_base is None else np.asarray(T_base, dtype=float)).tolist(),
        (eye if T_ee is None else np.asarray(T_ee, dtype=float)).tolist(),
    )
