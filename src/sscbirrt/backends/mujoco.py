# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""MuJoCo backend for robot model, collision checking, and IK."""

import numpy as np

from sscbirrt.interfaces.collision_checker import CollisionChecker

try:
    import mujoco
except ImportError:
    raise ImportError('The MuJoCo backend requires mujoco: pip install "sscbirrt[mujoco]" (mujoco==3.14.0)') from None


def _names(model: "mujoco.MjModel", kind: str) -> list[str]:
    count = {"site": model.nsite, "joint": model.njnt}[kind]
    return [getattr(model, kind)(i).name for i in range(count)]


def _site_id(model: "mujoco.MjModel", name: str) -> int:
    """The site's id, or a ValueError that lists the sites the model has."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id == -1:
        raise ValueError(
            f"site '{name}' not found in the model; its sites are: {', '.join(_names(model, 'site')) or '(none)'}"
        )
    return site_id


def _joint_ids(model: "mujoco.MjModel", joint_names: list[str] | None) -> np.ndarray:
    """Ids of the named joints, in order. ``None`` means every joint, allowed only when all are hinges or slides:
    a free or ball joint (an object in the scene) is never an arm joint, and taking it silently gives the wrong DOF
    and qpos mapping (#173)."""
    if joint_names is None:
        scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        others = [model.joint(i).name or f"#{i}" for i in range(model.njnt) if int(model.jnt_type[i]) not in scalar]
        if others:
            arm = [model.joint(i).name for i in range(model.njnt) if int(model.jnt_type[i]) in scalar]
            raise ValueError(
                f"joint_names is required: the model has free or ball joints ({', '.join(others)}); "
                f"pass the arm's joints, from: {', '.join(arm)}"
            )
        return np.arange(model.njnt)
    ids = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id == -1:
            raise ValueError(
                f"joint '{name}' not found in the model; its joints are: {', '.join(_names(model, 'joint'))}"
            )
        ids.append(joint_id)
    return np.array(ids, dtype=int)


class MuJoCoRobotModel:
    """Robot model using MuJoCo for forward kinematics."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        ee_site: str,
        joint_names: list[str] | None = None,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Initialize MuJoCo robot model.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            ee_site: Name of end-effector site
            joint_names: The controlled joints, in order. ``None`` means every joint, and is allowed only
                when every joint is a hinge or slide (a scene with free objects must name the arm's joints).
            joint_limits: Optional ``(lower, upper)`` planning limits that replace the model's, for example
                finite limits on a joint the model leaves unlimited.
        """
        self.model = model
        self.data = data
        self.ee_site = ee_site
        self.ee_site_id = _site_id(model, ee_site)
        self.joint_ids = _joint_ids(model, joint_names)
        self._dof = len(self.joint_ids)

        # Cache joint limits. MuJoCo stores an unlimited joint's range as (0, 0);
        # report it honestly as unbounded so the planner's JointSpace can insist
        # the caller declare the joint angular or give finite planning limits.
        if joint_limits is not None:
            lower, upper = (np.array(b, dtype=float) for b in joint_limits)
            if lower.shape != (self._dof,) or upper.shape != (self._dof,):
                raise ValueError(
                    f"joint_limits: expected two arrays of length {self._dof}, got {lower.shape} and {upper.shape}"
                )
            self._lower, self._upper = lower, upper
        else:
            self._lower = self.model.jnt_range[self.joint_ids, 0].copy()
            self._upper = self.model.jnt_range[self.joint_ids, 1].copy()
            unlimited = ~self.model.jnt_limited[self.joint_ids].astype(bool)
            self._lower[unlimited] = -np.inf
            self._upper[unlimited] = np.inf
        self.joint_names = [model.joint(int(i)).name for i in self.joint_ids]

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self._lower, self._upper

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute end-effector pose from joint configuration.

        Args:
            q: Joint configuration

        Returns:
            4x4 homogeneous transform
        """
        # Set joint positions
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = q[i]

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get site pose
        pos = self.data.site_xpos[self.ee_site_id].copy()
        rot_mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

        # Build 4x4 transform
        transform = np.eye(4)
        transform[:3, :3] = rot_mat
        transform[:3, 3] = pos

        return transform


class MuJoCoCollisionChecker:
    """Collision checker using MuJoCo."""

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        joint_names: list[str] | None = None,
    ):
        """Initialize MuJoCo collision checker.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            joint_names: List of joint names (must match robot model)
        """
        self.model = model
        self.data = data
        self.joint_ids = _joint_ids(model, joint_names)

    @property
    def full_turn_invariant(self) -> bool:
        """With only hinges, a full turn leaves the world unchanged: no verdict depends on winding (#200)."""
        return all(int(self.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE) for j in self.joint_ids)

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if configuration is collision-free.

        Args:
            q: Joint configuration

        Returns:
            True if collision-free
        """
        # Set joint positions
        for i, jnt_id in enumerate(self.joint_ids):
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[qpos_adr] = q[i]

        # Run collision detection
        mujoco.mj_forward(self.model, self.data)

        # Check for contacts
        return self.data.ncon == 0

    def is_valid_batch(self, qs: np.ndarray) -> np.ndarray:
        """Check multiple configurations for collisions.

        Args:
            qs: Array of shape (N, dof)

        Returns:
            Boolean array of shape (N,)
        """
        return np.array([self.is_valid(q) for q in qs])


class MuJoCoIKSolver:
    """Differential IK solver using MuJoCo's Jacobian.

    Uses damped least squares (Levenberg-Marquardt) for numerical stability.
    Unlike analytical solvers (e.g., SSIK), this returns at most one solution
    per call, found iteratively from the current configuration.

    To find multiple solutions, call solve() with different initial configurations
    using the `q_init` parameter.
    """

    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        ee_site: str,
        joint_names: list[str] | None = None,
        joint_limits: tuple[np.ndarray, np.ndarray] | None = None,
        collision_checker: CollisionChecker | None = None,
        damping: float = 0.1,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
        restarts: int = 3,
        seed: int | None = None,
    ):
        """Initialize MuJoCo IK solver.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            ee_site: Name of end-effector site
            joint_names: List of joint names to control (if None, uses all joints)
            joint_limits: Optional (lower, upper) arrays for validation
            collision_checker: Optional collision checker for validation
            damping: Damping factor for damped least squares
            max_iterations: Maximum iterations for IK convergence
            restarts: When ``solve`` is called without ``q_init``, number of
                random initial configurations tried after the current state.
                Differential IK from a single start converges on only part of
                the workspace and depends on whatever the shared MjData was
                last left in; restarts make unseeded solves (e.g. TSR goal
                sampling) reliable and surface several distinct solutions.
                Restart windows are anchored at the current configuration
                (clamped into the limits) and derived from each joint's MuJoCo
                type: a hinge samples within one turn of the anchor,
                intersected with its limits; a limited slide samples its whole
                interval; an unlimited slide samples ±1 (model units) around
                the anchor. The window always contains the anchor, so it is
                never empty.
            seed: Seed for the restart generator. This solver is stateful and
                stochastic when unseeded; seed it explicitly for reproducible
                runs (``CBiRRT.plan(seed=...)`` seeds the planner only).
            tolerance: Position/orientation error tolerance for convergence
        """
        self.model = model
        self.data = data
        self.damping = damping
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.joint_limits = joint_limits
        if restarts < 0:
            raise ValueError("restarts must be nonnegative")
        self.restarts = restarts
        self._rng = np.random.default_rng(seed)
        self.collision_checker = collision_checker

        self.ee_site_id = _site_id(model, ee_site)
        self.joint_ids = _joint_ids(model, joint_names)
        self.qpos_adrs = np.array([model.jnt_qposadr[i] for i in self.joint_ids], dtype=int)
        self.dof_adrs = np.array([model.jnt_dofadr[i] for i in self.joint_ids], dtype=int)

        self._dof = len(self.joint_ids)

        # Joint topology for restart windows: hinge joints get a one-turn window,
        # anything else (slide) a translational one. Limited from the model, or
        # from explicit joint_limits when those were given.
        self._is_hinge = self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE
        self._restart_limited = self.model.jnt_limited[self.joint_ids].astype(bool) | (joint_limits is not None)

        # Get joint limits from model if not provided
        if self.joint_limits is None:
            # Check which joints are actually limited
            limited = self.model.jnt_limited[self.joint_ids]
            if np.any(limited):
                lower = self.model.jnt_range[self.joint_ids, 0].copy()
                upper = self.model.jnt_range[self.joint_ids, 1].copy()
                # For unlimited joints, use very large range
                lower[~limited.astype(bool)] = -1e10
                upper[~limited.astype(bool)] = 1e10
                self.joint_limits = (lower, upper)

        # Pre-allocate Jacobian arrays
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def _get_config(self) -> np.ndarray:
        """Get current joint configuration."""
        return np.array([self.data.qpos[adr] for adr in self.qpos_adrs])

    def _set_config(self, q: np.ndarray) -> None:
        """Set joint configuration."""
        for i, adr in enumerate(self.qpos_adrs):
            self.data.qpos[adr] = q[i]

    def _get_jacobian(self) -> np.ndarray:
        """Get 6xN Jacobian for the end-effector site."""
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.ee_site_id)
        # Extract columns for controlled joints only
        jacp = self._jacp[:, self.dof_adrs]
        jacr = self._jacr[:, self.dof_adrs]
        return np.vstack([jacp, jacr])

    def _pose_error(self, target_pose: np.ndarray) -> np.ndarray:
        """Compute 6D pose error (position + orientation)."""
        # Current pose
        current_pos = self.data.site_xpos[self.ee_site_id]
        current_rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)

        # Target pose
        target_pos = target_pose[:3, 3]
        target_rot = target_pose[:3, :3]

        # Position error
        pos_error = target_pos - current_pos

        # Orientation error (using rotation matrix difference)
        rot_error_mat = target_rot @ current_rot.T
        # Convert to axis-angle
        rot_error = self._rotation_matrix_to_axis_angle(rot_error_mat)

        return np.concatenate([pos_error, rot_error])

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation."""
        # Use MuJoCo's quat functions for robustness
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        # Convert quaternion to axis-angle
        angle = 2.0 * np.arccos(np.clip(quat[0], -1.0, 1.0))
        if angle < 1e-10:
            return np.zeros(3)
        axis = quat[1:4] / np.sin(angle / 2.0)
        return axis * angle

    def _clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamp configuration to joint limits."""
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
            return np.clip(q, lower, upper)
        return q

    def solve(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK for a single end-effector pose using differential IK.

        Args:
            pose: 4x4 homogeneous transform of desired end-effector pose
            q_init: Initial configuration. If given, one solve runs from it
                (the seed-nearest solution, as projection wants). If None, one
                solve runs from the current model state and then from
                ``restarts`` random configurations within the joint limits,
                and every distinct converged solution is returned.

        Returns:
            Converged solutions (possibly empty). One when seeded.
        """
        if q_init is not None:
            return self._solve_from(pose, np.asarray(q_init, dtype=float))

        solutions: list[np.ndarray] = []
        q_current = self._get_config()
        inits = [q_current]
        if self.restarts:
            lo, hi = self._restart_bounds(q_current)
            inits += [self._rng.uniform(lo, hi) for _ in range(self.restarts)]
        for q0 in inits:
            for q in self._solve_from(pose, q0):
                if not any(np.linalg.norm(q - s) < 1e-3 for s in solutions):
                    solutions.append(q)
        return solutions

    def _restart_bounds(self, q_anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-joint sampling window for restarts, anchored at ``q_anchor`` (see ``__init__``).

        Never empty: the anchor is clamped into the limits and every window
        contains it. Hinge joints use one turn around the anchor intersected
        with the limits (a differential solver has no use for far windings);
        limited slides use their full interval; unlimited slides use ±1.
        """
        if self.joint_limits is not None:
            lower, upper = self.joint_limits
        else:
            lower, upper = np.full(self._dof, -np.inf), np.full(self._dof, np.inf)
        c = np.clip(np.asarray(q_anchor, dtype=float), lower, upper)
        hinge_lo, hinge_hi = np.maximum(lower, c - np.pi), np.minimum(upper, c + np.pi)
        slide_lo = np.where(self._restart_limited, lower, c - 1.0)
        slide_hi = np.where(self._restart_limited, upper, c + 1.0)
        lo = np.where(self._is_hinge, hinge_lo, slide_lo)
        hi = np.where(self._is_hinge, hinge_hi, slide_hi)
        assert np.all(lo <= hi), (lo, hi)
        return lo, hi

    def _solve_from(self, pose: np.ndarray, q_init: np.ndarray) -> list[np.ndarray]:
        """Damped least squares from one initial configuration; one solution or none."""
        self._set_config(q_init)

        for _ in range(self.max_iterations):
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)

            # Compute error
            error = self._pose_error(pose)
            error_norm = np.linalg.norm(error)

            if error_norm < self.tolerance:
                # Converged
                q = self._get_config()
                return [q]

            # Get Jacobian
            J = self._get_jacobian()

            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
            JJT = J @ J.T
            damped = JJT + self.damping**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(damped, error)

            # Update configuration
            q = self._get_config() + dq
            q = self._clamp_to_limits(q)
            self._set_config(q)

        # Did not converge
        return []

    def solve_valid(self, pose: np.ndarray, q_init: np.ndarray | None = None) -> list[np.ndarray]:
        """Solve IK and return only valid solutions.

        Filters solutions to return only those that are:
        - Within joint limits (if joint_limits provided)
        - Collision-free (if collision_checker provided)

        Args:
            pose: 4x4 homogeneous transform
            q_init: Initial configuration (if None, uses current model state)

        Returns:
            List of valid joint configurations (may be empty)
        """
        solutions = self.solve(pose, q_init)

        valid = []
        for q in solutions:
            # Check joint limits if provided
            if self.joint_limits is not None:
                lower, upper = self.joint_limits
                if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
                    continue

            # Check collisions if checker provided
            if self.collision_checker is not None:
                if not self.collision_checker.is_valid(q):
                    continue

            valid.append(q)

        return valid

    def solve_from_multiple_inits(
        self,
        pose: np.ndarray,
        q_inits: list[np.ndarray],
        return_all: bool = False,
    ) -> list[np.ndarray]:
        """Solve IK from multiple initial configurations.

        Useful for finding multiple solutions with a differential IK solver.

        Args:
            pose: 4x4 homogeneous transform
            q_inits: List of initial configurations to try
            return_all: If True, return all solutions; if False, return first valid

        Returns:
            List of valid solutions
        """
        solutions = []
        for q_init in q_inits:
            result = self.solve_valid(pose, q_init)
            if result:
                solutions.extend(result)
                if not return_all:
                    return solutions
        return solutions


def site_offset_in_body(model: "mujoco.MjModel", site_name: str) -> np.ndarray:
    """The fixed 4x4 transform of a site within its parent body's frame.

    Use it as ``T_ee`` for ``SSIKSolver`` when the SSIK model ends at the
    site's parent body (``ssik.Manipulator.from_mjcf(xml, base="world", ee=<body>)``),
    so that SSIK's forward kinematics lands on the site, as
    ``MuJoCoRobotModel.forward_kinematics`` does.
    """
    site_id = _site_id(model, site_name)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.site_quat[site_id])
    T = np.eye(4)
    T[:3, :3] = rot.reshape(3, 3)
    T[:3, 3] = model.site_pos[site_id]
    return T
