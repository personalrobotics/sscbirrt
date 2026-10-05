# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""Example: Simple 2-DOF planar arm planning with CBiRRT.

This example demonstrates three planning scenarios:
1. Basic planning with fixed start and goal TSR
2. Planning with both start and goal TSRs (no fixed configurations)
3. Constrained planning with a trajectory-wide constraint TSR
4. Several start and goal configurations, connected by whichever pair the planner reaches

This example requires only numpy and TSR - no MuJoCo or SSIK needed.

Run with:
    python examples/planar_arm.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from tsr import TSR

from sscbirrt import CBiRRT, CBiRRTConfig
from sscbirrt.testing import PlanarArm, PlanarIK


def joint_positions(robot: PlanarArm, q: np.ndarray) -> list[np.ndarray]:
    """Base, elbow, and tip of the arm in the plane, for drawing and for the obstacle checker."""
    elbow = np.array([robot.l1 * np.cos(q[0]), robot.l1 * np.sin(q[0])])
    tip = elbow + np.array([robot.l2 * np.cos(q[0] + q[1]), robot.l2 * np.sin(q[0] + q[1])])
    return [np.zeros(2), elbow, tip]


class CircleObstacleChecker:
    """Collision checker with circular obstacles."""

    def __init__(self, robot: PlanarArm, obstacles: list[tuple[np.ndarray, float]]):
        """
        Args:
            robot: The planar arm robot
            obstacles: List of (center, radius) tuples
        """
        self.robot = robot
        self.obstacles = obstacles

    def is_valid(self, q: np.ndarray) -> bool:
        """Check if arm configuration collides with any obstacle."""
        positions = joint_positions(self.robot, q)

        # Check each link (base->elbow, elbow->ee)
        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]

            for center, radius in self.obstacles:
                if self._segment_circle_collision(p1, p2, center, radius):
                    return False

        return True

    def _segment_circle_collision(self, p1: np.ndarray, p2: np.ndarray, center: np.ndarray, radius: float) -> bool:
        """Check if line segment intersects circle."""
        d = p2 - p1
        f = p1 - center

        a = np.dot(d, d)
        b = 2 * np.dot(f, d)
        c = np.dot(f, f) - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return False

        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if intersection is within segment
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


def visualize_result(
    robot: PlanarArm,
    path: list[np.ndarray],
    obstacles: list[tuple[np.ndarray, float]],
    tree_start,
    tree_goal,
    collision_checker: CircleObstacleChecker,
    start_region: tuple[np.ndarray, float] | None = None,
    goal_region: tuple[np.ndarray, float] | None = None,
    constraint_region: tuple[float, float, float, float] | None = None,
    title: str = "CBiRRT Planning Result",
    filename: str = "planar_arm_result.png",
):
    """Visualize planning result with workspace path and C-space trees side by side.

    Args:
        robot: The planar arm robot
        path: List of joint configurations
        obstacles: List of (center, radius) obstacle tuples
        tree_start: RRT tree rooted at start
        tree_goal: RRT tree rooted at goal
        collision_checker: Collision checker for C-space visualization
        start_region: Optional (center, radius) for start TSR visualization
        goal_region: Optional (center, radius) for goal TSR visualization
        constraint_region: Optional (x_min, x_max, y_min, y_max) for constraint TSR
        title: Plot title
        filename: Output filename
    """
    fig, (ax_ws, ax_cs) = plt.subplots(1, 2, figsize=(16, 7))

    # === Left panel: Workspace visualization ===
    # Draw constraint region first (behind everything)
    if constraint_region is not None:
        x_min, x_max, y_min, y_max = constraint_region
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, color="yellow", alpha=0.2, label="Constraint region"
        )
        ax_ws.add_patch(rect)

    # Draw obstacles
    for center, radius in obstacles:
        circle = Circle(center, radius, color="red", alpha=0.5)
        ax_ws.add_patch(circle)

    # Draw start region
    if start_region is not None:
        center, radius = start_region
        circle = Circle(center, radius, color="blue", alpha=0.2, label="Start region")
        ax_ws.add_patch(circle)

    # Draw goal region
    if goal_region is not None:
        center, radius = goal_region
        circle = Circle(center, radius, color="green", alpha=0.2, label="Goal region")
        ax_ws.add_patch(circle)

    # Draw path (fading from blue to green)
    n_frames = len(path)
    for i, q in enumerate(path):
        alpha = 0.2 + 0.8 * (i / n_frames)
        positions = joint_positions(robot, q)
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        color = plt.cm.viridis(i / n_frames)
        ax_ws.plot(xs, ys, "o-", color=color, alpha=alpha, linewidth=2, markersize=5)

    # Draw start and end prominently
    start_positions = joint_positions(robot, path[0])
    end_positions = joint_positions(robot, path[-1])

    ax_ws.plot(
        [p[0] for p in start_positions],
        [p[1] for p in start_positions],
        "bo-",
        linewidth=4,
        markersize=10,
        label="Start config",
    )
    ax_ws.plot(
        [p[0] for p in end_positions],
        [p[1] for p in end_positions],
        "go-",
        linewidth=4,
        markersize=10,
        label="End config",
    )

    ax_ws.set_xlim(-2.5, 2.5)
    ax_ws.set_ylim(-2.5, 2.5)
    ax_ws.set_aspect("equal")
    ax_ws.grid(True, alpha=0.3)
    ax_ws.legend(loc="upper left")
    ax_ws.set_title(f"Workspace ({len(path)} waypoints)")
    ax_ws.set_xlabel("x")
    ax_ws.set_ylabel("y")

    # === Right panel: Configuration space visualization ===
    # Draw collision regions in C-space by sampling
    q1_range = np.linspace(-np.pi, np.pi, 100)
    q2_range = np.linspace(-np.pi, np.pi, 100)
    Q1, Q2 = np.meshgrid(q1_range, q2_range)
    collision_map = np.zeros_like(Q1)

    for i in range(Q1.shape[0]):
        for j in range(Q1.shape[1]):
            q = np.array([Q1[i, j], Q2[i, j]])
            collision_map[i, j] = 0 if collision_checker.is_valid(q) else 1

    ax_cs.contourf(Q1, Q2, collision_map, levels=[0.5, 1.5], colors=["red"], alpha=0.3)

    # Draw tree edges (handling angular wraparound)
    def draw_tree(tree, color: str, label: str):
        for node in tree.nodes:
            if node.parent is not None:
                parent = tree.nodes[node.parent]
                # Skip edges that wrap around ±π (would draw long lines across plot)
                diff = np.abs(node.config - parent.config)
                if diff[0] > np.pi or diff[1] > np.pi:
                    continue
                ax_cs.plot(
                    [parent.config[0], node.config[0]],
                    [parent.config[1], node.config[1]],
                    color=color,
                    alpha=0.4,
                    linewidth=0.5,
                )
        # Draw nodes
        configs = np.array([n.config for n in tree.nodes])
        ax_cs.scatter(configs[:, 0], configs[:, 1], c=color, s=5, alpha=0.6, label=label)

    draw_tree(tree_start, "blue", f"Start tree ({len(tree_start)} nodes)")
    draw_tree(tree_goal, "green", f"Goal tree ({len(tree_goal)} nodes)")

    # Draw path (handling angular wraparound)
    path_arr = np.array(path)
    for i in range(len(path_arr) - 1):
        diff = np.abs(path_arr[i + 1] - path_arr[i])
        if diff[0] > np.pi or diff[1] > np.pi:
            continue  # Skip segments that wrap around
        ax_cs.plot(
            [path_arr[i, 0], path_arr[i + 1, 0]],
            [path_arr[i, 1], path_arr[i + 1, 1]],
            "k-",
            linewidth=2,
        )
    ax_cs.plot([], [], "k-", linewidth=2, label="Path")  # For legend
    ax_cs.scatter(path_arr[:, 0], path_arr[:, 1], c="black", s=30, zorder=5)

    # Mark start and goal
    ax_cs.scatter(
        [tree_start.nodes[0].config[0]],
        [tree_start.nodes[0].config[1]],
        c="blue",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=10,
        label="Start",
    )
    ax_cs.scatter(
        [tree_goal.nodes[0].config[0]],
        [tree_goal.nodes[0].config[1]],
        c="green",
        s=200,
        marker="*",
        edgecolors="black",
        linewidths=2,
        zorder=10,
        label="Goal",
    )

    ax_cs.set_xlim(-np.pi, np.pi)
    ax_cs.set_ylim(-np.pi, np.pi)
    ax_cs.set_xlabel("q1 (rad)")
    ax_cs.set_ylabel("q2 (rad)")
    ax_cs.set_aspect("equal")
    ax_cs.grid(True, alpha=0.3)
    ax_cs.legend(loc="upper right")
    ax_cs.set_title("Configuration Space")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved visualization to {filename}")
    plt.show()


def make_position_tsr(x: float, y: float, tolerance: float = 0.1) -> TSR:
    """Create a TSR for a position with given tolerance."""
    T0_w = np.eye(4)
    T0_w[0, 3] = x
    T0_w[1, 3] = y

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array(
            [
                [-tolerance, tolerance],  # x tolerance
                [-tolerance, tolerance],  # y tolerance
                [0, 0],  # z (unused in 2D)
                [0, 0],  # roll
                [0, 0],  # pitch
                [0, 0],  # yaw
            ]
        ),
    )


def make_y_constraint_tsr(y_min: float, y_max: float) -> TSR:
    """Create a TSR that constrains the end-effector y-coordinate.

    This creates a "horizontal band" constraint where the end-effector
    must stay within y_min <= y <= y_max.
    """
    # Center the TSR at y=(y_min+y_max)/2
    T0_w = np.eye(4)
    T0_w[1, 3] = (y_min + y_max) / 2

    y_range = (y_max - y_min) / 2

    return TSR(
        T0_w=T0_w,
        Tw_e=np.eye(4),
        Bw=np.array(
            [
                [-10, 10],  # x: effectively unconstrained
                [-y_range, y_range],  # y: constrained to band
                [0, 0],  # z
                [0, 0],  # roll
                [0, 0],  # pitch
                [0, 0],  # yaw
            ]
        ),
    )


def example_basic():
    """Example 1: Basic planning with fixed start and goal TSR."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Planning (fixed start, goal TSR)")
    print("=" * 60)

    robot = PlanarArm()
    obstacles = [
        (np.array([1.2, 0.5]), 0.2),
        (np.array([0.5, 1.2]), 0.15),
    ]
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarIK(robot)

    config = CBiRRTConfig(
        step_size=0.3,
        goal_sample_probability=0.15,
        smooth_path=True,
        smoothing_iterations=50,
        continuous_joints=(True, True),  # both joints turn freely; ±π is only how their angles are written
    )
    planner = CBiRRT(
        robot=robot,
        ik=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Fixed start configuration
    start = np.array([0.0, 0.0])

    # Goal TSR at (1.5, 1.0) with tolerance
    goal_pos = np.array([1.5, 1.0])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.1)

    print(f"  Start: {start} (fixed)")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.1")
    print(f"  Obstacles: {len(obstacles)}")

    result = planner.plan(start, goal_tsrs=[goal_tsr], seed=42, return_details=True)

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        return

    print(f"Found path with {len(result.path)} waypoints")

    final_pose = robot.forward_kinematics(result.path[-1])
    final_pos = final_pose[:2, 3]
    print(f"  Final position: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")

    visualize_result(
        robot,
        result.path,
        obstacles,
        result.tree_start,
        result.tree_goal,
        collision_checker,
        goal_region=(goal_pos, 0.1),
        title="Example 1: Basic Planning",
        filename="example1_result.png",
    )


def example_start_goal_tsrs():
    """Example 2: Planning with both start and goal TSRs."""
    print("\n" + "=" * 60)
    print("Example 2: Planning with Start and Goal TSRs")
    print("=" * 60)

    robot = PlanarArm()
    # Small obstacle that doesn't block too much
    obstacles = [
        (np.array([1.0, 0.0]), 0.15),
    ]
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarIK(robot)

    config = CBiRRTConfig(
        step_size=0.3,
        goal_sample_probability=0.15,
        membership_tolerance=0.05,  # larger tolerances for TSR membership and tree connection
        connection_tolerance=0.05,
        smooth_path=True,
        smoothing_iterations=50,
        continuous_joints=(True, True),  # both joints turn freely; ±π is only how their angles are written
    )
    planner = CBiRRT(
        robot=robot,
        ik=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Start TSR: anywhere near (1.5, 0.5) - upper right quadrant
    start_pos = np.array([1.5, 0.5])
    start_tsr = make_position_tsr(start_pos[0], start_pos[1], tolerance=0.2)

    # Goal TSR: anywhere near (0.5, 1.5) - upper left quadrant
    goal_pos = np.array([0.5, 1.5])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.2)

    print(f"  Start region: ({start_pos[0]}, {start_pos[1]}) +/- 0.2")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.2")
    print(f"  Obstacles: {len(obstacles)}")

    result = planner.plan(
        start=None,  # No fixed start - sample from start_tsrs
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        seed=42,
        return_details=True,
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        return

    print(f"Found path with {len(result.path)} waypoints")

    # Verify start and goal
    start_pose = robot.forward_kinematics(result.path[0])
    final_pose = robot.forward_kinematics(result.path[-1])
    print(f"  Sampled start: ({start_pose[0, 3]:.3f}, {start_pose[1, 3]:.3f})")
    print(f"  Final position: ({final_pose[0, 3]:.3f}, {final_pose[1, 3]:.3f})")

    visualize_result(
        robot,
        result.path,
        obstacles,
        result.tree_start,
        result.tree_goal,
        collision_checker,
        start_region=(start_pos, 0.2),
        goal_region=(goal_pos, 0.2),
        title="Example 2: Start and Goal TSRs",
        filename="example2_result.png",
    )


def example_constrained():
    """Example 3: Constrained planning with trajectory-wide constraint TSR.

    The end-effector must stay within a horizontal band (y constraint)
    throughout the entire trajectory.
    """
    print("\n" + "=" * 60)
    print("Example 3: Constrained Planning (y-band constraint)")
    print("=" * 60)

    robot = PlanarArm()
    # No obstacles for this example - the constraint is the challenge
    obstacles = []
    collision_checker = CircleObstacleChecker(robot, obstacles)
    ik = PlanarIK(robot)

    config = CBiRRTConfig(
        step_size=0.2,
        goal_sample_probability=0.1,
        smooth_path=True,
        smoothing_iterations=50,
        membership_tolerance=0.05,
        connection_tolerance=0.05,
        continuous_joints=(True, True),  # both joints turn freely; ±π is only how their angles are written
    )
    planner = CBiRRT(
        robot=robot,
        ik=ik,
        collision_checker=collision_checker,
        config=config,
    )

    # Constraint: end-effector must stay in y-band [0.8, 1.2]
    y_min, y_max = 0.8, 1.2
    constraint_tsr = make_y_constraint_tsr(y_min, y_max)

    # Start: left side of the band
    start_pos = np.array([-1.5, 1.0])
    start_tsr = make_position_tsr(start_pos[0], start_pos[1], tolerance=0.1)

    # Goal: right side of the band
    goal_pos = np.array([1.5, 1.0])
    goal_tsr = make_position_tsr(goal_pos[0], goal_pos[1], tolerance=0.1)

    print(f"  Start region: ({start_pos[0]}, {start_pos[1]}) +/- 0.1")
    print(f"  Goal region: ({goal_pos[0]}, {goal_pos[1]}) +/- 0.1")
    print(f"  Constraint: y in [{y_min}, {y_max}] (horizontal band)")

    result = planner.plan(
        start=None,
        goal_tsrs=[goal_tsr],
        start_tsrs=[start_tsr],
        constraint_tsrs=[constraint_tsr],
        seed=42,
        return_details=True,
    )

    print(f"  Iterations: {result.iterations}")
    print(f"  Start tree nodes: {len(result.tree_start)}")
    print(f"  Goal tree nodes: {len(result.tree_goal)}")

    if not result.success:
        print("No path found!")
        print("  (Constrained planning is harder - the path must stay on the constraint manifold)")
        return

    print(f"Found path with {len(result.path)} waypoints")

    # Verify constraint satisfaction along path
    max_violation = 0.0
    for q in result.path:
        pose = robot.forward_kinematics(q)
        y = pose[1, 3]
        if y < y_min:
            max_violation = max(max_violation, y_min - y)
        elif y > y_max:
            max_violation = max(max_violation, y - y_max)

    print(f"  Max constraint violation: {max_violation:.4f}")

    start_pose = robot.forward_kinematics(result.path[0])
    final_pose = robot.forward_kinematics(result.path[-1])
    print(f"  Start position: ({start_pose[0, 3]:.3f}, {start_pose[1, 3]:.3f})")
    print(f"  Final position: ({final_pose[0, 3]:.3f}, {final_pose[1, 3]:.3f})")

    visualize_result(
        robot,
        result.path,
        obstacles,
        result.tree_start,
        result.tree_goal,
        collision_checker,
        start_region=(start_pos, 0.1),
        goal_region=(goal_pos, 0.1),
        constraint_region=(-2.5, 2.5, y_min, y_max),
        title="Example 3: Constrained Planning (y-band)",
        filename="example3_result.png",
    )


def example_multiple_configs():
    """Example 4: several start and goal configurations; the planner connects whichever pair it can."""
    print("\n" + "=" * 60)
    print("Example 4: Several start and goal configurations")
    print("=" * 60)

    robot = PlanarArm()
    collision_checker = CircleObstacleChecker(robot, [(np.array([0.5, 1.2]), 0.15)])
    config = CBiRRTConfig(step_size=0.3, continuous_joints=(True, True))  # both joints turn freely
    planner = CBiRRT(robot, PlanarIK(robot), collision_checker, config)

    starts = [np.array([np.pi / 4, 0.0]), np.array([np.pi / 4, 0.2])]
    goals = [np.array([np.pi / 2, 0.0]), np.array([np.pi / 2, np.pi / 4])]
    result = planner.plan(start=starts, goal=goals, seed=42, return_details=True)
    if not result.success:
        print(f"Planning failed: {result.failure_reason}")
        return
    print(f"Found a path with {len(result.path)} waypoints in {result.iterations} iterations")
    print(f"  Start tree: {len(result.tree_start)} nodes from {result.tree_start.num_roots} roots")
    print(f"  Goal tree:  {len(result.tree_goal)} nodes from {result.tree_goal.num_roots} roots")
    # The result records which member each end came from (matching endpoints by value would break on continuous
    # joints, where an endpoint may be re-expressed by a multiple of 2*pi).
    print(f"  Connected start {result.start_index + 1} to goal {result.goal_index + 1}")


def main():
    parser = argparse.ArgumentParser(description="CBiRRT planar arm examples")
    parser.add_argument(
        "--example",
        "-e",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run one example (1=basic, 2=start/goal TSRs, 3=constrained, 4=several starts and goals)",
    )
    args = parser.parse_args()

    if args.example == 1:
        example_basic()
    elif args.example == 2:
        example_start_goal_tsrs()
    elif args.example == 3:
        example_constrained()
    elif args.example == 4:
        example_multiple_configs()
    else:
        # Run all examples
        example_basic()
        example_start_goal_tsrs()
        example_constrained()
        example_multiple_configs()


if __name__ == "__main__":
    main()
