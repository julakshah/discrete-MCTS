from math import dist
from statistics import mean

from astar import astar, heuristic
from planner import AStar, Planner
from grid import (
    static_sparse_grid,
    dynamic_sparse_grid,
    dynamic_dense_grid,
    dynamic_wall_maze_grid,
)

# Benchmark settings
SEEDS = [0, 1, 2, 3, 4]
MAX_STEPS = 500


def static_reference_path(grid):
    """Find the shortest path through the grid's stationary obstacles."""
    # Keep the walls, but remove the moving obstacles from a copy of the grid
    reference_grid = grid.clone()
    static_obstacles = []

    for obstacle in reference_grid.obstacles:
        if obstacle.stay_prob == 1:
            static_obstacles.append(obstacle)

    reference_grid.obstacles = static_obstacles

    path, _ = astar(
        reference_grid, reference_grid.start, reference_grid.goal, heuristic
    )
    return path


def path_progress(path, position):
    """Find the percent along the path at the node closest to the position.

    Return None if there is no reference path. If nodes are equally close,
    use the one that comes first along the path.
    """
    if path is None:
        return None

    if len(path) == 1:
        return 100.0

    # Find the closest point on the path to the actor
    closest = 0
    for i in range(1, len(path)):
        if dist(position, path[i]) < dist(position, path[closest]):
            closest = i

    total_distance = 0
    distance_to_closest = 0

    for i in range(1, len(path)):
        step_distance = dist(path[i - 1], path[i])
        total_distance += step_distance

        if i <= closest:
            distance_to_closest += step_distance

    return 100 * distance_to_closest / total_distance


def path_distance(path):
    """Return the total distance along a path, or None if there is no path."""
    if path is None:
        return None

    total = 0
    for i in range(1, len(path)):
        total += dist(path[i - 1], path[i])
    return total


def benchmark(
    planner_class,
    scenario_factory,
    seeds=SEEDS,
    max_steps=MAX_STEPS,
    planner_kwargs=None,
):
    """Run a planner on each seed and return the results and averages.

    scenario_factory takes a seed and returns a new grid. Each run counts
    planner.step() calls, including calls that end without moving. Averages
    include all runs; progress only includes runs with a static reference path.
    Nodes per step are normalized by each run's nonzero reference distance.
    Normalized distance and planning time average the per-run ratios for
    successful runs with a nonzero reference distance.
    """
    if not isinstance(planner_class, type) or not issubclass(planner_class, Planner):
        raise TypeError("planner_class must be a Planner subclass")

    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")

    if seeds is None:
        seeds = SEEDS
    seeds = list(seeds)

    if not seeds:
        raise ValueError("At least one seed is required")

    if planner_kwargs is None:
        planner_kwargs = {}

    runs = []
    for seed in seeds:
        grid = scenario_factory(seed)
        reference_path = static_reference_path(grid)
        reference_distance = path_distance(reference_path)
        planner = planner_class(grid, **planner_kwargs)
        outcome = "step limit"

        for step in range(1, max_steps + 1):
            result = planner.step()

            if result.get("collision"):
                outcome = "collision"
                break

            if result.get("reached_goal"):
                outcome = "success"
                break

            if result.get("no_path"):
                outcome = "no path"
                break

        progress = path_progress(reference_path, grid.actor_pos)

        runs.append(
            {
                "seed": seed,
                "outcome": outcome,
                "steps": step,
                "distance": result["dist_traveled"],
                "reference_path_distance": reference_distance,
                "planning_time": result["time_planning"],
                "nodes_visited": result["nodes_visited"],
                # Count every planner step, including one that ends without moving
                "nodes_visited_per_step": result["nodes_visited"] / step,
                "progress_percent": progress,
            }
        )

        message = f"Finished run {len(runs)}/{len(seeds)} (seed {seed}): {outcome}"
        if progress is None:
            message += " (no static reference path)"
        else:
            message += f" ({progress:.1f}% of reference path)"

        print(message, flush=True)

    successes = 0
    distances = []
    planning_times = []
    nodes_per_step = []
    normalized_nodes_per_step = []
    progresses = []
    reference_distances = []
    normalized_distances = []
    planning_times_per_distance = []

    for run in runs:
        if run["outcome"] == "success":
            successes += 1
            if (
                run["reference_path_distance"] is not None
                and run["reference_path_distance"] > 0
            ):
                normalized_distances.append(
                    run["distance"] / run["reference_path_distance"]
                )
                planning_times_per_distance.append(
                    run["planning_time"] / run["reference_path_distance"]
                )

        distances.append(run["distance"])
        planning_times.append(run["planning_time"])
        nodes_per_step.append(run["nodes_visited_per_step"])

        if run["reference_path_distance"] is not None and run["reference_path_distance"] > 0:
            normalized_nodes_per_step.append(
                run["nodes_visited_per_step"] / run["reference_path_distance"]
            )

        if run["progress_percent"] is not None:
            progresses.append(run["progress_percent"])
        if run["reference_path_distance"] is not None:
            reference_distances.append(run["reference_path_distance"])

    # Average each run equally, including runs that don't reach the goal
    summary = {
        "runs": runs,
        "success_rate": successes / len(runs),
        "average_distance": mean(distances),
        "average_planning_time": mean(planning_times),
        "average_nodes_visited_per_step": mean(nodes_per_step),
        "average_nodes_per_step_per_reference_distance": (
            mean(normalized_nodes_per_step) if normalized_nodes_per_step else None
        ),
        "average_progress_percent": mean(progresses) if progresses else None,
        "average_reference_path_distance": (
            mean(reference_distances) if reference_distances else None
        ),
        "average_successful_distance_ratio_to_reference": (
            mean(normalized_distances) if normalized_distances else None
        ),
        "average_successful_planning_time_per_reference_distance": (
            mean(planning_times_per_distance) if planning_times_per_distance else None
        ),
    }

    print(f"Successes / runs = {summary['success_rate']:.1%} ({successes}/{len(runs)})")
    print(f"Average distance (all runs): {summary['average_distance']:.2f}")

    if reference_distances:
        print(
            f"Average reference path distance: {summary['average_reference_path_distance']:.2f} "
            f"({len(reference_distances)}/{len(runs)} runs)"
        )
    else:
        print("Average reference path distance: unavailable (no reference paths)")

    if normalized_distances:
        print(
            f"mean(distance / reference path distance) = "
            f"{summary['average_successful_distance_ratio_to_reference']:.2f} "
            f"({len(normalized_distances)}/{successes} successes)"
        )
        print(
            f"mean(planning time / reference path distance) = "
            f"{summary['average_successful_planning_time_per_reference_distance']:.4f} s/unit "
            f"({len(normalized_distances)}/{successes} successes)"
        )
    else:
        print(
            "mean(distance / reference path distance) = unavailable (no eligible successes)"
        )
        print(
            "mean(planning time / reference path distance) = unavailable (no eligible successes)"
        )

    if progresses:
        print(
            f"mean(reference distance to closest node / reference path distance) = "
            f"{summary['average_progress_percent']:.1f}% "
            f"({len(progresses)}/{len(runs)} runs with a reference path)"
        )
    else:
        print(
            "mean(reference distance to closest node / reference path distance) = unavailable "
            "(no reference paths)"
        )

    if normalized_nodes_per_step:
        print(
            f"mean(nodes visited / planner steps / reference path distance) = "
            f"{summary['average_nodes_per_step_per_reference_distance']:.2f} "
            f"({len(normalized_nodes_per_step)}/{len(runs)} runs)"
        )
    else:
        print("mean(nodes visited / planner steps / reference path distance) = unavailable "
              "(no nonzero reference paths)")
    print(
        f"Average planning time (all runs): {summary['average_planning_time']:.4f} seconds"
    )
    return summary


if __name__ == "__main__":
    scenarios = [
        ("Static sparse", static_sparse_grid),
        ("Dynamic sparse", dynamic_sparse_grid),
        ("Dynamic dense", dynamic_dense_grid),
        ("Dynamic wall maze", dynamic_wall_maze_grid),
    ]

    for name, scenario in scenarios:
        print(f"\n{name}", flush=True)
        benchmark(
            AStar,
            lambda seed: scenario(seed=seed),
            planner_kwargs={"dynamic": True, "replan_every": 5, "verbose": False},
        )
