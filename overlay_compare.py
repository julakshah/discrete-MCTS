"""Animate four planners on the same grid.py-generated scenario.

Choose SCENARIO below from the factory functions already defined in grid.py.
The grid is run independently for each planner, but every planner receives
the same initial obstacle layout, actor position, goal, and Grid RNG state.
This is a visualization tool, not a strict speed benchmark.

Run from this directory with:

    python overlay_compare.py

Set SAVE_FIGURE to a filename if you also want a saved image.
"""

from contextlib import redirect_stdout
import io
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from emm import Expectimax
from grid import (
    Grid,
    dynamic_dense_grid,
    dynamic_sparse_grid,
    dynamic_wall_maze_grid,
    dynamic_wall_maze_one_section_grid,
    dynamic_wall_maze_two_section_grid,
    static_sparse_grid,
)
from mcts import ACTION_COSTS as MCTS_ACTION_COSTS
from mcts import DIRECTIONS as MCTS_DIRECTIONS
from mcts import mcts_closed_loop
from planner import AStar
from rtdpNEW import LRTDP


# ---------------------------------------------------------------------------
# Choose one of the existing grid.py scenarios. The default matches the
# dynamic wall maze used by the other planner entry points.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "static_sparse": static_sparse_grid,
    "dynamic_sparse": dynamic_sparse_grid,
    "dynamic_dense": dynamic_dense_grid,
    "dynamic_wall_maze": dynamic_wall_maze_grid,
    "dynamic_wall_maze_one_section": dynamic_wall_maze_one_section_grid,
    "dynamic_wall_maze_two_section": dynamic_wall_maze_two_section_grid,
}
SCENARIO = "dynamic_dense"
GRID_WIDTH = 30
GRID_HEIGHT =30
GRID_SEED = 12
START_CLEARANCE = 5

# Planner settings. Increase MCTS_MAX_ITERS for a stronger but slower MCTS.
# MCTS is substantially slower than the other implementations in this repo;
# these defaults keep the overlay useful as an interactive diagnostic.
MAX_STEPS = 100
MCTS_MAX_ITERS = 500
MCTS_ROLLOUT_HORIZON = 10
EXPECTIMAX_DEPTH = 8
COLLISION_PENALTY = 100.0
ANIMATION_DELAY = 0.15

# Static A* is appropriate for the static factory scenario; dynamic A* uses
# the density forecast for the other grid.py scenarios.
DYNAMIC = SCENARIO != "static_sparse"

# Set to None to only display the plot. Example: "overlay_results.png".
SAVE_FIGURE = None


class CountingGrid(Grid):
    """Grid variant that distinguishes real steps from planning simulations."""

    def step(self, actor_action=None):
        if self._counting_simulation:
            self._step_counts["simulated"] += 1
        else:
            self._step_counts["real"] += 1
        return super().step(actor_action=actor_action)

    def clone(self, include_density=True):
        cloned = super().clone(include_density=include_density)
        # All clones share the planner's counter, and every clone is a
        # simulated environment. The live planner grid is marked real in
        # make_entry().
        cloned._counting_simulation = True
        return cloned


def make_grid():
    """Build the selected scenario using the existing grid.py factory."""
    if SCENARIO not in SCENARIOS:
        raise ValueError(f"Unknown SCENARIO {SCENARIO!r}")

    grid = SCENARIOS[SCENARIO](
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        seed=GRID_SEED,
        start_clearance=START_CLEARANCE,
    )
    # The factory creates a normal Grid. It is safe to use the subclass here:
    # both classes have the same ordinary Python object layout.
    grid.__class__ = CountingGrid
    grid._step_counts = {"real": 0, "simulated": 0}
    grid._counting_simulation = False
    return grid


def seed_process_randomness(seed=1234):
    """Make MCTS's module-level random calls repeatable per planner run."""
    random.seed(seed)
    np.random.seed(seed)


PLANNER_NAMES = ("LRTDP", "MCTS", "Expectimax", "A*")
COLORS = {
    "LRTDP": "tab:blue",
    "MCTS": "tab:orange",
    "Expectimax": "tab:green",
    "A*": "tab:red",
}


def make_standard_planner(name, grid):
    """Construct one of the Planner-subclass implementations."""
    if name == "A*":
        planner = AStar(
            grid,
            dynamic=DYNAMIC,
            density_constant=COLLISION_PENALTY,
            replan_every=1,
            verbose=False,
        )
    elif name == "Expectimax":
        planner = Expectimax(
            grid,
            depth=EXPECTIMAX_DEPTH,
            collision_penalty=COLLISION_PENALTY,
            replan_every=1,
            verbose=False,
        )
    elif name == "LRTDP":
        planner = LRTDP(
            grid,
            max_trials=10,
            max_trial_steps=25,
            transition_samples=3,
            verbose=False,
        )
    else:
        raise ValueError(f"Unknown standard planner: {name}")
    return planner

def make_entry(name, base_grid):
    """Create one independent planner/environment pair."""
    grid = base_grid.clone()
    grid._step_counts = {"real": 0, "simulated": 0}
    grid._counting_simulation = False
    seed_process_randomness(1234)
    return {
        "name": name,
        "grid": grid,
        "planner": None if name == "MCTS" else make_standard_planner(name, grid),
        "history": [grid.actor_pos],
        "status": "running",
        "distance": 0.0,
        "planning_time": 0.0,
        "nodes": 0,
        "real_steps": 0,
        "simulated_steps": 0,
    }


def mark_result(entry, result):
    """Update terminal status after one real environment step."""
    if result.get("reached_goal"):
        entry["status"] = "success"
    elif result.get("collision"):
        entry["status"] = "collision"
    elif result.get("no_path"):
        entry["status"] = "no path"


def advance_entry(entry):
    """Advance one planner by exactly one real simulation step."""
    if entry["status"] != "running":
        return

    grid = entry["grid"]
    if entry["name"] != "MCTS":
        result = entry["planner"].step()
        entry["history"] = list(entry["planner"].history)
        entry["distance"] = result.get(
            "dist_traveled", entry["planner"].dist_traveled
        )
        entry["planning_time"] = result.get(
            "time_planning", entry["planner"].time_planning
        )
        entry["nodes"] = result.get(
            "nodes_visited", entry["planner"].nodes_visited
        )
        entry["real_steps"] = entry["grid"]._step_counts["real"]
        entry["simulated_steps"] = entry["grid"]._step_counts["simulated"]
        mark_result(entry, result)
        return

    before = grid.actor_pos
    start_time = time.perf_counter()
    try:
        # The current MCTS implementation prints root statistics every call.
        # Suppress that output so the live animation remains readable.
        with redirect_stdout(io.StringIO()):
            action = mcts_closed_loop(
                grid.actor_pos,
                grid.goal,
                grid,
                c=0.2,
                max_iters=MCTS_MAX_ITERS,
                rollout_horizon=MCTS_ROLLOUT_HORIZON,
                rollout_policy="safe_random",
                reward_heuristic="legacy_progress",
                verbose=False,
            )
    except Exception as exc:
        entry["status"] = f"error: {type(exc).__name__}: {exc}"
        entry["planning_time"] += time.perf_counter() - start_time
        return

    entry["planning_time"] += time.perf_counter() - start_time
    entry["nodes"] += MCTS_MAX_ITERS
    result = grid.step(actor_action=int(action))

    if grid.actor_pos != before:
        dx = grid.actor_pos[0] - before[0]
        dy = grid.actor_pos[1] - before[1]
        action_index = int(
            np.flatnonzero(np.all(MCTS_DIRECTIONS == (dx, dy), axis=1))[0]
        )
        entry["distance"] += MCTS_ACTION_COSTS[action_index]
    entry["history"].append(grid.actor_pos)
    entry["real_steps"] = grid._step_counts["real"]
    entry["simulated_steps"] = grid._step_counts["simulated"]
    mark_result(entry, result)


def result_from_entry(entry):
    """Return the serializable result shape used by the final summary."""
    return {
        "name": entry["name"],
        "history": entry["history"],
        "status": entry["status"] if entry["status"] != "running" else "step limit",
        "distance": entry["distance"],
        "planning_time": entry["planning_time"],
        "nodes": entry["nodes"],
        "real_steps": entry["real_steps"],
        "simulated_steps": entry["simulated_steps"],
    }


def make_live_plot(base_grid, entries):
    """Create artists that are updated after every synchronized step."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    obstacle_image = ax.imshow(
        entries[0]["grid"].obstacle_matrix(),
        cmap="Greys",
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
    )
    lines = {}
    markers = {}
    for entry in entries:
        line, = ax.plot([], [], color=COLORS[entry["name"]], linewidth=2,
                         marker=".", markersize=4, label=entry["name"])
        marker, = ax.plot([], [], color=COLORS[entry["name"]], marker="o",
                          markersize=7, linestyle="None")
        lines[entry["name"]] = line
        markers[entry["name"]] = marker

    ax.scatter(*base_grid.start, color="black", marker="o", s=80, label="Start")
    ax.scatter(*base_grid.goal, color="gold", edgecolor="black", marker="*", s=150, label="Goal")
    ax.set_xlim(-0.5, base_grid.width - 0.5)
    ax.set_ylim(base_grid.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(base_grid.width))
    ax.set_yticks(range(base_grid.height))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    return fig, ax, obstacle_image, lines, markers


def update_live_plot(fig, ax, obstacle_image, lines, markers, entries, step):
    """Refresh the animation after all planners have taken one step."""
    active = next((entry for entry in entries if entry["status"] == "running"), entries[0])
    obstacle_image.set_data(active["grid"].obstacle_matrix())
    statuses = []
    for entry in entries:
        path = np.asarray(entry["history"])
        lines[entry["name"]].set_data(path[:, 0], path[:, 1])
        current_x, current_y = entry["grid"].actor_pos
        markers[entry["name"]].set_data([current_x], [current_y])
        status = entry["status"] if entry["status"] != "running" else "running"
        statuses.append(f"{entry['name']}: {status}")
    ax.set_title(f"{SCENARIO} | step {step} | " + " | ".join(statuses))
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(ANIMATION_DELAY)


def run_all(animate=True):
    """Run all planners in lockstep, optionally updating the live plot."""
    base_grid = make_grid()
    entries = [make_entry(name, base_grid) for name in PLANNER_NAMES]
    live_artists = make_live_plot(base_grid, entries) if animate else None

    for step in range(1, MAX_STEPS + 1):
        for entry in entries:
            advance_entry(entry)
        if live_artists is not None:
            update_live_plot(*live_artists, entries, step)
        if all(entry["status"] != "running" for entry in entries):
            break

    results = [result_from_entry(entry) for entry in entries]
    if live_artists is not None and SAVE_FIGURE:
        live_artists[0].savefig(SAVE_FIGURE, dpi=160, bbox_inches="tight")
        print(f"Saved figure to {SAVE_FIGURE}")
    return base_grid, results


def plot_results(base_grid, results):
    """Plot the initial obstacles and all actor trajectories."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(
        base_grid.obstacle_matrix(),
        cmap="Greys",
        vmin=0,
        vmax=1,
        origin="upper",
        interpolation="nearest",
    )

    colors = {
        "LRTDP": "tab:blue",
        "MCTS": "tab:orange",
        "Expectimax": "tab:green",
        "A*": "tab:red",
    }

    for result in results:
        path = np.asarray(result["history"])
        label = f"{result['name']} ({result['status']})"
        ax.plot(
            path[:, 0],
            path[:, 1],
            color=colors[result["name"]],
            linewidth=2,
            marker=".",
            markersize=4,
            label=label,
        )

    ax.scatter(*base_grid.start, color="black", marker="o", s=80, label="Start")
    ax.scatter(*base_grid.goal, color="gold", edgecolor="black", marker="*", s=150, label="Goal")
    ax.set_xlim(-0.5, base_grid.width - 0.5)
    ax.set_ylim(base_grid.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(base_grid.width))
    ax.set_yticks(range(base_grid.height))
    ax.grid(True, alpha=0.2)
    ax.set_title("Planner trajectories from the same initial grid")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(SAVE_FIGURE, dpi=160, bbox_inches="tight")
        print(f"Saved figure to {SAVE_FIGURE}")

    return fig


if __name__ == "__main__":
    base_grid, results = run_all()
    print(f"Grid: {base_grid.width} x {base_grid.height}")
    for result in results:
        print(
            f"{result['name']:10s} | {result['status']:12s} | "
            f"steps={len(result['history']) - 1:3d} | "
            f"distance={result['distance']:.2f} | "
            f"planning={result['planning_time']:.3f}s | "
            f"nodes/work={result['nodes']} | "
            f"real_env_steps={result['real_steps']} | "
            f"simulated_env_steps={result['simulated_steps']}"
        )

    # run_all(animate=True) already leaves the live figure on screen.
    plt.ioff()
    plt.show(block=True)
