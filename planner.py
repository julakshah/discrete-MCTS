from abc import ABC, abstractmethod
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from astar import COSTS, astar, dynamic_astar, heuristic


class Planner(ABC):
    def __init__(self, grid):
        """Set up the grid, history, and totals shared by planners."""
        self.grid = grid
        self.history = [grid.actor_pos]
        self.frames = [(grid.obstacle_matrix(), grid.actor_pos)]

        self.time_planning = 0
        self.dist_traveled = 0
        self.nodes_visited = 0
        self.plot_label = "Planner"

    @abstractmethod
    def step(self):
        """Take a planner step and return its outcome and cumulative metrics."""
        raise NotImplementedError("This planner needs a step method")

    def add_metrics(self, result):
        """Add distance, planning time, and nodes visited to a step result."""
        # Keep track of totals across all planning steps
        result["dist_traveled"] = self.dist_traveled
        result["time_planning"] = self.time_planning
        result["nodes_visited"] = self.nodes_visited

        return result

    def init_plot(self):
        """Set up a live plot of the grid and actor's path."""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(9, 9))
        self.ax.set_xlim(-0.5, self.grid.width - 0.5)
        self.ax.set_ylim(self.grid.height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)

        self.ax.plot(*self.grid.start, "go", markersize=8, label="Start")
        self.ax.plot(*self.grid.goal, "r*", markersize=12, label="Goal")
        (self.path_line,) = self.ax.plot([], [], "b-", linewidth=2, label="Path")
        (self.actor_marker,) = self.ax.plot([], [], "bo", markersize=8, label="Actor")
        self.obstacle_image = self.ax.imshow(
            self.grid.obstacle_matrix(),
            cmap="Greys",
            vmin=0,
            vmax=1,
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )

        self.ax.legend()
        self.fig.show()
        self.update_plot()

    def update_plot(self):
        """Update the live plot with the current actor and obstacles."""
        if not hasattr(self, "fig"):
            return

        path = np.asarray(self.history)
        self.path_line.set_data(path[:, 0], path[:, 1])
        x, y = self.grid.actor_pos
        self.actor_marker.set_data([x], [y])

        # for obstacle in self.grid.obstacles:
        #     rectangle = Rectangle(
        #         (obstacle.x - 0.5, obstacle.y - 0.5),
        #         obstacle.width,
        #         obstacle.height,
        #         facecolor="black",
        #         alpha=0.6,
        #         edgecolor="black",
        #     )
        #     self.ax.add_patch(rectangle)
        self.obstacle_image.set_data(self.grid.obstacle_matrix())

        self.ax.set_title(
            f"{self.plot_label} Navigation | Step {len(self.history) - 1} "
            f"| Distance {self.dist_traveled:.1f} | Planning: {self.time_planning:.1f}s"
        )
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close_plot(self):
        """Stop live plotting and show the final figure."""
        plt.ioff()
        plt.show()

    def plot_path(self, title="Actor path"):
        """Replay the saved actor path and obstacle positions."""
        fig, ax = plt.subplots(figsize=(8, 8))
        obstacle_map, actor_pos = self.frames[0]
        image = ax.imshow(obstacle_map, cmap="Greys", vmin=0, vmax=1, origin="lower")
        actor_marker = ax.scatter(*actor_pos, c="blue", label="Actor", zorder=3)
        ax.scatter(*self.grid.goal, c="red", label="Goal", zorder=3)
        ax.scatter(*self.grid.start, c="green", label="Start", zorder=3)
        (path_line,) = ax.plot([], [], "b-", alpha=0.6, label="Actor path")
        ax.set_xlim(-0.5, self.grid.width - 0.5)
        ax.set_ylim(-0.5, self.grid.height - 0.5)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()

        for frame_index, (obstacle_map, actor_pos) in enumerate(self.frames):
            image.set_data(obstacle_map)
            actor_marker.set_offsets([actor_pos])
            path = np.asarray(self.history[: frame_index + 1])
            path_line.set_data(path[:, 0], path[:, 1])
            ax.set_title(f"{title} - step {frame_index}")
            plt.pause(0.15)

        print(f"Time spent planning: {self.time_planning}")
        plt.show(block=True)


class AStar(Planner):
    def __init__(
        self,
        grid,
        heuristic=heuristic,
        dynamic=False,
        verbose=True,
        replan_every=1,
        density_constant=50.0,
    ):
        """Set up A*, with optional density forecasts and replanning interval.

        Set replan_every to None to plan only once.
        """
        if replan_every is not None:
            if type(replan_every) is not int or replan_every < 1:
                raise ValueError("replan_every must be a positive integer or None")

        if dynamic and not grid.track_density:
            raise ValueError("Dynamic A* needs a grid with track_density=True")

        super().__init__(grid)
        self.plot_label = "A*"
        self.heuristic = heuristic
        self.dynamic = dynamic
        self.verbose = verbose
        self.replan_every = replan_every
        self.density_constant = density_constant

        self.path = None
        self.has_planned = False
        self.next_step = 1
        self.steps_since_plan = 0
        self.result = None

    def step(self):
        """Plan when needed, then move or wait one step and return the result."""
        if self.result is not None:
            return self.result

        # Plan once if replan_every is None, otherwise plan every N moves
        should_replan = (
            self.replan_every is not None and self.steps_since_plan >= self.replan_every
        )

        if not self.has_planned or should_replan:
            t0 = time.perf_counter()

            if self.dynamic:
                self.path, nodes = dynamic_astar(
                    self.grid,
                    self.grid.actor_pos,
                    self.grid.goal,
                    self.heuristic,
                    self.density_constant,
                )
            else:
                self.path, nodes = astar(
                    self.grid, self.grid.actor_pos, self.grid.goal, self.heuristic
                )

            self.time_planning += time.perf_counter() - t0
            self.nodes_visited += nodes
            self.has_planned = True
            self.next_step = 1
            self.steps_since_plan = 0

        if self.path is None:
            # Let obstacles move, then try planning again on the next step
            self.has_planned = False
            result = self.grid.step()
            self.history.append(self.grid.actor_pos)
            self.frames.append((self.grid.obstacle_matrix(), self.grid.actor_pos))
            self.update_plot()
            result = self.add_metrics(result)

            if result["collision"] or result["reached_goal"]:
                self.result = result
            return result

        if self.next_step == len(self.path):
            self.result = self.add_metrics({"collision": False, "reached_goal": True})
            return self.result

        # Move along the current path and record the result
        current = self.grid.actor_pos
        next_pos = self.path[self.next_step]
        move = (next_pos[0] - current[0], next_pos[1] - current[1])
        action = self.grid.DIRECTION_LABELS[move]
        result = self.grid.step(actor_action=action)

        if self.grid.actor_pos != current:
            self.dist_traveled += COSTS[move]

        self.history.append(self.grid.actor_pos)
        self.frames.append((self.grid.obstacle_matrix(), self.grid.actor_pos))
        self.update_plot()
        result = self.add_metrics(result)

        if self.verbose:
            print(f"Following planned move: {move}, action {action}")

        if result["collision"] or result["reached_goal"]:
            self.result = result
        else:
            self.next_step += 1
            self.steps_since_plan += 1
        return result


if __name__ == "__main__":
    from grid import (
        static_sparse_grid,
        dynamic_sparse_grid,
        dynamic_dense_grid,
        dynamic_wall_maze_grid,
    )

    # Choose: static_sparse, dynamic_sparse, dynamic_dense, dynamic_wall_maze
    scenario = "dynamic_wall_maze"
    width = 100
    height = 100
    seed = 0
    start_clearance = 5
    max_steps = 500

    scenarios = {
        "static_sparse": static_sparse_grid,
        "dynamic_sparse": dynamic_sparse_grid,
        "dynamic_dense": dynamic_dense_grid,
        "dynamic_wall_maze": dynamic_wall_maze_grid,
    }

    grid = scenarios[scenario](
        width=width, height=height, seed=seed, start_clearance=start_clearance
    )
    planner = AStar(grid, dynamic=True, replan_every=5, density_constant=50.0)

    planner.init_plot()
    try:
        for step in range(max_steps):
            result = planner.step()

            if result["reached_goal"]:
                print("Robot reached the goal!")
                break
            if result["collision"]:
                print("Robot collided with an obstacle.")
                break
        else:
            print("Stopped at the step limit.")
    finally:
        planner.close_plot()
