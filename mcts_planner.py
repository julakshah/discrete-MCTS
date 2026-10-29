
import cProfile
import pstats
import time

import numpy as np

from planner import Planner
from mcts import mcts_closed_loop, mcts_open_loop
from grid import dynamic_sparse_grid, dynamic_dense_grid, static_sparse_grid, dynamic_wall_maze_grid

class MCTSPlanner(Planner):
    def __init__(
        self,
        grid,
        open_loop=False,
        max_iters=1000,
        rollout_horizon=50,
        timeout=None,
        c=0.1,
        profile=False,
        verbose=True,
        policy="safe_random",
        heuristic="legacy_progress",
        eps=0.2,
        plot=True,
    ):
        super().__init__(grid)

        self.plot_label = "MCTS"

        self.open_loop = open_loop
        self.max_iters = max_iters
        self.rollout_horizon = rollout_horizon
        self.timeout = timeout
        self.c = c
        self.profile = profile
        self.verbose = verbose
        self.policy = policy
        self.heuristic = heuristic
        self.eps = eps
        self.plot = True
        self.timeout = timeout

        self.result = None

    def step(self):
        """Run MCTS, execute one action, and return cumulative metrics."""
        if self.result is not None:
            return self.result

        # Select the MCTS implementation
        mcts_fn = (
            mcts_open_loop if self.open_loop
            else mcts_closed_loop
        )

        # Plan the next action
        t0 = time.perf_counter()

        if self.profile:
            with cProfile.Profile() as profiler:
                action = mcts_fn(
                    self.grid.actor_pos,
                    self.grid.goal,
                    self.grid,
                    c=self.c,
                    max_iters=self.max_iters,
                    rollout_horizon=self.rollout_horizon,
                    rollout_policy=self.policy,
                    #rollout_policy="epsilon_greedy",
                    epsilon=self.eps,
                    #reward_heuristic="absorbing_failure",
                    #reward_heuristic="progress",
                    reward_heuristic=self.heuristic,
                    collision_penalty=20.0,
                    failure_cost_per_step=20.0,
                    verbose=True,
                    timeout=self.timeout
                )
        else:
            action = mcts_fn(
                self.grid.actor_pos,
                self.grid.goal,
                self.grid,
                c=self.c,
                max_iters=self.max_iters,
                rollout_horizon=self.rollout_horizon,
                rollout_policy=self.policy,
                #rollout_policy="epsilon_greedy",
                epsilon=self.eps,
                #reward_heuristic="absorbing_failure",
                #reward_heuristic="progress",
                reward_heuristic=self.heuristic,
                collision_penalty=20.0,
                failure_cost_per_step=20.0,
                verbose=True,
                timeout=self.timeout
            )

        self.time_planning += time.perf_counter() - t0
        if self.verbose:
            print(f"Action: {action}")

        if self.profile:
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumtime").print_stats(20)

        # Count search iterations as the MCTS work metric
        # Assumes MCTS completes max_iters iterations.
        if self.max_iters is not None:
            self.nodes_visited += self.max_iters

        # Handle a planner that cannot select an action
        if action is None:
            self.result = self.add_metrics({
                "collision": False,
                "reached_goal": False,
                "no_path": True,
            })
            return self.result

        action = int(action)

        # Execute the chosen action
        previous_pos = self.grid.actor_pos
        result = self.grid.step(actor_action=action)

        # Record actual distance traveled
        current_pos = self.grid.actor_pos
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        self.dist_traveled += np.hypot(dx, dy)

        # Record state for visualization and replay
        self.history.append(current_pos)
        self.frames.append((
            self.grid.obstacle_matrix(),
            current_pos,
        ))
        if self.plot:
            self.update_plot()
        result = self.add_metrics(result)

        if self.verbose:
            print(f"MCTS action: {action}")

        # Cache terminal result
        if (
            result["collision"]
            or result["reached_goal"]
            or result.get("no_path")
        ):
            self.result = result

        return result

if __name__ == "__main__":
    grid = dynamic_sparse_grid(
        width=50,
        height=50,
        seed=0,
        start_clearance=5,
    )

    # grid = dynamic_dense_grid(
    #     width=50,
    #     height=50,
    #     seed=0,
    #     start_clearance=5,
    # )

    # grid = dynamic_wall_maze_grid(
    #     width=100,
    #     height=100,
    #     seed=42,
    #     start_clearance=5,
    # )

    # grid = static_sparse_grid(
    #     width=50,
    #     height=50,
    #     seed=42,
    #     start_clearance=5,
    # )

    grid.track_density = False
    planner = MCTSPlanner(
        grid,
        open_loop=False,
        max_iters=1000,
        rollout_horizon=10,
        c=0.2,
        profile=False,
        verbose=False,
        policy="safe_random",
        heuristic="legacy_progress",
        eps=0.2
    )

    planner.init_plot()

    try:
        for step in range(500):
            result = planner.step()

            if result["reached_goal"]:
                print("Robot reached the goal!")
                break

            if result["collision"]:
                print("Robot collided with an obstacle.")
                break

            if result.get("no_path"):
                print("No path found!")
                break

    finally:
        planner.close_plot()

    print(f"Distance traveled: {result['dist_traveled']:.2f}")
    print(f"Planning time: {result['time_planning']:.3f}s")
    print(f"Search iterations: {result['nodes_visited']}")
