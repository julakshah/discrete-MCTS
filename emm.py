import time

from grid import Grid
from astar import COSTS, heuristic
from planner import Planner

# steal from astar for consistency
MOVES = tuple(move for move in COSTS if move != (0, 0))


def expectimax(
    grid: Grid,
    start,
    end,
    heuristic,
    depth=4,
    collision_penalty=100.0,
):
    """
    Return the path and the number of neighbor checks during the search.

    Depth-limited expectimax over costs. Max nodes are the actor's moves.
    Chance nodes are the obstacles' turn:
    """
    nodes_visited = 0

    prediction_grid = grid.clone()
    prediction_grid.reset_density()
    density_grids = [prediction_grid.density.copy()]
    for _ in range(depth):
        prediction_grid.diffuse_step()
        density_grids.append(prediction_grid.density.copy())

    memo = {}

    def expected_cost(node, remaining):
        """Return (expected cost to go, best next node) from node."""
        nonlocal nodes_visited

        if node == end:
            return 0.0, None
        if remaining == 0:
            return heuristic(node, end), None

        key = (node, remaining)
        if key in memo:
            return memo[key]

        timestep = depth - remaining + 1
        occupancy = density_grids[timestep]

        if remaining == depth:
            candidates = grid.neighbors(node)
        else:
            candidates = [
                (node[0] + dx, node[1] + dy)
                for dx, dy in MOVES
                if grid.in_bounds((node[0] + dx, node[1] + dy))
            ]

        best_cost = float("inf")
        best_next = None

        for neighbor in candidates:
            nodes_visited += 1

            # chance node calculations
            p_hit = occupancy[neighbor[1], neighbor[0]]
            if p_hit >= 1.0:
                continue

            dx, dy = neighbor[0] - node[0], neighbor[1] - node[1]
            dg = COSTS[(dx, dy)]

            future, _ = expected_cost(neighbor, remaining - 1)
            hit_cost = collision_penalty + heuristic(neighbor, end)
            cost = dg + (1 - p_hit) * future + p_hit * hit_cost

            if cost < best_cost:
                best_cost = cost
                best_next = neighbor

        memo[key] = (best_cost, best_next)
        return memo[key]

    cost, _ = expected_cost(start, depth)
    if cost == float("inf"):
        return None, nodes_visited

    path = [start]
    node, remaining = start, depth
    while node != end and remaining > 0:
        _, node = memo[(node, remaining)]
        remaining -= 1
        path.append(node)

    if len(path) < 2:
        return None, nodes_visited

    return path, nodes_visited


class Expectimax(Planner):
    def __init__(
        self,
        grid,
        heuristic=heuristic,
        depth=4,
        collision_penalty=100.0,
        verbose=True,
        replan_every=1,
    ):
        """Set up expectimax with a lookahead depth and replanning interval.

        Replans when the path runs out too, so replan_every may exceed depth.
        """
        if type(depth) is not int or depth < 1:  # to prevent hang when depth = 0
            raise ValueError("depth must be a positive integer")
        if not grid.track_density:  # needs a grid with track density like astar
            raise ValueError("Expectimax needs a grid with track_density=True")

        super().__init__(grid)
        self.plot_label = "Expectimax"
        self.heuristic = heuristic
        self.depth = depth
        self.collision_penalty = collision_penalty
        self.verbose = verbose
        self.replan_every = replan_every

        self.path = None
        self.next_step = 1
        self.steps_since_plan = 0
        self.result = None

    def _finish_step(self, result):
        self.history.append(self.grid.actor_pos)
        self.frames.append((self.grid.obstacle_matrix(), self.grid.actor_pos))
        self.update_plot()
        result = self.add_metrics(result)
        if result["collision"] or result["reached_goal"]:
            self.result = result
        return result

    def step(self):
        if self.result is not None:
            return self.result

        if self.grid.actor_pos == self.grid.goal:
            self.result = self.add_metrics({"collision": False, "reached_goal": True})
            return self.result

        path_used_up = self.path is None or self.next_step >= len(self.path)
        if path_used_up or self.steps_since_plan >= self.replan_every:
            t0 = time.perf_counter()
            self.path, nodes = expectimax(
                self.grid,
                self.grid.actor_pos,
                self.grid.goal,
                self.heuristic,
                self.depth,
                self.collision_penalty,
            )
            self.time_planning += time.perf_counter() - t0
            self.nodes_visited += nodes
            self.next_step = 1
            self.steps_since_plan = 0

        if self.path is None:
            return self._finish_step(self.grid.step())

        current = self.grid.actor_pos
        next_pos = self.path[self.next_step]
        move = (next_pos[0] - current[0], next_pos[1] - current[1])
        action = self.grid.DIRECTION_LABELS[move]
        result = self.grid.step(actor_action=action)

        if self.grid.actor_pos != current:
            self.dist_traveled += COSTS[move]

        if self.verbose:
            print(f"planned move: {move}, action {action}")

        result = self._finish_step(result)
        if not self.result:
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

    # static_sparse, dynamic_sparse, dynamic_dense, dynamic_wall_maze
    scenario = "static_sparse"
    width = 100
    height = 100
    seed = 0
    start_clearance = 10
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
    planner = Expectimax(
        grid, depth=30, collision_penalty=100.0, replan_every=2, verbose=False
    )

    planner.init_plot()
    try:
        for step in range(max_steps):
            result = planner.step()

            if result["reached_goal"]:
                break
            if result["collision"]:
                break
        else:
            print("Stopped at the step limit")
    finally:
        planner.close_plot()
