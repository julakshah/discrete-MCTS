from bfs import bfs
from astar import astar
from mcts import mcts_open_loop, mcts_closed_loop
from grid import Grid, Obstacle
import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle

DIRECTION = np.array([
    (0, -1),   # N
    (1, -1),   # NE
    (1, 0),    # E
    (1, 1),    # SE
    (0, 1),    # S
    (-1, 1),   # SW
    (-1, 0),   # W
    (-1, -1),  # NW
])

ACTION_COSTS = {
    0: 1,
    1: np.sqrt(2),
    2: 1,
    3: np.sqrt(2),
    4: 1,
    5: np.sqrt(2),
    6: 1,
    7: np.sqrt(2) 
}

class ReplanAStar:
    def __init__(self,grid,use_prob_density = 0):
        self.grid = grid
        self.history = [grid.actor_pos]
        self.use_prob_density = use_prob_density
        self.time_planning = 0
    
    def step(self):
        if not self.use_prob_density:
            t0 = time.perf_counter()
            path = astar(
                self.grid.actor_pos, 
                self.grid.goal, 
                self.grid.neighbors, 
                lambda a,b: 1,
                lambda a,b: max(abs(a[0]-b[0]),abs(a[1]-b[1])) # Chebyshev distance (distance where diag movement is same cost as orthogonal)
            )
            self.time_planning += time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            path = astar(
                self.grid.actor_pos, 
                self.grid.goal, 
                self.grid.neighbors, 
                lambda a,b: 1,
                lambda a,b: max(abs(a[0]-b[0]),abs(a[1]-b[1])) # Chebyshev distance (distance where diag movement is same cost as orthogonal)
            )
            self.time_planning += time.perf_counter() - t0

        if path is None:
            return {"collision": False, "reached_goal": False, "no_path": True}

        if len(path) == 1:
            return {"collision": False, "reached_goal": True}

        move = np.subtract(path[1], path[0])
        action = np.where(np.all(DIRECTION == move, axis=1))[0][0]

        res = self.grid.step(actor_action=int(action))
        self.history.append(self.grid.actor_pos)

        print(f"Path: {path[:5]}")
        print(f"Move: {move}, action {action}")

        return res
    
    def plot_path(self):
        path = np.array(self.history)

        plt.figure(figsize=(8, 8))

        plt.plot(path[:, 0], path[:, 1], "b-", label="Actor path")
        plt.scatter(*self.grid.start, c="green", label="Start", zorder=3)
        plt.scatter(*self.grid.goal, c="red", label="Goal", zorder=3)

        plt.xlim(-0.5, self.grid.width - 0.5)
        plt.ylim(-0.5, self.grid.height - 0.5)
        plt.gca().set_aspect("equal")
        plt.grid(True)

        obstacles = np.array(list(self.grid.initial_obstacles))
        if len(obstacles):
            plt.scatter(
                obstacles[:, 0],
                obstacles[:, 1],
                c="black",
                marker="s",
                label="Initial obstacles",
            )
        
        print(f"Time spent planning: {self.time_planning}")
        
        plt.legend()
        plt.show()
    

    def init_plot(self):
        """Initialize the live visualization."""
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(9, 9))

        self.ax.set_xlim(-0.5, self.grid.width - 0.5)
        self.ax.set_ylim(self.grid.height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.set_title("MCTS Navigation")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)

        # Start and goal
        self.ax.plot(*self.grid.start, "go", markersize=8, label="Start")
        self.ax.plot(*self.grid.goal, "r*", markersize=12, label="Goal")

        # Actor path and current position
        (self.path_line,) = self.ax.plot(
            [], [], "b-", linewidth=2, label="Path"
        )
        (self.actor_marker,) = self.ax.plot(
            [], [], "bo", markersize=8, label="Actor"
        )

        # Obstacle patches
        self.obstacle_patches = []

        self.ax.legend()
        self.fig.show()
        self.update_plot()

    def update_plot(self):
        """Update the visualization after an environment step."""
        if not hasattr(self, "fig"):
            return

        path = np.asarray(self.history)

        self.path_line.set_data(path[:, 0], path[:, 1])

        x, y = self.grid.actor_pos
        self.actor_marker.set_data([x], [y])

        # Remove old obstacle patches
        for patch in self.obstacle_patches:
            patch.remove()

        self.obstacle_patches.clear()

        # Draw obstacles in their current positions
        for obs in self.grid.obstacles:
            rect = Rectangle(
                (obs.x - 0.5, obs.y - 0.5),
                obs.width,
                obs.height,
                facecolor="black",
                alpha=0.6,
                edgecolor="black",
            )
            self.ax.add_patch(rect)
            self.obstacle_patches.append(rect)

        self.ax.set_title(
            f"MCTS Navigation | Step {len(self.history) - 1} "
            f"| Planning: {self.time_planning:.1f}s"
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)


    def close_plot(self):
        """Disable interactive mode when the simulation finishes."""
        plt.ioff()
        plt.show()

class MCTSPlanner:
    def __init__(self,grid,open_loop=False,max_iters=None,rollout_horizon=50,timeout=1.0,c=0.1,profile=False):
        self.grid = grid
        self.history = [grid.actor_pos]
        self.time_planning = 0
        self.dist_traveled = 0
        self.c = c

        # Cap on execution time for MCTS 
        self.max_iters = max_iters
        self.rollout_horizon = rollout_horizon
        self.timeout = timeout

        self.open_loop = open_loop
        self.profile = profile

    def step(self):
        t0 = time.perf_counter()
        if self.profile:
            with cProfile.Profile() as profiler:
                if self.open_loop:
                    action = mcts_open_loop(self.grid.actor_pos, self.grid.goal, self.grid, c=self.c, max_iters=self.max_iters, rollout_horizon=self.rollout_horizon)
                else:
                    action = mcts_closed_loop(self.grid.actor_pos, self.grid.goal, self.grid, c=self.c, max_iters=self.max_iters, rollout_horizon=self.rollout_horizon)
        else:
            if self.open_loop:
                action = mcts_open_loop(self.grid.actor_pos, self.grid.goal, self.grid, c=self.c, max_iters=self.max_iters, rollout_horizon=self.rollout_horizon)
            else:
                action = mcts_closed_loop(self.grid.actor_pos, self.grid.goal, self.grid, c=self.c, max_iters=self.max_iters, rollout_horizon=self.rollout_horizon)

        self.time_planning += time.perf_counter() - t0
        self.dist_traveled += ACTION_COSTS[action]

        if self.profile:
            stats = pstats.Stats(profiler)
            stats.sort_stats("cumtime").print_stats(20)

        res = self.grid.step(actor_action=int(action))
        self.history.append(self.grid.actor_pos)

        self.update_plot()

        print(f"Action: {action}")

        return res
    
    def plot_path(self):
        path = np.array(self.history)

        plt.figure(figsize=(8, 8))

        plt.plot(path[:, 0], path[:, 1], "b-", label="Actor path")
        plt.scatter(*self.grid.start, c="green", label="Start", zorder=3)
        plt.scatter(*self.grid.goal, c="red", label="Goal", zorder=3)

        plt.xlim(-0.5, self.grid.width - 0.5)
        plt.ylim(-0.5, self.grid.height - 0.5)
        plt.gca().set_aspect("equal")
        plt.grid(True)

        obstacles = np.array(list(self.grid.initial_obstacles))
        if len(obstacles):
            plt.scatter(
                obstacles[:, 0],
                obstacles[:, 1],
                c="black",
                marker="s",
                label="Initial obstacles",
            )
        
        print(f"Time spent planning: {self.time_planning}")
        
        plt.legend()
        plt.show()
    
    def init_plot(self):
        """Initialize the live visualization."""
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(9, 9))

        self.ax.set_xlim(-0.5, self.grid.width - 0.5)
        self.ax.set_ylim(self.grid.height - 0.5, -0.5)
        self.ax.set_aspect("equal")
        self.ax.set_title("MCTS Navigation")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, alpha=0.3)

        # Start and goal
        self.ax.plot(*self.grid.start, "go", markersize=8, label="Start")
        self.ax.plot(*self.grid.goal, "r*", markersize=12, label="Goal")

        # Actor path and current position
        (self.path_line,) = self.ax.plot(
            [], [], "b-", linewidth=2, label="Path"
        )
        (self.actor_marker,) = self.ax.plot(
            [], [], "bo", markersize=8, label="Actor"
        )

        # Obstacle patches
        self.obstacle_patches = []

        self.ax.legend()
        self.fig.show()
        self.update_plot()

    def update_plot(self):
        """Update the visualization after an environment step."""
        if not hasattr(self, "fig"):
            return

        path = np.asarray(self.history)

        self.path_line.set_data(path[:, 0], path[:, 1])

        x, y = self.grid.actor_pos
        self.actor_marker.set_data([x], [y])

        # Remove old obstacle patches
        for patch in self.obstacle_patches:
            patch.remove()

        self.obstacle_patches.clear()

        # Draw obstacles in their current positions
        for obs in self.grid.obstacles:
            rect = Rectangle(
                (obs.x - 0.5, obs.y - 0.5),
                obs.width,
                obs.height,
                facecolor="black",
                alpha=0.6,
                edgecolor="black",
            )
            self.ax.add_patch(rect)
            self.obstacle_patches.append(rect)

        self.ax.set_title(
            f"MCTS Navigation | Step {len(self.history) - 1} | Distance {self.dist_traveled:.1f} "
            f"| Planning: {self.time_planning:.1f}s"
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


    def close_plot(self):
        """Disable interactive mode when the simulation finishes."""
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    static_grid = Grid(
        width=100,
        height=100,
        obstacles=[
            Obstacle(10,30,10,3),
            Obstacle(8,45,5,5),
            Obstacle(70,50,20,5),
            Obstacle(75,80,11,7)
        ],
        stay_prob=1.0,
        start=(0, 0),
        goal=(99,99),
        movement_probs=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        seed=42,
    )

    small_static_grid = Grid(
        width=10,
        height=10,
        obstacles=[
            Obstacle(3,4,7,1),
            #Obstacle(6,7,7,2),
            Obstacle(5,5,1,3)
        ],
        stay_prob=1.0,
        start=(0, 0),
        goal=(9,9),
        movement_probs=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        seed=42
    )

    #dynamic_grid = Grid()
    actor = MCTSPlanner(grid=small_static_grid,c=0.1,open_loop=False,max_iters=1000,rollout_horizon=50,profile=False)
    #actor = ReplanAStar(grid=static_grid)

    actor.init_plot()

    try:
        while True:
            result = actor.step()

            if result["reached_goal"]:
                print("Reached goal!")
                break

            if result["collision"]:
                print("Collision!")
                break

            if result.get("no_path"):
                print("No path found!")
                break

    finally:
        actor.close_plot()