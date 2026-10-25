from bfs import bfs
from astar_ben import astar
from mcts import mcts
from grid import Grid
import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
import time

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


class ReplanBFS:
    def __init__(self,grid):
        self.grid = grid
        self.history = [grid.actor_pos]
        self.time_planning = 0

    def step(self):
        t0 = time.perf_counter()
        path = bfs(self.grid.actor_pos, self.grid.goal, self.grid.neighbors)
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


class MCTS:
    def __init__(self,grid):
        self.grid = grid
        self.history = [grid.actor_pos]
        self.time_planning = 0

    def step(self):
        t0 = time.perf_counter()
        with cProfile.Profile() as profiler:
            action = mcts(self.grid.actor_pos, self.grid.goal, self.grid)
        
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumtime").print_stats(20)

        self.time_planning += time.perf_counter() - t0

        # if path is None:
        #     return {"collision": False, "reached_goal": False, "no_path": True}

        # if len(path) == 1:
        #     return {"collision": False, "reached_goal": True}

        res = self.grid.step(actor_action=int(action))
        self.history.append(self.grid.actor_pos)

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

if __name__ == "__main__":
    static_grid = Grid(
        width=100,
        height=100,
        obstacles=[(i, 30) for i in range(70)] + [(99-i, 70) for i in range(70)],
        stay_prob=1.0,
        start=(0, 0),
        goal=(99,99),
        movement_probs=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
        seed=42,
    )

    #dynamic_grid = Grid()
    #actor = ReplanBFS(grid=static_grid)
    actor = MCTS(grid=static_grid)
    #actor = ReplanAStar(grid=static_grid)
    
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
    
    actor.plot_path()
