
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time

class Grid:
    # Directions as (dx, dy)
    DIRECTIONS = (
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, 0),
    )

    DIRECTION_LABELS = {
        (0, -1): 0,
        (1, -1): 1,
        (1, 0): 2,
        (1, 1): 3,
        (0, 1): 4,
        (-1, 1): 5,
        (-1, 0): 6,
        (-1, -1): 7,
    }

    EMPTY = 0
    OBSTACLE = 1
    ACTOR = 2
    GOAL = 3

    def __init__(
        self,
        width,
        height,
        obstacles=None,
        start=(0, 0),
        goal=None,
        movement_probs=None,
        stay_prob=0.0,
        seed=None,
    ):
        self.width = width
        self.height = height

        self.start = tuple(start)
        self.goal = tuple(goal) if goal is not None else (width - 1, height - 1)
        self.actor_pos = self.start

        self.obstacles = set(map(tuple, obstacles or []))
        self.initial_obstacles = set(map(tuple, obstacles or []))

        self.density = np.zeros((height, width), dtype=float)

        for x, y in self.obstacles:
            self.density[y, x] += 1.0

        # Default: uniform probability across 8 directions
        if movement_probs is None:
            movement_probs = np.ones(8) / 8

        self.movement_probs = np.asarray(movement_probs, dtype=float)
        self.stay_prob = stay_prob
        self.movement_choice = np.append(self.movement_probs, self.stay_prob)

        if self.movement_probs.shape != (8,):
            raise ValueError("movement_probs must contain 8 values")

        if np.any(self.movement_probs < 0) or not 0 <= stay_prob <= 1:
            raise ValueError("Probabilities must be nonnegative")

        if not np.isclose(self.movement_probs.sum() + stay_prob, 1.0):
            raise ValueError("Movement probabilities must sum to 1")

        self.rng = np.random.default_rng(seed)

        for pos in [self.start, self.goal, *self.obstacles]:
            if not self.in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the grid")

        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start and goal must be free of obstacles")

        self.grid = np.zeros((height, width), dtype=np.int8)
        self._update_grid()

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _update_grid(self):
        self.grid.fill(self.EMPTY)

        for x, y in self.obstacles:
            self.grid[y, x] = self.OBSTACLE

        gx, gy = self.goal
        self.grid[gy, gx] = self.GOAL

        ax, ay = self.actor_pos
        self.grid[ay, ax] = self.ACTOR
    
    def add_obstacle(self, min_y, max_y):
        y_pos = int(self.rng.choice(np.arange(min_y,max_y+1)))
        print(y_pos)
        self.grid[y_pos, 0] = self.OBSTACLE
        self.density[y_pos, 0] = 1.0
        self.obstacles.add((0,y_pos))

    def step(self, actor_action=None):
        """
        Advance the simulation by one time step.

        actor_action: integer 0-7 corresponding to DIRECTIONS,
                      or None to stay still.

        Returns:
            dict with collision and goal status.
        """
        if actor_action is not None:
            if not isinstance(actor_action, (int, np.integer)) or not 0 <= actor_action < 8:
                raise ValueError("actor_action must be an integer from 0 to 7")

            dx, dy = self.DIRECTIONS[actor_action]
            x, y = self.actor_pos
            new_pos = (int(x + dx), int(y + dy))

            if self.in_bounds(new_pos):
                self.actor_pos = new_pos

        # Move obstacles in randomized order
        new_obstacles = set()
        remaining = set(self.obstacles)

        obstacle_order = list(self.obstacles)
        self.rng.shuffle(obstacle_order)

        actions = self.rng.choice(
            9,
            size=len(self.obstacles),
            p=self.movement_choice
        )

        new_positions = self.obstacles + self.DIRECTIONS[actions]

        # Keep obstacles inside the grid
        new_positions[:, 0] = np.clip(
            new_positions[:, 0], 0, self.width - 1
        )
        new_positions[:, 1] = np.clip(
            new_positions[:, 1], 0, self.height - 1
        )

        self.obstacles = new_positions
        self._update_grid()

        return {
            "collision": self.actor_pos in self.obstacles,
            "reached_goal": self.actor_pos == self.goal,
            "actor_pos": self.actor_pos,
        }
    
    def diffuse_step(self):
        """
        Advance the predicted obstacle density by one time step.
        Does not change the actual obstacle positions.
        """

        kernel = np.zeros((3, 3))

        for prob, (dx, dy) in zip(self.movement_probs, self.DIRECTIONS):
            kernel[1 + dy, 1 + dx] = prob

        kernel[1, 1] = self.stay_prob

        t0 = time.time()
        self.density = convolve2d(
            self.density,
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        print(f"convolve took {time.time() - t0} seconds")

        return self.density

    def neighbors(self, pos):
        x, y = pos
        neighbors = []

        for dx, dy in self.DIRECTIONS:
            next_pos = (int(x + dx), int(y + dy))

            if self.in_bounds(next_pos) and next_pos not in self.obstacles:
                neighbors.append(next_pos)
            
        return neighbors

    def reset(self):
        """Reset actor and grid to their initial configuration."""
        self.actor_pos = self.start
        self.obstacles = self.initial_obstacles.copy()
        self._update_grid()

if __name__ == "__main__":
    grid = Grid(
        width=100,
        height=100,
        obstacles=[(3, 54), (5, 48), (2, 57), (7,42)],
        start=(0, 0),
        goal=(9, 9),
        movement_probs=[0.025, 0.075, 0.75, 0.075, 0.025, 0.02, 0.01, 0.02],
        seed=42,
    )

    print(grid.grid)

    fig,ax = plt.subplots()

    for i in range(100):
        ax.imshow(grid.density, cmap='viridis', norm=LogNorm())
        grid.diffuse_step()
        if i % 5 == 0:
            grid.add_obstacle(0.4*100,0.6*100)
        plt.pause(0.3)
        