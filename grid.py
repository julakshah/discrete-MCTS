
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import copy
from dataclasses import dataclass

@dataclass
class Obstacle:
    x: int
    y: int
    width: int
    height: int

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

        self.obstacles = copy.deepcopy(obstacles or [])
        self.initial_obstacles = copy.deepcopy(self.obstacles)

        self.density = np.zeros((height, width), dtype=float)

        for obstacle in self.obstacles:
            self.density[
                obstacle.y:obstacle.y + obstacle.height,
                obstacle.x:obstacle.x + obstacle.width
            ] = 1.0

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

        for pos in [self.start, self.goal]:
            if not self.in_bounds(pos):
                raise ValueError(f"Position {pos} is outside the grid")

        for i, obstacle in enumerate(self.obstacles):
            if obstacle.width <= 0 or obstacle.height <= 0:
                raise ValueError("Obstacle dimensions must be positive")

            if self.obstacle_contains(obstacle, self.start):
                raise ValueError("Start must be free of obstacles")

            if self.obstacle_contains(obstacle, self.goal):
                raise ValueError("Goal must be free of obstacles")

            for other in self.obstacles[:i]:
                if self.obstacles_overlap(obstacle, other):
                    raise ValueError("Initial obstacles cannot overlap")

        self.grid = np.zeros((height, width), dtype=np.int8)
        self._update_grid()
    
    def obstacle_contains(self, obstacle, pos):
        x, y = pos
        return (
            obstacle.x <= x < obstacle.x + obstacle.width
            and obstacle.y <= y < obstacle.y + obstacle.height
        )

    def obstacles_overlap(self, a, b):
        return (
            a.x < b.x + b.width
            and a.x + a.width > b.x
            and a.y < b.y + b.height
            and a.y + a.height > b.y
        )

    def obstacle_in_bounds(self, obstacle):
        return (
            obstacle.x >= 0
            and obstacle.y >= 0
            and obstacle.x + obstacle.width <= self.width
            and obstacle.y + obstacle.height <= self.height
        )
    
    def obstacle_too_far_oob(self, obstacle, buffer=2):
        return (
            obstacle.x + obstacle.width <= -buffer
            or obstacle.x >= self.width + buffer
            or obstacle.y + obstacle.height <= -buffer
            or obstacle.y >= self.height + buffer
        )

    def is_occupied(self, pos):
        return any(
            self.obstacle_contains(obstacle, pos)
            for obstacle in self.obstacles
        )

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _update_grid(self):
        self.grid[:] = self.obstacle_matrix()

        gx, gy = self.goal
        self.grid[gy, gx] = self.GOAL

        ax, ay = self.actor_pos
        self.grid[ay, ax] = self.ACTOR
    
    def add_obstacle(self, min_y, max_y, width=1, height=1):
        y_pos = int(self.rng.integers(min_y, max_y + 1))

        obstacle = Obstacle(
            x=0,
            y=y_pos,
            width=width,
            height=height
        )

        if self.obstacle_contains(obstacle, self.start):
            return False

        if self.obstacle_contains(obstacle, self.goal):
            return False

        if any(
            self.obstacles_overlap(obstacle, other)
            for other in self.obstacles
        ):
            return False

        self.obstacles.append(obstacle)

        self.density[
            obstacle.y:obstacle.y + obstacle.height,
            obstacle.x:obstacle.x + obstacle.width
        ] = 1.0

        self._update_grid()
        return True

    def step(self, actor_action=None):
        if actor_action is not None:
            if not isinstance(actor_action, (int, np.integer)) or not 0 <= actor_action < 8:
                raise ValueError("actor_action must be an integer from 0 to 7")

            dx, dy = self.DIRECTIONS[actor_action]
            x, y = self.actor_pos
            new_pos = (x + dx, y + dy)

            if self.in_bounds(new_pos):
                self.actor_pos = new_pos

        # Randomize obstacle update order
        obstacle_order = list(range(len(self.obstacles)))
        self.rng.shuffle(obstacle_order)

        # Sample one movement per obstacle
        actions = self.rng.choice(
            9,
            size=len(self.obstacles),
            p=self.movement_choice
        )

        for idx, action in zip(obstacle_order, actions.tolist()):
            obstacle = self.obstacles[idx]

            dx, dy = self.DIRECTIONS[action]

            candidate = Obstacle(
                x=obstacle.x + dx,
                y=obstacle.y + dy,
                width=obstacle.width,
                height=obstacle.height
            )

            # Keep the goal cell free
            if self.obstacle_contains(candidate, self.goal):
                continue

            # Reject moves overlapping another obstacle
            if any(
                self.obstacles_overlap(candidate, other)
                for j, other in enumerate(self.obstacles)
                if j != idx
            ):
                continue

            obstacle.x = candidate.x
            obstacle.y = candidate.y
        
        # Remove any obstacles out of view + a few more buffer cells away
        self.obstacles = [
            obs for obs in self.obstacles
            if not self.obstacle_too_far_oob(obs, buffer=2)
        ]

        self._update_grid()

        return {
            "collision": self.is_occupied(self.actor_pos),
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

        for dx, dy in self.DIRECTIONS[:8]:
            next_pos = (x + dx, y + dy)

            if self.in_bounds(next_pos) and not self.is_occupied(next_pos):
                neighbors.append(next_pos)

        return neighbors

    def reset(self):
        self.actor_pos = self.start
        self.obstacles = copy.deepcopy(self.initial_obstacles)
        self._update_grid()
    
    def obstacle_matrix(self):
        matrix = np.zeros((self.height, self.width), dtype=np.uint8)

        for obstacle in self.obstacles:
            x0 = max(0, obstacle.x)
            y0 = max(0, obstacle.y)
            x1 = min(self.width, obstacle.x + obstacle.width)
            y1 = min(self.height, obstacle.y + obstacle.height)

            if x0 < x1 and y0 < y1:
                matrix[y0:y1, x0:x1] = 1

        return matrix

if __name__ == "__main__":
    grid = Grid(
        width=100,
        height=100,
        obstacles=[
            Obstacle(10,30,10,3),
            Obstacle(8,45,5,5),
            Obstacle(70,50,20,5),
            Obstacle(75,80,11,7)
        ],
        start=(0, 0),
        goal=(99, 99),
        movement_probs=[0.025, 0.075, 0.75, 0.075, 0.025, 0.02, 0.01, 0.02],
        seed=42,
    )

    print(grid.grid)

    fig,ax = plt.subplots()

    for i in range(100):
        ax.imshow(grid.obstacle_matrix())
        grid.step()
        plt.pause(0.3)
        