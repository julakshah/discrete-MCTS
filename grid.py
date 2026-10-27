
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

def overlaps_at(x, y, width, height, other):
    return (
        x < other.x + other.width
        and x + width > other.x
        and y < other.y + other.height
        and y + height > other.y
    )

class Grid:
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
        direction: i
        for i, direction in enumerate(DIRECTIONS[:8])
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
        track_density=False,
    ):
        self.width = width
        self.height = height

        self.start = tuple(start)
        self.goal = (
            tuple(goal)
            if goal is not None
            else (width - 1, height - 1)
        )
        self.actor_pos = self.start

        self.obstacles = copy.deepcopy(obstacles or [])
        self.initial_obstacles = copy.deepcopy(self.obstacles)

        self.track_density = track_density

        # Obstacle movement probabilities
        if movement_probs is None:
            movement_probs = np.full(8, (1.0 - stay_prob) / 8)

        self.movement_probs = np.asarray(
            movement_probs, dtype=float
        )

        if self.movement_probs.shape != (8,):
            raise ValueError("movement_probs must contain 8 values")

        if np.any(self.movement_probs < 0) or not 0 <= stay_prob <= 1:
            raise ValueError("Invalid movement probabilities")

        if not np.isclose(
            self.movement_probs.sum() + stay_prob, 1.0
        ):
            raise ValueError("Movement probabilities must sum to 1")

        self.stay_prob = stay_prob

        self.movement_choice = np.append(
            self.movement_probs, stay_prob
        )

        self.rng = np.random.default_rng(seed)
        self.cumulative_probs = np.cumsum(self.movement_choice)

        # Validate positions
        if not self.in_bounds(self.start):
            raise ValueError("Start is outside grid")

        if not self.in_bounds(self.goal):
            raise ValueError("Goal is outside grid")

        for i, obs in enumerate(self.obstacles):
            if obs.width <= 0 or obs.height <= 0:
                raise ValueError("Invalid obstacle dimensions")

            if self.obstacle_contains(obs, self.start):
                raise ValueError("Start overlaps obstacle")

            if self.obstacle_contains(obs, self.goal):
                raise ValueError("Goal overlaps obstacle")

            for other in self.obstacles[:i]:
                if self.obstacles_overlap(obs, other):
                    raise ValueError("Initial obstacles overlap")

        # Optional density map for A*
        self.density = None
        self.diffusion_kernel = None

        if self.track_density:
            self.density = self.obstacle_matrix().astype(float)
            self.diffusion_kernel = self._make_diffusion_kernel()

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    @staticmethod
    def obstacle_contains(obs, pos):
        x, y = pos

        return (
            obs.x <= x < obs.x + obs.width
            and obs.y <= y < obs.y + obs.height
        )

    @staticmethod
    def obstacles_overlap(a, b):
        return (
            a.x < b.x + b.width
            and a.x + a.width > b.x
            and a.y < b.y + b.height
            and a.y + a.height > b.y
        )

    def obstacle_too_far_oob(self, obs, buffer=2):
        return (
            obs.x + obs.width <= -buffer
            or obs.x >= self.width + buffer
            or obs.y + obs.height <= -buffer
            or obs.y >= self.height + buffer
        )

    def is_occupied(self, pos):
        return any(
            self.obstacle_contains(obs, pos)
            for obs in self.obstacles
        )

    def step(self, actor_action=None):
        if actor_action is not None:
            if (
                not isinstance(actor_action, (int, np.integer))
                or not 0 <= actor_action < 8
            ):
                raise ValueError("Invalid actor action")

            dx, dy = self.DIRECTIONS[actor_action]
            x, y = self.actor_pos

            new_pos = (x + dx, y + dy)

            if self.in_bounds(new_pos):
                self.actor_pos = new_pos

        # Move obstacles in random order
        n = len(self.obstacles)

        if n:
            obstacle_order = self.rng.permutation(n)

            actions = np.searchsorted(
                self.cumulative_probs,
                self.rng.random(n),
                side="right"
            )

            for idx, action in zip(obstacle_order, actions):
                obs = self.obstacles[idx]

                dx, dy = self.DIRECTIONS[action]

                new_x = obs.x + dx
                new_y = obs.y + dy

                # Check goal overlap
                gx, gy = self.goal

                if (
                    new_x <= gx < new_x + obs.width
                    and new_y <= gy < new_y + obs.height
                ):
                    continue

                # Check other obstacles
                if any(
                    overlaps_at(
                        new_x, new_y, obs.width, obs.height, other
                    )
                    for j, other in enumerate(self.obstacles)
                    if j != idx
                ):
                    continue

                obs.x = new_x
                obs.y = new_y

            # Remove obstacles sufficiently far OOB
            self.obstacles = [
                obs for obs in self.obstacles
                if not self.obstacle_too_far_oob(obs)
            ]

        return {
            "collision": self.is_occupied(self.actor_pos),
            "reached_goal": self.actor_pos == self.goal,
            "actor_pos": self.actor_pos,
        }

    def neighbors(self, pos):
        x, y = pos

        return [
            (x + dx, y + dy)
            for dx, dy in self.DIRECTIONS[:8]
            if (
                self.in_bounds((x + dx, y + dy))
                and not self.is_occupied((x + dx, y + dy))
            )
        ]

    def reset(self):
        self.actor_pos = self.start
        self.obstacles = copy.deepcopy(self.initial_obstacles)

        if self.track_density:
            self.density = self.obstacle_matrix().astype(float)

    def clone(self):
        """
        Copy the mutable simulation state without
        recursively copying the entire Grid.
        """
        new = copy.copy(self)

        new.actor_pos = self.actor_pos

        new.obstacles = [
            copy.copy(obs) for obs in self.obstacles
        ]

        # Independent RNG, preserving current state.
        new.rng = np.random.default_rng()
        new.rng.bit_generator.state = copy.deepcopy(
            self.rng.bit_generator.state
        )

        # Only copy density if enabled
        if self.track_density:
            new.density = self.density.copy()

        return new

    def add_obstacle(self, min_y, max_y, width=1, height=1):
        y = int(self.rng.integers(min_y, max_y + 1))

        obstacle = Obstacle(
            x=0,
            y=y,
            width=width,
            height=height,
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

        if self.track_density:
            self._add_obstacle_to_density(obstacle)

        return True

    def _make_diffusion_kernel(self):
        kernel = np.zeros((3, 3), dtype=float)

        for prob, (dx, dy) in zip(
            self.movement_probs,
            self.DIRECTIONS[:8],
        ):
            kernel[1 + dy, 1 + dx] = prob

        kernel[1, 1] = self.stay_prob

        return kernel

    def _add_obstacle_to_density(self, obs):
        x0 = max(0, obs.x)
        y0 = max(0, obs.y)
        x1 = min(self.width, obs.x + obs.width)
        y1 = min(self.height, obs.y + obs.height)

        if x0 < x1 and y0 < y1:
            self.density[y0:y1, x0:x1] = 1.0

    def diffuse_step(self):
        if not self.track_density:
            raise RuntimeError(
                "Density tracking is disabled"
            )

        self.density = convolve2d(
            self.density,
            self.diffusion_kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        return self.density

    def obstacle_matrix(self):
        matrix = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        for obs in self.obstacles:
            x0 = max(0, obs.x)
            y0 = max(0, obs.y)
            x1 = min(self.width, obs.x + obs.width)
            y1 = min(self.height, obs.y + obs.height)

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

    fig,ax = plt.subplots()

    for i in range(100):
        ax.imshow(grid.obstacle_matrix())
        grid.step()
        plt.pause(0.3)
        