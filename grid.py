
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
    # N, NE, E, SE, S, SW, W, NW. Movement + stay probabilities must add to 1.
    # Leaving both as None uses the grid's probabilities.
    movement_probs: tuple | None = None
    stay_prob: float | None = None


def validate_probabilities(movement_probs, stay_prob):
    movement_probs = np.asarray(movement_probs, dtype=float)
    if movement_probs.shape != (8,):
        raise ValueError("movement_probs must contain 8 values")
    if not np.all(np.isfinite(movement_probs)) or not np.isfinite(stay_prob):
        raise ValueError("Probabilities must be finite numbers")
    if np.any(movement_probs < 0) or not 0 <= stay_prob <= 1:
        raise ValueError("Probabilities must be between 0 and 1")
    if not np.isclose(movement_probs.sum() + stay_prob, 1.0):
        raise ValueError("Movement probabilities must sum to 1")
    return movement_probs.copy(), float(stay_prob)

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

        self.track_density = track_density

        # Obstacle movement probabilities
        if movement_probs is None:
            movement_probs = np.full(8, (1.0 - stay_prob) / 8)

        self.movement_probs, self.stay_prob = validate_probabilities(
            movement_probs, stay_prob
        )

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
            self.set_obstacle_probs(obs)
            if obs.width <= 0 or obs.height <= 0:
                raise ValueError("Invalid obstacle dimensions")

            if self.obstacle_contains(obs, self.start):
                raise ValueError("Start overlaps obstacle")

            if self.obstacle_contains(obs, self.goal):
                raise ValueError("Goal overlaps obstacle")

            for other in self.obstacles[:i]:
                if self.obstacles_overlap(obs, other):
                    raise ValueError("Initial obstacles overlap")

        self.initial_obstacles = copy.deepcopy(self.obstacles)

        # Optional density map for A*
        self.density = None
        self.diffusion_kernel = None
        self.density_maps = {}
        self.density_kernels = {}

        if self.track_density:
            self.diffusion_kernel = self._make_diffusion_kernel()
            self.reset_density()

    def set_obstacle_probs(self, obs):
        movement_probs = obs.movement_probs
        stay_prob = obs.stay_prob

        if movement_probs is None and stay_prob is None:
            movement_probs = self.movement_probs
            stay_prob = self.stay_prob
        elif movement_probs is None:
            # If only stay is given, make the other directions equally likely.
            movement_probs = np.full(8, (1.0 - stay_prob) / 8)
        elif stay_prob is None:
            stay_prob = 1.0 - np.sum(movement_probs)
            if abs(stay_prob) < 1e-12:  # Account for rounding near zero.
                stay_prob = 0.0

        movement_probs, stay_prob = validate_probabilities(movement_probs, stay_prob)
        # Tuples let us group obstacles with the same probabilities later.
        obs.movement_probs = tuple(movement_probs)
        obs.stay_prob = stay_prob

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

    def can_move(self, pos, new_pos):
        if not self.in_bounds(new_pos) or self.is_occupied(new_pos):
            return False

        x, y = pos
        new_x, new_y = new_pos
        if new_x != x and new_y != y:
            # Don't let a diagonal move squeeze past an occupied corner.
            if self.is_occupied((new_x, y)) or self.is_occupied((x, new_y)):
                return False
        return True

    def step_result(self, collision=False):
        if self.track_density:
            self.reset_density()
        collision = collision or self.is_occupied(self.actor_pos)
        return {
            "collision": collision,
            "reached_goal": not collision and self.actor_pos == self.goal,
            "actor_pos": self.actor_pos,
        }

    def step(self, actor_action=None):
        if actor_action is not None:
            if (
                not isinstance(actor_action, (int, np.integer))
                or not 0 <= actor_action < 8
            ):
                raise ValueError("Invalid actor action")

        # A collision can't be undone by moving away in the same turn.
        if self.is_occupied(self.actor_pos):
            return self.step_result(collision=True)

        if actor_action is not None:
            dx, dy = self.DIRECTIONS[actor_action]
            x, y = self.actor_pos

            new_pos = (x + dx, y + dy)

            if self.in_bounds(new_pos):
                if not self.can_move(self.actor_pos, new_pos):
                    return self.step_result(collision=True)
                self.actor_pos = new_pos

        # Move obstacles in random order
        n = len(self.obstacles)

        if n:
            obstacle_order = self.rng.permutation(n)

            for idx in obstacle_order:
                obs = self.obstacles[idx]

                # Pick a direction using this obstacle's probabilities.
                probabilities = np.append(obs.movement_probs, obs.stay_prob)
                probabilities = probabilities / probabilities.sum()
                action = self.rng.choice(9, p=probabilities)
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

                # Check the destination and both sides of a diagonal move.
                positions = [(new_x, new_y)]
                if dx != 0 and dy != 0:
                    positions.append((new_x, obs.y))
                    positions.append((obs.x, new_y))

                blocked = False
                for check_x, check_y in positions:
                    for j, other in enumerate(self.obstacles):
                        if j == idx:
                            continue
                        if overlaps_at(check_x, check_y, obs.width, obs.height, other):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue

                collision = False
                actor_x, actor_y = self.actor_pos
                for check_x, check_y in positions:
                    if (check_x <= actor_x < check_x + obs.width
                            and check_y <= actor_y < check_y + obs.height):
                        collision = True
                        break

                obs.x = new_x
                obs.y = new_y

                if collision:
                    return self.step_result(collision=True)

            # Remove obstacles sufficiently far OOB
            self.obstacles = [
                obs for obs in self.obstacles
                if not self.obstacle_too_far_oob(obs)
            ]

        return self.step_result()

    def neighbors(self, pos):
        x, y = pos

        return [
            (x + dx, y + dy)
            for dx, dy in self.DIRECTIONS[:8]
            if self.can_move(pos, (x + dx, y + dy))
        ]

    def reset(self):
        self.actor_pos = self.start
        self.obstacles = copy.deepcopy(self.initial_obstacles)

        if self.track_density:
            self.reset_density()

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
            new.density_maps = {}
            new.density_kernels = {}
            for key in self.density_maps:
                new.density_maps[key] = self.density_maps[key].copy()
                new.density_kernels[key] = self.density_kernels[key].copy()

        return new

    def add_obstacle(self, min_y, max_y, width=1, height=1,
                     movement_probs=None, stay_prob=None):
        y = int(self.rng.integers(min_y, max_y + 1))

        obstacle = Obstacle(
            x=0,
            y=y,
            width=width,
            height=height,
            movement_probs=movement_probs,
            stay_prob=stay_prob,
        )
        self.set_obstacle_probs(obstacle)

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
            self.reset_density()

        return True

    def _make_diffusion_kernel(self, movement_probs=None, stay_prob=None):
        if movement_probs is None:
            movement_probs = self.movement_probs
        if stay_prob is None:
            stay_prob = self.stay_prob
        kernel = np.zeros((3, 3), dtype=float)

        for prob, (dx, dy) in zip(
            movement_probs,
            self.DIRECTIONS[:8],
        ):
            kernel[1 + dy, 1 + dx] = prob

        kernel[1, 1] = stay_prob

        return kernel

    def _add_obstacle_to_density(self, obs, density):
        x0 = max(0, obs.x)
        y0 = max(0, obs.y)
        x1 = min(self.width, obs.x + obs.width)
        y1 = min(self.height, obs.y + obs.height)

        if x0 < x1 and y0 < y1:
            density[y0:y1, x0:x1] = 1.0

    def reset_density(self):
        """Start the density prediction from the obstacles' current positions."""
        if not self.track_density:
            raise RuntimeError("Density tracking is disabled")
        self.density_maps = {}
        self.density_kernels = {}
        # Obstacles with the same movement probabilities can share a map.
        for obs in self.obstacles:
            key = (obs.movement_probs, obs.stay_prob)
            if key not in self.density_maps:
                self.density_maps[key] = np.zeros((self.height, self.width))
                self.density_kernels[key] = self._make_diffusion_kernel(
                    obs.movement_probs, obs.stay_prob
                )
            self._add_obstacle_to_density(obs, self.density_maps[key])
        self.update_density()

    def update_density(self):
        self.density = np.zeros((self.height, self.width), dtype=float)
        for density_map in self.density_maps.values():
            self.density += density_map
        # Predicted obstacles can overlap, so cap the combined density at 1.
        # This prediction doesn't account for blocked obstacle movements.
        self.density = np.clip(self.density, 0.0, 1.0)

    def diffuse_step(self):
        if not self.track_density:
            raise RuntimeError(
                "Density tracking is disabled"
            )

        for key in self.density_maps:
            self.density_maps[key] = convolve2d(
                self.density_maps[key], self.density_kernels[key], mode="same",
                boundary="fill", fillvalue=0,
            )
        self.update_density()

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
