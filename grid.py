import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time
import copy
from dataclasses import dataclass
import random
from math import prod


def generate_grid(
    width=100,
    height=100,
    obstacle_count=None,
    density=None,
    obstacle_size=(3, 10),
    static_fraction=0.0,
    stay_prob=0.2,
    left_bias=0.7,
    start=(0, 0),
    goal=None,
    clearance=2,
    vertical_buffer=0.0,
    seed=42,
    track_density=False,
    start_clearance=5,
):
    """
    Generate a grid containing rectangular obstacles.

    Specify either obstacle_count or density, but not both.

    density: Approximate fraction of grid area initially occupied.
    obstacle_size: (minimum, maximum) side length.
    static_fraction: Fraction of obstacles that are stationary.
    stay_prob: Probability a dynamic obstacle stays in place.
    left_bias: Fraction of movement probability assigned to
                W, NW, SW (the remainder goes to other directions).
    clearance: Minimum separation around obstacles and the goal.
    start_clearance: Number of cells to keep clear around the start.
    vertical_buffer: Optional fraction of height to leave clear at each edge.
    track_density: Enable density forecasts for dynamic A*.
    """
    rng = random.Random(seed)
    goal = goal if goal is not None else (width - 1, height - 1)

    if (obstacle_count is None) == (density is None):
        raise ValueError("Specify exactly one of obstacle_count or density")
    if not (0 <= static_fraction <= 1):
        raise ValueError("static_fraction must be in [0, 1]")
    if not (0 <= stay_prob <= 1 and 0 <= left_bias <= 1):
        raise ValueError("stay_prob and left_bias must be in [0, 1]")
    if not isinstance(start_clearance, int) or start_clearance < 0:
        raise ValueError("start_clearance must be a nonnegative integer")

    min_size, max_size = obstacle_size
    if min_size < 1 or max_size < min_size:
        raise ValueError("Invalid obstacle_size")

    if density is not None and not (0 <= density <= 1):
        raise ValueError("density must be in [0, 1]")

    # N, NE, E, SE, S, SW, W, NW
    left_weights = [0, 0, 0, 0, 0, 0.25, 0.5, 0.25]
    other_weights = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0]

    movement_probs = tuple(
        (1 - stay_prob) * (left_bias * left + (1 - left_bias) * other)
        for left, other in zip(left_weights, other_weights)
    )

    obstacles = []
    occupied_area = 0
    target_area = density * width * height if density is not None else None

    def overlaps(x, y, w, h, ox, oy, ow, oh, margin):
        return (
            x < ox + ow + margin
            and x + w + margin > ox
            and y < oy + oh + margin
            and y + h + margin > oy
        )

    def valid_position(x, y, w, h):
        # Keep a local buffer around the start, and keep the goal clear.
        if overlaps(x, y, w, h, *start, 1, 1, start_clearance):
            return False
        if overlaps(x, y, w, h, *goal, 1, 1, clearance):
            return False

        for obs in obstacles:
            if overlaps(x, y, w, h, obs.x, obs.y, obs.width, obs.height, clearance):
                return False

        return True

    max_attempts = 10000
    attempts = 0

    while attempts < max_attempts:
        if obstacle_count is not None:
            if len(obstacles) >= obstacle_count:
                break
        elif occupied_area >= target_area:
            break

        attempts += 1

        w = rng.randint(min_size, min(max_size, width))
        h = rng.randint(min_size, min(max_size, height))

        # Optionally leave a clear area at the top and bottom.
        buffer_cells = int(height * vertical_buffer)

        min_y = buffer_cells
        max_y = height - buffer_cells - h

        if max_y < min_y:
            continue  # Obstacle cannot fit in the available vertical space

        x = rng.randint(0, width - w)
        y = rng.randint(min_y, max_y)

        if not valid_position(x, y, w, h):
            continue

        is_static = rng.random() < static_fraction

        obstacles.append(
            Obstacle(
                x=x,
                y=y,
                width=w,
                height=h,
                movement_probs=((0.0,) * 8 if is_static else movement_probs),
                stay_prob=1.0 if is_static else stay_prob,
            )
        )

        occupied_area += w * h

    if attempts >= max_attempts:
        print(
            f"Warning: placed {len(obstacles)} obstacles; "
            f"achieved density {occupied_area / (width * height):.3f}"
        )

    return Grid(
        width=width,
        height=height,
        obstacles=obstacles,
        stay_prob=stay_prob,
        movement_probs=movement_probs,
        start=start,
        goal=goal,
        seed=seed,
        track_density=track_density,
    )


def static_sparse_grid(
    width=100, height=100, start=(0, 0), goal=None, seed=42, start_clearance=5
):
    """Sparse world with about 10% coverage and no moving obstacles."""
    return generate_grid(
        width=width,
        height=height,
        start=start,
        goal=goal,
        seed=seed,
        density=0.10,
        static_fraction=1.0,
        stay_prob=1.0,
        track_density=True,
        start_clearance=start_clearance,
    )


def dynamic_sparse_grid(
    width=100, height=100, start=(0, 0), goal=None, seed=42, start_clearance=5
):
    """Scenario 1: sparse obstacles, each with an 80% chance of moving."""
    return generate_grid(
        width=width,
        height=height,
        start=start,
        goal=goal,
        seed=seed,
        density=0.075,
        static_fraction=0.0,
        stay_prob=0.2,
        track_density=True,
        start_clearance=start_clearance,
    )


def dynamic_dense_grid(
    width=100, height=100, start=(0, 0), goal=None, seed=42, start_clearance=5
):
    """Scenario 2: about 25% coverage, with the same motion as scenario 1."""
    return generate_grid(
        width=width,
        height=height,
        start=start,
        goal=goal,
        seed=seed,
        density=0.15,
        static_fraction=0.0,
        stay_prob=0.2,
        track_density=True,
        start_clearance=start_clearance,
    )


def dynamic_wall_maze_grid(
    width=100, height=100, start=(0, 0), goal=None, seed=42, start_clearance=5
):
    """Dynamic obstacles between walls with alternating gaps.

    With the default start and goal, the robot has to go back and forth
    across the grid instead of taking a direct route to the goal.
    """
    if width < 30 or height < 30:
        raise ValueError("The wall maze needs a grid at least 30 by 30")
    if not isinstance(start_clearance, int) or start_clearance < 0:
        raise ValueError("start_clearance must be a nonnegative integer")

    if goal is None:
        # The live plot shows y=0 at the top, so this is the bottom-left corner.
        goal = (0, height - 1)

    gap_width = max(4, width // 5)
    wall_rows = (height // 4, height // 2, 3 * height // 4)
    obstacles = [
        Obstacle(0, wall_rows[0], width - gap_width, 1, stay_prob=1),
        Obstacle(gap_width, wall_rows[1], width - gap_width, 1, stay_prob=1),
        Obstacle(0, wall_rows[2], width - gap_width, 1, stay_prob=1),
    ]

    def near_start(obstacle):
        x, y = start
        return (
            obstacle.x <= x + start_clearance
            and obstacle.x + obstacle.width > x - start_clearance
            and obstacle.y <= y + start_clearance
            and obstacle.y + obstacle.height > y - start_clearance
        )

    if any(near_start(wall) for wall in obstacles):
        raise ValueError("Start or its clearance overlaps a maze wall")

    # Each corridor has two lanes of pedestrians moving in opposite directions.
    obstacle_size = max(2, min(4, width // 25, height // 15))
    spacing = max(obstacle_size + 2, width // 6)
    # A small chance of moving up or down adds some jitter to each lane.
    west = (0.02, 0, 0.05, 0, 0.02, 0.03, 0.65, 0.03)
    east = (0.02, 0.03, 0.65, 0.03, 0.02, 0, 0.05, 0)
    corridor_edges = (-1, *wall_rows, height)
    for corridor in range(4):
        first_row = corridor_edges[corridor] + 1
        last_row = corridor_edges[corridor + 1] - obstacle_size
        space = last_row - first_row
        lane_rows = (first_row + space // 4, first_row + 3 * space // 4)

        for lane, y in enumerate(lane_rows):
            movement_probs = west if (corridor + lane) % 2 == 0 else east
            offset = 2 + lane * (spacing // 2)
            for x in range(offset, width - obstacle_size + 1, spacing):
                obstacle = Obstacle(
                    x,
                    y,
                    obstacle_size,
                    obstacle_size,
                    movement_probs=movement_probs,
                    stay_prob=0.2,
                )
                if near_start(obstacle):
                    continue
                if (
                    x <= goal[0] < x + obstacle_size
                    and y <= goal[1] < y + obstacle_size
                ):
                    continue
                obstacles.append(obstacle)

    return Grid(
        width,
        height,
        obstacles=obstacles,
        start=start,
        goal=goal,
        seed=seed,
        track_density=True,
    )


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

    DIRECTION_LABELS = {direction: i for i, direction in enumerate(DIRECTIONS[:8])}

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
        self.goal = tuple(goal) if goal is not None else (width - 1, height - 1)
        self.actor_pos = self.start

        self.obstacles = copy.deepcopy(obstacles or [])

        self.track_density = track_density

        # Obstacle movement probabilities
        if movement_probs is None:
            movement_probs = np.full(8, (1.0 - stay_prob) / 8)

        self.movement_probs, self.stay_prob = validate_probabilities(
            movement_probs, stay_prob
        )

        self.movement_choice = np.append(self.movement_probs, stay_prob)

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

        obs.cumulative_probs = np.cumsum((*obs.movement_probs, obs.stay_prob))

    def in_bounds(self, pos):
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def obstacle_contains(self, obs, pos):
        return self.rectangle_contains(
            obs.x, obs.y, obs.width, obs.height, pos
        )

    def obstacles_overlap(self, a, b):
        def overlap_1d(a0, alen, b0, blen, period):
            d = (b0 - a0) % period
            return d < alen or period - d < blen

        return overlap_1d(a.x, a.width, b.x, b.width, self.width) and overlap_1d(
            a.y, a.height, b.y, b.height, self.height
        )
    
    def rectangles_overlap(
        self, ax, ay, aw, ah,
        bx, by, bw, bh,
    ):
        dx = (bx - ax) % self.width
        dy = (by - ay) % self.height

        overlap_x = dx < aw or self.width - dx < bw
        overlap_y = dy < ah or self.height - dy < bh

        return overlap_x and overlap_y

    def obstacle_too_far_oob(self, obs, buffer=2):
        return (
            obs.x + obs.width <= -buffer
            or obs.x >= self.width + buffer
            or obs.y + obs.height <= -buffer
            or obs.y >= self.height + buffer
        )

    def is_occupied(self, pos):
        return any(self.obstacle_contains(obs, pos) for obs in self.obstacles)

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
                action = np.searchsorted(
                    obs.cumulative_probs,
                    self.rng.random(),
                    side="right",
                )
                dx, dy = self.DIRECTIONS[action]

                new_x = (obs.x + dx) % self.width
                new_y = (obs.y + dy) % self.height
                new_x = (obs.x + dx) % self.width
                new_y = (obs.y + dy) % self.height

                # Check goal overlap
                gx, gy = self.goal

                if self.rectangle_contains(
                    new_x, new_y, obs.width, obs.height, self.goal
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
                        # candidate = Obstacle(
                        #     check_x % self.width,
                        #     check_y % self.height,
                        #     obs.width,
                        #     obs.height,
                        # )

                        # if self.obstacles_overlap(candidate, other):
                        #     blocked = True
                        #     break
                        if self.rectangles_overlap(
                            check_x, check_y, obs.width, obs.height,
                            other.x, other.y, other.width, other.height,
                        ):
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    continue
                
                collision = False
                for check_x, check_y in positions:
                    if self.rectangle_contains(
                        check_x, check_y,
                        obs.width, obs.height,
                        self.actor_pos,
                    ):
                        collision = True
                        break

                obs.x = new_x
                obs.y = new_y

                if collision:
                    return self.step_result(collision=True)

        return self.step_result()

    def neighbors(self, pos):
        x, y = pos

        return [
            (x + dx, y + dy)
            for dx, dy in self.DIRECTIONS[:8]
            if self.can_move(pos, (x + dx, y + dy))
        ]
    
    def rectangle_contains(self, x, y, width, height, pos):
        px, py = pos
        return (
            (px - x) % self.width < width
            and (py - y) % self.height < height
        )

    def reset(self):
        self.actor_pos = self.start
        self.obstacles = copy.deepcopy(self.initial_obstacles)

        if self.track_density:
            self.reset_density()

    def clone(self, include_density=True):
        """
        Copy the mutable simulation state without
        recursively copying the entire Grid.
        """
        new = copy.copy(self)

        new.actor_pos = self.actor_pos

        new.obstacles = [copy.copy(obs) for obs in self.obstacles]

        # Independent RNG, preserving current state.
        new.rng = np.random.default_rng()
        new.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)

        # Only copy density if enabled
        if self.track_density and include_density:
            new.density = self.density.copy()
            new.density_maps = {}
            new.density_kernels = {}
            for key in self.density_maps:
                new.density_maps[key] = self.density_maps[key].copy()
                new.density_kernels[key] = self.density_kernels[key].copy()
        elif self.track_density:
            new.track_density = False
            new.density = None
            new.diffusion_kernel = None
            new.density_maps = {}
            new.density_kernels = {}

        return new

    def add_obstacle(
        self, min_y, max_y, width=1, height=1, movement_probs=None, stay_prob=None
    ):
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

        if any(self.obstacles_overlap(obstacle, other) for other in self.obstacles):
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
            raise RuntimeError("Density tracking is disabled")

        for key in self.density_maps:
            self.density_maps[key] = convolve2d(
                self.density_maps[key],
                self.density_kernels[key],
                mode="same",
                boundary="wrap",
            )
        self.update_density()

        return self.density

    
    def obstacle_matrix(self):
        matrix = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        for obs in self.obstacles:
            xs = np.arange(obs.x, obs.x + obs.width) % self.width
            ys = np.arange(obs.y, obs.y + obs.height) % self.height

            matrix[np.ix_(ys, xs)] = 1

        return matrix



if __name__ == "__main__":
    grid = Grid(
        width=100,
        height=100,
        obstacles=[
            Obstacle(10, 30, 10, 3),
            Obstacle(8, 45, 5, 5),
            Obstacle(70, 50, 20, 5),
            Obstacle(75, 80, 11, 7),
        ],
        start=(0, 0),
        goal=(99, 99),
        movement_probs=[0.025, 0.075, 0.75, 0.075, 0.025, 0.02, 0.01, 0.02],
        seed=42,
    )

    fig, ax = plt.subplots()

    for i in range(100):
        ax.imshow(grid.obstacle_matrix())
        grid.step()
        plt.pause(0.3)
