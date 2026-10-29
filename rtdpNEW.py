from __future__ import annotations

import copy
import math
import time

import numpy as np

from planner import Planner


# Each state needs to store the robot's cell and the obstacle configuration.
# V(s) is the estimated cost of going from state s to the goal.
value_function = {}

DIRECTIONS = (
    (0, -1),
    (1, -1),
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
)

WAIT_ACTION = len(DIRECTIONS)
# Waiting consumes one simulation tick, so it must not be a zero-cost
# self-loop in the Bellman backup.
WAIT_COST = 1.0

COSTS = {
    (0, -1): 1,
    (1, -1): math.sqrt(2),
    (1, 0): 1,
    (1, 1): math.sqrt(2),
    (0, 1): 1,
    (-1, 1): math.sqrt(2),
    (-1, 0): 1,
    (-1, -1): math.sqrt(2),
}


def heuristic(start, end):
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


class RTDP(Planner):
    """
    Sampled-transition RTDP for the existing Grid environment.

    This implementation:
    - Uses the full robot and obstacle configuration as the state.
    - Initializes state values with an octile-distance heuristic.
    - Runs RTDP trials using sampled environment transitions.
    - Uses Bellman backups to update state values.
    - Rejects actions that collide in any sampled outcome.
    - Uses the existing Planner interface for execution and metrics.

    Actions are filtered against all possible one-tick obstacle moves.
    Longer-horizon values and trials still use sampled transitions.
    """

    def __init__(
        self,
        grid,
        max_trials=100,
        max_trial_steps=20,
        transition_samples=10,
        verbose=True,
    ):
        super().__init__(grid)

        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        if max_trial_steps < 1:
            raise ValueError("max_trial_steps must be positive")
        if transition_samples < 1:
            raise ValueError("transition_samples must be positive")

        self.plot_label = "RTDP"
        self.max_trials = max_trials
        self.max_trial_steps = max_trial_steps
        self.transition_samples = transition_samples
        self.verbose = verbose

        # Keep planning randomness reproducible without advancing the live grid.
        self.simulation_rng = np.random.default_rng()
        self.simulation_rng.bit_generator.state = copy.deepcopy(
            grid.rng.bit_generator.state
        )

        # V(s) is estimated cost from a complete state to the goal.
        self.value_function = {}

        self.result = None

    def get_state(self, env):
        """
        Represent the complete state using immutable, hashable tuples.

        Preserve obstacle order because Grid.step() moves obstacles
        sequentially, so their order can affect the resulting transition.
        """
        obstacles = tuple(
            (
                obs.x,
                obs.y,
                obs.width,
                obs.height,
            )
            for obs in env.obstacles
        )

        return (
            tuple(env.actor_pos),
            obstacles,
        )

    def initialize_value(self, env):
        """Initialize an unseen state's value using the heuristic."""
        state = self.get_state(env)

        if state not in self.value_function:
            self.value_function[state] = heuristic(
                env.actor_pos,
                env.goal,
            )

        return self.value_function[state]

    def is_goal(self, env):
        return env.actor_pos == env.goal

    def safe_from_next_obstacle_moves(self, env, actor_pos):
        """Reject positions any obstacle could sweep through next tick."""
        for obs in env.obstacles:
            for action, (dx, dy) in enumerate(env.DIRECTIONS[:8]):
                if obs.movement_probs[action] == 0:
                    continue

                x = (obs.x + dx) % env.width
                y = (obs.y + dy) % env.height
                if env.rectangle_contains(x, y, obs.width, obs.height, env.goal):
                    continue

                positions = [(x, y)]
                if dx and dy:
                    positions.extend(((x, obs.y), (obs.x, y)))

                if any(
                    env.rectangle_contains(px, py, obs.width, obs.height, actor_pos)
                    for px, py in positions
                ):
                    return False

        return True

    def legal_actions(self, env):
        """
        Return actions valid now and safe against any possible next obstacle move.

        Movement actions are indices 0 through 7.  WAIT_ACTION represents
        Grid.step(actor_action=None): the actor stays put while obstacles
        advance one tick.
        """
        actions = []

        for action, (dx, dy) in enumerate(DIRECTIONS):
            new_pos = (
                env.actor_pos[0] + dx,
                env.actor_pos[1] + dy,
            )

            if not env.in_bounds(new_pos):
                continue

            if not env.can_move(env.actor_pos, new_pos):
                continue

            if not self.safe_from_next_obstacle_moves(env, new_pos):
                continue

            actions.append(action)

        # Waiting is a real Grid transition, not a planner-side no-op.  It is
        # safe only when no possible obstacle move can enter the actor's cell.
        if self.safe_from_next_obstacle_moves(env, env.actor_pos):
            actions.append(WAIT_ACTION)

        return actions

    def apply_action(self, env, action):
        """Apply an RTDP action using Grid's movement/wait convention."""
        if action == WAIT_ACTION:
            return env.step()
        return env.step(actor_action=action)

    def action_cost(self, action):
        if action == WAIT_ACTION:
            return WAIT_COST
        return COSTS[DIRECTIONS[action]]

    def sample_transition(self, env, action):
        """
        Simulate one action on a clone, leaving the original environment
        unchanged. Returns the successor environment and step result.
        """
        simulated_env = env.clone(include_density=False)
        simulated_env.rng = self.simulation_rng
        self.nodes_visited += 1
        result = self.apply_action(simulated_env, action)

        return simulated_env, result

    def q_value(self, env, action):
        """
        Estimate Q(s,a) using sampled transitions.

        Q(s,a) = c(s,a) + E[V(s')]

        If any sampled outcome collides, reject this action for this
        decision. This is a conservative sample-based rule, not a
        mathematical guarantee of safety.
        """
        total_cost = 0.0

        for _ in range(self.transition_samples):
            next_env, result = self.sample_transition(env, action)

            # Reject actions with sampled collisions.
            if result["collision"]:
                return float("inf")

            immediate_cost = self.action_cost(action)

            if result["reached_goal"]:
                total_cost += immediate_cost
            else:
                next_value = self.initialize_value(next_env)
                total_cost += immediate_cost + next_value

        return total_cost / self.transition_samples

    def greedy_action(self, env):
        """Pick the action that minimizes the estimated expected Q-value."""
        best_action = None
        best_q = float("inf")

        for action in self.legal_actions(env):
            q = self.q_value(env, action)

            if q < best_q:
                best_q = q
                best_action = action

        return best_action, best_q

    def bellman_backup(self, env):
        """Update V(s) using the current estimated action values."""
        state = self.get_state(env)

        if self.is_goal(env):
            self.value_function[state] = 0.0
            return None

        self.initialize_value(env)

        best_action, best_q = self.greedy_action(env)

        if math.isfinite(best_q):
            self.value_function[state] = best_q

        return best_action

    def trial(self, initial_env):
        """
        Start a trial from the current environment configuration.

        Each step chooses a greedy action, performs a Bellman backup, and
        samples the resulting transition.  WAIT_ACTION advances obstacles
        without moving the actor.
        """
        env = initial_env.clone(include_density=False)

        for _ in range(self.max_trial_steps):
            if self.is_goal(env):
                return True

            action = self.bellman_backup(env)

            if action is None:
                # No sampled-safe movement action exists.
                # Do not deliberately move into a known obstacle.
                return False

            next_env, result = self.sample_transition(env, action)

            if result["collision"]:
                return False

            env = next_env

            if result["reached_goal"]:
                self.value_function[self.get_state(env)] = 0.0
                return True

        return False

    def RTDP(self, initial_env):
        """
        Run repeated trials and return the greedy action at the
        initial state, based on the learned value function.
        """
        for _ in range(self.max_trials):
            self.trial(initial_env)

        best_action, _ = self.greedy_action(initial_env)
        return best_action

    def step(self):
        """
        Perform one real simulation step using the learned RTDP policy.
        Compatible with the existing Planner interface.
        """
        if self.result is not None:
            return self.result

        if self.is_goal(self.grid):
            self.result = self.add_metrics({
                "collision": False,
                "reached_goal": True,
            })
            return self.result

        start_pos = self.grid.actor_pos

        t0 = time.perf_counter()
        action = self.RTDP(self.grid)
        self.time_planning += time.perf_counter() - t0

        if action is None:
            # No action passed the sampled safety checks.
            # Do not execute a movement that the planner rejected.
            result = {
                "collision": False,
                "reached_goal": False,
                "no_path": True,
                "actor_pos": self.grid.actor_pos,
            }
        else:
            result = self.apply_action(self.grid, action)

            if self.grid.actor_pos != start_pos:
                self.dist_traveled += self.action_cost(action)

            if self.verbose:
                if action == WAIT_ACTION:
                    print("RTDP selected wait action")
                else:
                    print(
                        f"RTDP selected action {action}: "
                        f"{DIRECTIONS[action]}"
                    )

        self.history.append(self.grid.actor_pos)
        self.frames.append((
            self.grid.obstacle_matrix(),
            self.grid.actor_pos,
        ))

        self.update_plot()
        result = self.add_metrics(result)

        if (
            result["collision"]
            or result["reached_goal"]
            or result.get("no_path")
        ):
            self.result = result

        return result


if __name__ == "__main__":
    from grid import dynamic_sparse_grid, dynamic_wall_maze_grid, dynamic_dense_grid

    # grid = dynamic_sparse_grid(
    #     width=50,
    #     height=50,
    #     start=(0,0),
    #     goal=(20,20),
    #     seed=0,
    #     start_clearance=2,
    # )
    # grid = dynamic_wall_maze_grid(
    # width=30,
    # height=30,
    # seed=0,
    # start_clearance=5)

    grid = dynamic_dense_grid()

    planner = RTDP(
        grid,
        # max_trials=20,
        # max_trial_steps=100,
        max_trials=20,
        max_trial_steps=50,
        transition_samples=3,
    )

    planner.init_plot()

    try:
        for _ in range(200):
            result = planner.step()

            if result["reached_goal"]:
                print("Robot reached the goal!")
                break

            if result["collision"]:
                print("Robot collided with an obstacle.")
                break

            if result.get("no_path"):
                print("No immediately safe action available.")
                break
        else:
            print("Stopped at the step limit.")
    finally:
        planner.close_plot()
