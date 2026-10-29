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


class LRTDP(Planner):
    """
    Sampled-transition, labeled RTDP for the existing Grid environment.

    This implementation:
    - Uses the full robot and obstacle configuration as the state.
    - Initializes state values with an octile-distance heuristic.
    - Runs labeled RTDP trials using sampled environment transitions.
    - Uses Bellman backups to update state values.
    - Labels states solved when sampled Bellman residuals are small and all
      sampled greedy successors are solved.
    - Rejects actions that collide in any sampled outcome.
    - Uses the existing Planner interface for execution and metrics.

    Actions are filtered against all possible one-tick obstacle moves.
    Longer-horizon values, labels, and trials still use sampled transitions,
    so solved labels are practical approximations rather than exact proofs.
    """

    def __init__(
        self,
        grid,
        max_trials=30,
        max_trial_steps=20,
        transition_samples=3,
        epsilon=0.1,
        verbose=True,
    ):
        super().__init__(grid)

        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        if max_trial_steps < 1:
            raise ValueError("max_trial_steps must be positive")
        if transition_samples < 1:
            raise ValueError("transition_samples must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.plot_label = "LRTDP"
        self.max_trials = max_trials
        self.max_trial_steps = max_trial_steps
        self.transition_samples = transition_samples
        self.epsilon = epsilon
        self.verbose = verbose

        # Keep planning randomness reproducible without advancing the live grid.
        self.simulation_rng = np.random.default_rng()
        self.simulation_rng.bit_generator.state = copy.deepcopy(
            grid.rng.bit_generator.state
        )

        # V(s) is estimated cost from a complete state to the goal.
        self.value_function = {}
        # Approximate LRTDP labels.  Labels are scoped to this planner/grid.
        self.solved = set()

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

    def q_from_outcomes(self, action, outcomes):
        """Estimate Q from an already sampled set of successor outcomes."""
        total_cost = 0.0
        immediate_cost = self.action_cost(action)

        for next_env, result in outcomes:
            if result["collision"]:
                return float("inf")

            if result["reached_goal"]:
                total_cost += immediate_cost
            else:
                total_cost += immediate_cost + self.initialize_value(next_env)

        return total_cost / len(outcomes)

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
        outcomes = [
            self.sample_transition(env, action)
            for _ in range(self.transition_samples)
        ]
        return self.q_from_outcomes(action, outcomes)

    def best_action_evaluation(self, env, transition_cache=None):
        """Return the best action, its Q-value, and sampled outcomes.

        A transition cache is used during CHECK_SOLVED.  It reuses the same
        sampled successors while values are updated, avoiding repeated grid
        clones without pretending that the samples are exact probabilities.
        """
        state = self.get_state(env)
        best_action = None
        best_q = float("inf")
        best_outcomes = []

        for action in self.legal_actions(env):
            cache_key = (state, action)
            if transition_cache is not None and cache_key in transition_cache:
                outcomes = transition_cache[cache_key]
            else:
                outcomes = [
                    self.sample_transition(env, action)
                    for _ in range(self.transition_samples)
                ]
                if transition_cache is not None:
                    transition_cache[cache_key] = outcomes

            q = self.q_from_outcomes(action, outcomes)
            if q < best_q:
                best_q = q
                best_action = action
                best_outcomes = outcomes

        return best_action, best_q, best_outcomes

    def greedy_action(self, env):
        """Pick the action that minimizes the estimated expected Q-value."""
        best_action, best_q, _ = self.best_action_evaluation(env)
        return best_action, best_q

    def bellman_backup(self, env, transition_cache=None, return_outcomes=False):
        """Update V(s) using the current estimated action values."""
        state = self.get_state(env)

        if self.is_goal(env):
            self.value_function[state] = 0.0
            self.solved.add(state)
            if return_outcomes:
                return None, float("inf"), []
            return None

        self.initialize_value(env)

        best_action, best_q, best_outcomes = self.best_action_evaluation(
            env,
            transition_cache=transition_cache,
        )

        if math.isfinite(best_q):
            self.value_function[state] = best_q

        if return_outcomes:
            return best_action, best_q, best_outcomes
        return best_action

    def check_solved(self, env, transition_cache=None):
        """Run a bounded, one-step sampled LRTDP solved check.

        Exact LRTDP recursively checks the complete greedy envelope.  That
        can explode for this environment because every obstacle configuration
        is a distinct state.  This practical variant only labels a state when
        its sampled residual is small and every sampled greedy successor is
        already solved or is the goal.  Trials propagate these labels backward
        without exploring an unbounded extra graph.
        """
        state = self.get_state(env)

        if self.is_goal(env):
            self.value_function[state] = 0.0
            self.solved.add(state)
            return True

        if state in self.solved:
            return True

        if transition_cache is None:
            transition_cache = {}

        old_value = self.initialize_value(env)
        action, best_q, outcomes = self.best_action_evaluation(
            env,
            transition_cache=transition_cache,
        )

        if action is None or not math.isfinite(best_q):
            return False

        self.value_function[state] = best_q
        if abs(old_value - best_q) > self.epsilon:
            return False

        successors_solved = True
        for next_env, result in outcomes:
            if result["collision"]:
                successors_solved = False
                break
            if result["reached_goal"]:
                continue
            if self.get_state(next_env) not in self.solved:
                successors_solved = False
                break

        if successors_solved:
            self.solved.add(state)
            return True
        return False

    def trial(self, initial_env):
        """
        Start a trial from the current environment configuration.

        Each step chooses a greedy action, performs a Bellman backup, and
        samples the resulting transition.  WAIT_ACTION advances obstacles
        without moving the actor.
        """
        env = initial_env.clone(include_density=False)
        visited = []
        completed = False

        for _ in range(self.max_trial_steps):
            if self.is_goal(env):
                self.solved.add(self.get_state(env))
                completed = True
                break

            state = self.get_state(env)
            if state in self.solved:
                completed = True
                break

            # sample_transition clones env and never mutates it, so retaining
            # this snapshot directly avoids one extra clone per trial step.
            visited.append(env)

            action, _, outcomes = self.bellman_backup(
                env,
                return_outcomes=True,
            )

            if action is None:
                # No sampled-safe movement action exists.
                # Do not deliberately move into a known obstacle.
                break

            # The sampled outcomes already contain independent transitions
            # for this action.  Reuse one instead of cloning and stepping a
            # second time for the trial, which is a major RTDP cost.
            next_env, result = outcomes[
                self.simulation_rng.integers(len(outcomes))
            ]

            if result["collision"]:
                break

            env = next_env

            if result["reached_goal"]:
                self.value_function[self.get_state(env)] = 0.0
                self.solved.add(self.get_state(env))
                completed = True
                break

        # Check the greedy envelope from the end of the trial backward.  This
        # is the practical sampled analogue of LRTDP's CHECK_SOLVED pass.
        transition_cache = {}
        for visited_env in reversed(visited):
            if not self.check_solved(
                visited_env,
                transition_cache=transition_cache,
            ):
                break

        return completed

    def LRTDP(self, initial_env):
        """
        Run repeated trials and return the greedy action at the
        initial state, based on the learned value function.
        """
        for _ in range(self.max_trials):
            if self.get_state(initial_env) in self.solved:
                break
            self.trial(initial_env)
            if self.get_state(initial_env) in self.solved:
                break

        best_action, _ = self.greedy_action(initial_env)
        return best_action

    # Preserve the old method name for callers using the original RTDP class.
    RTDP = LRTDP

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
        action = self.LRTDP(self.grid)
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


RTDP = LRTDP


if __name__ == "__main__":
    from grid import dynamic_sparse_grid, dynamic_wall_maze_grid, dynamic_dense_grid, dynamic_wall_maze_one_section_grid

    grid = dynamic_sparse_grid(
        width=50,
        height=50,
        start=(0,0),
        goal=(20,20),
        seed=0,
        start_clearance=2,
    )
    # grid = dynamic_wall_maze_grid(
    # width=30,
    # height=30,
    # seed=0,
    # start_clearance=5)

    grid = dynamic_sparse_grid(width=50, height=50)

    planner = LRTDP(
        grid,
        # max_trials=20,
        # max_trial_steps=100,
        max_trials=10,
        max_trial_steps=20,
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
