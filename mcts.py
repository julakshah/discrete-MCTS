from __future__ import annotations
import heapq
import copy
import random
from dataclasses import dataclass, field
import numpy as np
import math

DIRECTIONS = np.array([
    (0, -1),   # N
    (1, -1),   # NE
    (1, 0),    # E
    (1, 1),    # SE
    (0, 1),    # S
    (-1, 1),   # SW
    (-1, 0),   # W
    (-1, -1),  # NW
])

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

# Enums for node types
TYPE_STATE = 0
TYPE_CHANCE = 1

OBSTACLE_COST = 20.0

def all_neighbors(pos):
    return [(x[0] + pos[0],x[1] + pos[1]) for x in DIRECTIONS]

@dataclass(frozen=True)
class ObstacleState:
    pos: tuple[int, int]
    height: int
    width: int
    movement_probs: tuple[float, ...]
    stay_prob: float

@dataclass(frozen=True)
class State:
    actor_pos: tuple[int, int]
    obstacles: tuple[ObstacleState, ...]

@dataclass
class OpenLoopNode:
    """
    Node to construct tree with for MCTS
    """
    state: tuple
    parent: OpenLoopNode | None = None
    children: dict = None
    action: int | None = None
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list = None

@dataclass
class StateNode:
    state: State
    parent: "ChanceNode | None" = None

    # Action -> ChanceNode
    children: dict[int, "ChanceNode"] = field(default_factory=dict)

    visits: int = 0
    total_reward: float = 0.0
    num_collide: int = 0
    num_finish: int = 0

@dataclass
class ChanceNode:
    action: int
    parent: StateNode

    # Resulting State -> StateNode
    children: dict[State, StateNode] = field(default_factory=dict)

    visits: int = 0
    total_reward: float = 0.0
    num_collide: int = 0
    num_finish: int = 0

def get_state(grid):
    # Preserve obstacle order. Grid.step() samples and moves obstacles by
    # list index, and different obstacles can have different policies.
    # Sorting by position aliases states with different transition dynamics.
    obs_tuple = tuple(ObstacleState(
        pos=(obs.x,obs.y),
        width=obs.width,
        height=obs.height,
        movement_probs=tuple(float(prob) for prob in obs.movement_probs),
        stay_prob=float(obs.stay_prob),
    ) for obs in grid.obstacles)

    return State(
        actor_pos=(grid.actor_pos[0],grid.actor_pos[1]),
        obstacles=obs_tuple,
    )

def ucb(parent, child, c):
    if child.visits == 0:
        return float("inf")

    return (
        child.total_reward / child.visits
        + c * np.sqrt(np.log(max(1, parent.visits)) / child.visits)
    )

def octile_distance(pos, goal):
    dx = abs(pos[0] - goal[0])
    dy = abs(pos[1] - goal[1])

    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


# Configuration is passed explicitly through both MCTS entry points.
# Rewards are maximized; all cost-based evaluations return negative costs.
from typing import Literal

RolloutPolicy = Literal['random', 'greedy', 'epsilon_greedy', 'safe_random']
RewardHeuristic = Literal['absorbing_failure', 'cost_to_go', 'progress',"legacy_progress"]


def legal_actions(env):
    """Return movement actions that are legal in the current Grid state."""
    x, y = env.actor_pos
    return [
        action
        for action, (dx, dy) in enumerate(DIRECTIONS)
        if env.can_move((x, y), (x + int(dx), y + int(dy)))
    ]


def choose_rollout_action(env, goal, policy='safe_random', epsilon=0.2):
    if policy == 'random':
        return random.randrange(8)
    if policy == 'safe_random':
        candidates = legal_actions(env)
        return random.choice(candidates) if candidates else None
    if policy not in ('greedy', 'epsilon_greedy'):
        raise ValueError(f'Unknown rollout policy: {policy}')
    if policy == 'epsilon_greedy' and random.random() < epsilon:
        return random.randrange(8)
    x, y = env.actor_pos
    dx = (goal[0] > x) - (goal[0] < x)
    dy = (goal[1] > y) - (goal[1] < y)
    return env.DIRECTION_LABELS.get((dx, dy), random.randrange(8))


def evaluate_rollout(start_pos, final_pos, goal, path_cost, *,
                     reached_goal=False, collision=False, no_path=False,
                     depth=0, horizon=50, reward_heuristic='absorbing_failure',
                     collision_penalty=20.0, failure_cost_per_step=1.0,
                     progress_weight=0.5, path_cost_weight=1.0):
    """Evaluate a complete root-to-terminal/leaf simulation exactly once.

    depth counts executed grid.step calls from the root, including a terminal
    collision step. horizon is the TOTAL root-relative horizon.
    """
    remaining = max(0, horizon - depth)
    travel = path_cost_weight * path_cost
    if reward_heuristic == 'absorbing_failure':
        if collision:
            return -(travel + collision_penalty + failure_cost_per_step * remaining)
        if reached_goal:
            return -travel
        return -(travel + octile_distance(final_pos, goal))
    if reward_heuristic == "legacy_progress":
        if reached_goal:
            return 1.0 - 0.001 * path_cost

        if collision:
            return -1.0 - 0.001 * path_cost

        # Match the original rollout behavior: a dead end receives only the
        # small accumulated path-cost penalty, rather than being credited for
        # geometric progress made before getting stuck.
        if no_path:
            return -0.001 * path_cost

        initial_distance = math.dist(start_pos, goal)
        final_distance = math.dist(final_pos, goal)

        return (
            0.5 * (initial_distance - final_distance)
            / max(initial_distance, 1.0)
            - 0.001 * path_cost
        )
    if reward_heuristic == 'cost_to_go':
        if collision:
            return -(travel + collision_penalty + octile_distance(final_pos, goal))
        return -(travel + (0 if reached_goal else octile_distance(final_pos, goal)))
    if reward_heuristic == 'progress':
        initial = math.dist(start_pos, goal)
        final = math.dist(final_pos, goal)
        progress = progress_weight * (initial - final) / max(initial, 1.0)
        if reached_goal:
            return 1.0 - travel
        if collision:
            return -collision_penalty - travel
        return progress - travel
    raise ValueError(f'Unknown reward heuristic: {reward_heuristic}')


def rollout(env, goal, path_cost, *, horizon, depth, rollout_policy,
            epsilon, start_pos):
    """Simulate only the steps remaining in the root-relative horizon."""
    result = None
    for _ in range(max(0, horizon - depth)):
        action = choose_rollout_action(env, goal, rollout_policy, epsilon)
        if action is None:
            result = {'no_path': True}
            break
        before = env.actor_pos
        result = env.step(actor_action=int(action))
        depth += 1
        path_cost += ACTION_COSTS[action]

        if result.get('collision') or result.get('reached_goal') or result.get('no_path'):
            break
    return path_cost, depth, result or {}


def _terminal_reward(env, start_pos, goal, path_cost, depth, horizon,
                     result, reward_kwargs):
    return evaluate_rollout(
        start_pos, env.actor_pos, goal, path_cost,
        reached_goal=result.get('reached_goal', False),
        collision=result.get('collision', False),
        no_path=result.get('no_path', False),
        depth=depth, horizon=horizon, **reward_kwargs,
    )


def _mcts(env, *, open_loop, c, max_iters, rollout_horizon,
          rollout_policy, epsilon, reward_heuristic, collision_penalty,
          failure_cost_per_step, progress_weight, path_cost_weight,
          verbose):
    if max_iters < 1 or rollout_horizon < 1:
        raise ValueError('max_iters and rollout_horizon must be positive')
    if rollout_policy not in ('random', 'greedy', 'epsilon_greedy', 'safe_random'):
        raise ValueError(f'Unknown rollout policy: {rollout_policy}')
    if reward_heuristic not in (
        "absorbing_failure",
        "cost_to_go",
        "progress",
        "legacy_progress",  # NEW
    ):
        raise ValueError(f"Unknown reward heuristic: {reward_heuristic}")
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be between 0 and 1')

    reward_kwargs = dict(
        reward_heuristic=reward_heuristic,
        collision_penalty=collision_penalty,
        failure_cost_per_step=failure_cost_per_step,
        progress_weight=progress_weight,
        path_cost_weight=path_cost_weight,
    )
    if open_loop:
        root = OpenLoopNode(
            state=env.actor_pos,
            children={},
            untried_actions=legal_actions(env),
        )
    else:
        root = StateNode(state=get_state(env))

    for _ in range(max_iters):
        sim = env.clone()
        start_pos = sim.actor_pos
        current = root
        path = [root]
        depth = 0
        path_cost = 0.0
        result = {}

        while depth < rollout_horizon:
            available_actions = legal_actions(sim)
            if not available_actions:
                result = {'no_path': True}
                break

            if open_loop:
                untried = [
                    action
                    for action in current.untried_actions
                    if action in available_actions
                ]
                if untried:
                    action = random.choice(untried)
                    current.untried_actions.remove(action)
                    child = OpenLoopNode(
                        state=None,
                        parent=current,
                        children={},
                        action=action,
                        untried_actions=[],
                    )
                    current.children[action] = child
                    expanding = True
                else:
                    candidates = [
                        child
                        for action, child in current.children.items()
                        if action in available_actions
                    ]
                    if not candidates:
                        result = {'no_path': True}
                        break
                    child = max(candidates, key=lambda n: ucb(current, n, c))
                    action = child.action
                    expanding = False
                before = sim.actor_pos
                result = sim.step(actor_action=int(action))
                depth += 1

                if open_loop and expanding:
                    child.untried_actions = legal_actions(sim)

                # The action was geometrically legal when selected. A moving
                # obstacle can still cause a collision during this tick.
                path_cost += ACTION_COSTS[action]

                path.append(child)
                current = child
            else:
                untried = [
                    action
                    for action in available_actions
                    if action not in current.children
                ]
                if untried:
                    action = random.choice(untried)
                    chance = ChanceNode(action=action, parent=current)
                    current.children[action] = chance
                else:
                    candidates = [
                        chance
                        for action, chance in current.children.items()
                        if action in available_actions
                    ]
                    if not candidates:
                        result = {'no_path': True}
                        break
                    chance = max(candidates, key=lambda n: ucb(current, n, c))
                    action = chance.action
                path.append(chance)
                before = sim.actor_pos
                result = sim.step(actor_action=int(action))
                depth += 1

                # The action was geometrically legal when selected. A moving
                # obstacle can still cause a collision during this tick.
                path_cost += ACTION_COSTS[action]

                state = get_state(sim)
                expanding = state not in chance.children
                if expanding:
                    chance.children[state] = StateNode(state=state, parent=chance)
                current = chance.children[state]
                path.append(current)

            if result.get('collision') or result.get('reached_goal') or result.get('no_path'):
                break
            if expanding:
                break

        if not (result.get('collision') or result.get('reached_goal') or result.get('no_path')):
            path_cost, depth, result = rollout(
                sim, sim.goal, path_cost, horizon=rollout_horizon,
                depth=depth, rollout_policy=rollout_policy,
                epsilon=epsilon, start_pos=start_pos,
            )

        reward = _terminal_reward(sim, start_pos, sim.goal, path_cost,
                                  depth, rollout_horizon, result, reward_kwargs)
        collided = int(result.get('collision', False))
        finished = int(result.get('reached_goal', False))
        for node in reversed(path):
            node.visits += 1
            node.total_reward += reward
            if hasattr(node, 'num_collide'):
                node.num_collide += collided
                node.num_finish += finished

    if not root.children:
        return None
    best = max(root.children.values(), key=lambda n: n.visits)
    if verbose:
        for action, child in sorted(root.children.items()):
            print(f'Action {action} ({DIRECTIONS[action]}): '
                  f'visits={child.visits}, '
                  f'collisions={getattr(child, "num_collide", "n/a")}, '
                  f'successes={getattr(child, "num_finish", "n/a")}, '
                  f'mean_reward={child.total_reward / max(1, child.visits):.5f}')
    return best.action


def mcts_closed_loop(start, goal, env, c=1.0, max_iters=1000,
                     rollout_horizon=50, *, rollout_policy='safe_random',
                     epsilon=0.2, reward_heuristic='legacy_progress',
                     collision_penalty=20.0, failure_cost_per_step=1.0,
                     progress_weight=0.5, path_cost_weight=1.0, verbose=False):
    return _mcts(env, open_loop=False, c=c, max_iters=max_iters,
                 rollout_horizon=rollout_horizon, rollout_policy=rollout_policy,
                 epsilon=epsilon, reward_heuristic=reward_heuristic,
                 collision_penalty=collision_penalty,
                 failure_cost_per_step=failure_cost_per_step,
                 progress_weight=progress_weight,
                 path_cost_weight=path_cost_weight, verbose=verbose)


def mcts_open_loop(start, goal, env, c=1.0, max_iters=1000,
                   rollout_horizon=50, *, rollout_policy='safe_random',
                   epsilon=0.2, reward_heuristic='legacy_progress',
                   collision_penalty=20.0, failure_cost_per_step=1.0,
                   progress_weight=0.5, path_cost_weight=1.0, verbose=False):
    return _mcts(env, open_loop=True, c=c, max_iters=max_iters,
                 rollout_horizon=rollout_horizon, rollout_policy=rollout_policy,
                 epsilon=epsilon, reward_heuristic=reward_heuristic,
                 collision_penalty=collision_penalty,
                 failure_cost_per_step=failure_cost_per_step,
                 progress_weight=progress_weight,
                 path_cost_weight=path_cost_weight, verbose=verbose)
