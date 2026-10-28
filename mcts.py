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

def all_neighbors(pos):
    return [(x[0] + pos[0],x[1] + pos[1]) for x in DIRECTIONS]

@dataclass(frozen=True)
class ObstacleState:
    pos: tuple[int, int]
    height: int
    width: int

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
    # Count all obstacle positions 
    obs_tuple = tuple(ObstacleState(
        pos=(obs.x,obs.y),
        width=obs.width,
        height=obs.height
    ) for obs in grid.obstacles)

    # Sort obstacles by center position (ensures order of obstacles doesn't matter)
    obstacles = tuple(sorted(obs_tuple, key=lambda o: (
        o.pos[0],
        o.pos[1],
        o.width,
        o.height,
    )))

    return State(
        actor_pos=(grid.actor_pos[0],grid.actor_pos[1]),
        obstacles=obstacles
    )

def ucb(parent, child, c):
    if child.visits == 0:
        return float("inf")

    return (
        child.total_reward / child.visits
        + c * np.sqrt(np.log(max(1, parent.visits)) / child.visits)
    )

def choose_rollout_action(env, goal, epsilon=0.2):
    if random.random() < epsilon:
        return random.randrange(8)

    x, y = env.actor_pos
    gx, gy = goal

    dx = (gx > x) - (gx < x)
    dy = (gy > y) - (gy < y)

    if dx == 0 and dy == 0:
        return random.randrange(8)

    return env.DIRECTION_LABELS[(dx, dy)]

def choose_rollout_action_safe(env):
    x, y = env.actor_pos
    safe_actions = []

    for action, (dx, dy) in enumerate(DIRECTIONS):
        new_pos = (x + int(dx), y + int(dy))

        if not env.in_bounds(new_pos):
            continue

        if env.is_occupied(new_pos):
            continue

        safe_actions.append(action)

    if safe_actions:
        return random.choice(safe_actions)

    return None

def traverse(root, env, c=1.0, max_depth=100):
    current = root
    path = [current]
    cost = 0
    depth = 0

    while depth < max_depth:
        #print(f"depth: {depth}")
        # Check for terminal states here (goal, collision, horizon, etc.)

        # Find actions that haven't been expanded
        available_actions = list(DIRECTION_LABELS.values())

        untried = [
            a for a in available_actions
            if a not in current.children
        ]

        if untried:
            #print(f"trying something in untried")
            # EXPANSION: try a new action
            action = random.choice(untried)

            chance_node = ChanceNode(
                action=action,
                parent=current,
            )
            current.children[action] = chance_node
            path.append(chance_node)

        else:
            #print(f"selection")
            # SELECTION: choose an existing action using UCT
            # chance_node = max(
            #     current.children.values(),
            #     key=lambda n: ucb(current, n, c),
            # )
            # action = chance_node.action
            # path.append(chance_node)

            log_parent = math.log(max(1, current.visits))

            chance_node = max(
                current.children.values(),
                key=lambda n: (
                    float("inf")
                    if n.visits == 0
                    else n.total_reward / n.visits
                    + c * math.sqrt(log_parent / n.visits)
                ),
            )

            action = chance_node.action
            path.append(chance_node)

        # CHANCE: sample what actually happens
        #print(f"action {action}")
        res = env.step(actor_action=action)
        cost += ACTION_COSTS[action]
        depth += 1
        next_state = get_state(env)

        if next_state not in chance_node.children:
            #print(f"adding state node")
            next_node = StateNode(
                state=next_state,
                parent=chance_node,
            )
            chance_node.children[next_state] = next_node
            expanded = True
        else:
            #print(f"go to state node")
            next_node = chance_node.children[next_state]
            expanded = False

        path.append(next_node)

        if res["reached_goal"] or res["collision"] or res.get("no_path"):
            return next_node, path, cost, depth, res

        if expanded:
            return next_node, path, cost, depth, None

        # Already explored this outcome: keep traversing
        current = next_node
    
    return current, path, cost, depth, None

def rollout(current_env, goal, current_path_cost, node_start, rollout_horizon):
    # Rollout step from current env state
    reward = None
    if rollout_horizon < 1:
        # We have no more horizon to rollout
        # Reward is just the penalty of the cost length 
        reward = -0.001 * current_path_cost
        return reward, False, False

    max_iters = rollout_horizon
    rollout_state = current_env.actor_pos
    for i in range(max_iters):
        # Choose rollout action
        # next_pos = all_neighbors(rollout_state)[np.random.choice(np.arange(len(all_neighbors(rollout_state))))]
        # action = DIRECTION_LABELS[(next_pos[0]-rollout_state[0],next_pos[1]-rollout_state[1])]

        # action = choose_rollout_action(current_env,current_env.goal,epsilon=1.0)
        action = choose_rollout_action_safe(current_env)
        collide = False
        finish = False
        
        #print(f"started stepping env")
        res = current_env.step(actor_action=action)
        #print(f"finished stepping env")
        current_path_cost += ACTION_COSTS[action]
        rollout_state = current_env.actor_pos
        if res["reached_goal"]:
            #reward = 0.5 + 0.5 / (1 + current_path_cost)
            reward = 1 - 0.001 * current_path_cost
            finish = True
            break

        if res["collision"]:
            #reward = -0.5 - 0.001 * current_path_cost
            reward = -1 - 0.001 * current_path_cost
            collide = True
            break

        if res.get("no_path"):
            reward = -0.001 * current_path_cost
            break

    if reward is None:
        # Euclidean distance value to incentivize progress in rollouts
        # reward = 0.5 / (1 + current_path_cost + np.sqrt(
        #     (current_env.actor_pos[0]-current_env.goal[0])**2 + (current_env.actor_pos[1]-current_env.goal[1])**2
        # ))
        initial_distance = math.dist(node_start, goal)
        final_distance = math.dist(current_env.actor_pos, goal)

        reward = 0.5 * (initial_distance - final_distance) / max(initial_distance, 1)
        reward -= 0.001 * current_path_cost
        # reward = initial_distance - final_distance
    
    return reward, collide, finish

def mcts_closed_loop(start, goal, env, c=1.0, max_iters=1000, rollout_horizon=50):
    root = StateNode(state=get_state(env))

    for i in range(max_iters):
        # if i % 1 == 0:
        #     print(f"iter {i} ({100*i/max_iters:.1f} done)")
        # Each iteration starts with a fresh copy of the environment
        current_env = env.clone()
        node_start = current_env.actor_pos

        # Selection + expansion step, starting from root node
        leaf, path, accum_cost, depth, step_result = traverse(root, current_env, c)
        #print("finished traverse")
        collide = False
        finish = False
        if step_result is not None:
            #print("step result is not none")
            if step_result["collision"]:
                reward =  -1 - 0.001 * accum_cost
                collide = True
            elif step_result["reached_goal"]:
                reward = 1 - 0.001 * accum_cost
                finish = True
            else:
                reward = 0.0
        else:
            #print(f"starting rollout with depth {max(0,rollout_horizon-depth)}")
            reward, collide, finish = rollout(
                current_env,
                goal,
                accum_cost,
                node_start=node_start,
                rollout_horizon=max(0,rollout_horizon-depth),
            )
        #print("finished rollout stage")
        # 3. Backpropagation
        for node in reversed(path):
            node.visits += 1
            node.total_reward += reward
            node.num_collide += int(collide)
            node.num_finish += int(finish)

    # Choose the most visited action from the root
    best_action = max(
        root.children.values(),
        key=lambda n: n.visits
    ).action

    # best_action = max(
    #     root.children.values(),
    #     key=lambda n: n.total_reward / n.visits
    # ).action

    for action, child in sorted(root.children.items()):
        print(
            f"Action {action} ({DIRECTIONS[action]}): "
            f"visits={child.visits}, Collisions={child.num_collide}, Successes={child.num_finish}, "
            f"mean_reward={child.total_reward / max(child.visits, 1):.5f}"
        )

    return best_action

def mcts_open_loop(start, goal, env, c=1.0, max_iters=1000, rollout_horizon=50):
    start_node = OpenLoopNode(parent=None, children={}, state=start, untried_actions=all_neighbors(start))
    
    for i in range(max_iters):
        #print(f"Starting iteration {i}")
        # Always start at the root node
        current_node = start_node
        current_state = start
        current_env = env.clone()
        node_start = current_env.actor_pos
        current_path_cost = 0

        while not current_node.untried_actions and current_node.children:
            # We have no unexpanded actions; step down the tree
            # Compute the UCB reward for each node
            best_reward,best_index = -np.inf, None
            for idx,n in current_node.children.items():
                rew = n.total_reward / n.visits + c * np.sqrt(np.log(current_node.visits)/n.visits)
                if rew > best_reward:
                    best_reward = rew
                    best_index = idx
            
            current_node = current_node.children[best_index]
            current_state = current_node.state
            res = current_env.step(actor_action=current_node.action)
            current_path_cost += ACTION_COSTS[current_node.action]

        if len(current_node.untried_actions) > 0:
            # We have unexpanded actions --- create a node for these
            next_pos_idx = np.random.choice(np.arange(len(current_node.untried_actions)))
            next_pos = current_node.untried_actions.pop(next_pos_idx)
            action = DIRECTION_LABELS[(next_pos[0]-current_state[0],next_pos[1]-current_state[1])]
            res = current_env.step(actor_action=action)
            current_path_cost += ACTION_COSTS[action]

            child = OpenLoopNode(
                parent=current_node,
                children={},
                state=current_env.actor_pos,
                action=action,
                untried_actions=all_neighbors(current_env.actor_pos)
            )
            current_node.children[action] = child

            reward = None
            if res["collision"]:
                reward = 0
            elif res["reached_goal"]:
                reward = 1
            else:
                # Rollout
                reward, collide, finish = rollout(current_env=current_env,goal=current_env.goal,current_path_cost=current_path_cost,node_start=node_start,rollout_horizon=rollout_horizon)
            
            # backprop the reward up the tree
            n = child
            while n is not None:
                n.visits += 1
                n.total_reward += reward
                n = n.parent

    # MCTS is done, choose the best action
    best_index = max(
        start_node.children,
        key=lambda action: start_node.children[action].total_reward / start_node.children[action].visits
    )
    return best_index

            