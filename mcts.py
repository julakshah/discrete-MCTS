from __future__ import annotations
import heapq
import copy
from dataclasses import dataclass
import numpy as np

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

def all_neighbors(pos):
    return [(x[0] + pos[0],x[1] + pos[1]) for x in DIRECTIONS]

@dataclass
class Node:
    """
    Node to construct tree with for MCTS
    """
    state: tuple
    parent: Node | None = None
    children: dict = None
    action: int | None = None
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list = None

def mcts(start, goal, env, c=1.0):
    start_node = Node(parent=None, children={}, state=start, untried_actions=all_neighbors(start))
    
    max_mcts_iters = 100
    for i in range(max_mcts_iters):
        #print(f"Starting iteration {i}")
        # Always start at the root node
        current_node = start_node
        current_state = start
        current_env = copy.deepcopy(env)

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

        if len(current_node.untried_actions) > 0:
            # We have unexpanded actions --- create a node for these
            next_pos_idx = np.random.choice(np.arange(len(current_node.untried_actions)))
            next_pos = current_node.untried_actions.pop(next_pos_idx)
            action = DIRECTION_LABELS[(next_pos[0]-current_state[0],next_pos[1]-current_state[1])]
            res = current_env.step(actor_action=action)

            child = Node(
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
                reward = 100
            else:
            # Rollout
                rollout_done = 0
                max_iters = 50
                rollout_state = current_env.actor_pos
                for i in range(max_iters):
                    next_pos = all_neighbors(rollout_state)[np.random.choice(np.arange(len(all_neighbors(rollout_state))))]
                    action = DIRECTION_LABELS[(next_pos[0]-rollout_state[0],next_pos[1]-rollout_state[1])]
                    res = current_env.step(actor_action=action)
                    rollout_state = current_env.actor_pos
                    if res["reached_goal"]:
                        reward = 100
                        break

                    if res["collision"]:
                        reward = 0
                        break

                    if res.get("no_path"):
                        reward = 0
                        break
            
            if reward is None:
                # Chebyshev distance value to incentivize progress in rollouts
                reward = 10 - 0.1*max(abs(current_env.goal[1]-rollout_state[1]),abs(current_env.goal[0]-rollout_state[0]))
            
            child.visits += 1
            child.total_reward += reward
            
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

            