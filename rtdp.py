import math

value_function = {} #each state needs to store which cell its in and also its value, this gets updated over time
# V_s is estimated cost of going from state s to the goal, we are continuously updating this
def __init__(self, start_state, goal_state, obstacles, max_trials=500): #with init or something like that
    self.start_state = start_state
    self.goal_state = goal_state
    self.obstacles = obstacles
    self.max_trails = max_trials
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

COSTS = {
    (0, -1): 1,
    (1, -1): math.sqrt(2),
    (1, 0): 1,
    (1, 1): math.sqrt(2),
    (0, 1): 1,
    (-1, 1): math.sqrt(2),
    (-1, 0): 1,
    (-1, -1): math.sqrt(2),
    (0, 0): 0,
}


def heuristic(start, end):
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
#idk what to do with heuristic, also don't know how to represent obstacles, maybe the cell just has infinite cost? 
#also want to implement stochastic Markov decision process version because there are dynamic obstacles, not just static

#haven't implemented the labeled yet
def RTDP(self, initial_state, goal_state):
    value_function = ...heuristic #initialize a value using heuristic
    trials = 0
    while(trials < self.max_trials):#not sure what to put here, maybe count number of trials like in mcts and vary that?)
        trial(initial_state)
        trials+=1
    return policy # really lost on what to do here

def trial(self, initial_state):
    current_state = initial_state # start a new trial at initial state
    while (current_state != self.goal_state): #if we haven't reached the goal, keep going
        next_action = greedy_action(current_state)#find cost of going to each one
        states[]= ... #bellman backup
        current_state = current_state + next_action #update current state based on which action you just took

def greedy_action(self, state):
    """Picks the action that minimizes the expected Q-value."""
        # \[Q(s,a)=c(s,a)+\sum_{s'}P(s'|s,a)V(s')\]

    best_action = None
    best_q = float('inf')
    for action in DIRECTIONS: #USE DIRECTIONS CORRECTLY 
        q = self.q_value(state, action)
        if q > best_q:
            best_q = q
            best_action = action
    return best_action, best_q
