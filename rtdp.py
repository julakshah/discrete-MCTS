states = {} # each state needs to store which cell its in and also its value
value_function
# V_s is estimated cost of going from state s to the goal, we are continuously updating this
def init...(goal_state) #with init or something like that
goal_state = goal_state

#idk what to do with heuristic, also don't know how to represent obstacles, maybe the cell just has infinite cost? 
#also want to implement stochastic Markov decision process version because there are dynamic obstacles, not just static

#haven't implemented the labeled yet
def RTDP(initial_state, goal_state):
    value_function = heuristic #initialize a value using heuristic

    while():#not sure what to put here, maybe count number of trials like in mcts and vary that?)
        trial(initial_state)
    return policy

def trial(initial_state):
    current_state = initial_state # start a new trial at initial state
    while (current_state != goal_state): #if we haven't reached the goal, keep going
        next_action = choose_next_action(current_state)#find cost of going to each one
        states[]= ... #bellman backup
        current_state = current_state + next_action #update current state based on which action you just took

def choose_next_action(current_state):
    # \[Q(s,a)=c(s,a)+\sum_{s'}P(s'|s,a)V(s')\]
    min_cost = inf
    for(all 8 directions):
        cost = ... #find lowest expected cost
        min_cost = min(min_cost, cost)
    return action which lead to mincost