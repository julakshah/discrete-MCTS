from collections import deque

def bfs(start, goal, neighbors):
    """
    Find the shortest path from start to goal.

    neighbors(state) returns an iterable of reachable states.
    Returns a list of states, or None if no path exists.
    """
    queue = deque([start])
    parents = {start: None}

    while queue:
        current = queue.popleft() # take a node in the queue

        if current == goal:
            path = []
            while current is not None: # walk up the tree until finding root node
                path.append(current)
                current = parents[current]
            return path[::-1] # correct order of path

        for neighbor in neighbors(current):
            if neighbor not in parents: # avoids re-walking same node if re-encountered
                parents[neighbor] = current
                queue.append(neighbor)

    return None
