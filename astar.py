
import heapq
import numpy as np

def astar(start, goal, neighbors, cost, heuristic):
    """
    neighbors(node): iterable of adjacent nodes
    cost(a, b): cost of moving from a to b
    heuristic(node, goal): estimated remaining cost
    """
    queue = [(heuristic(start, goal), start)]
    parents = {start: None}
    g_score = {start: 0}

    while queue:
        _, current = heapq.heappop(queue)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parents[current]
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g = g_score[current] + cost(current, neighbor)

            if tentative_g < g_score.get(neighbor, float("inf")):
                parents[neighbor] = current
                g_score[neighbor] = tentative_g

                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(queue, (f, neighbor))

    return None
