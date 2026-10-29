import math
import heapq
from grid import Grid

# Costs for moving in each direction based on distance
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
    """
    Octile distance heuristic for A*.

    Octile distance is meant for grids that allow diagonal movement.

    Args:
        start (tuple): The starting coordinates (x, y).
        end (tuple): The goal coordinates (x, y).

    Returns:
        float: The octile distance between start and end.
    """
    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def reconstruct_path(parents, end):
    """
    Reconstruct the path from the start to the end using the parent pointers.

    It works by traversing the graph from the end node back to the start node
    using the parent pointers.

    Args:
        parents (dict): A dictionary mapping each node to its parent.
        end (tuple): The end coordinates (x, y).

    Returns:
        list: The reconstructed path from start to end.
    """
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1]


def astar(grid: Grid, start, end, heuristic):
    """
    Return the path and the number of neighbor checks during the search.

    This implementation assumes the heuristic function is consistent, and therefore admissible
    meaning a closed node never needs to be re-opened since the shortest path to it has already been found.

    Args:
        grid (Grid): The grid to search on.
        start (tuple): The starting coordinates (x, y).
        end (tuple): The goal coordinates (x, y).
        heuristic (function): The heuristic function to use.

    Returns:
        tuple: A tuple containing the reconstructed path and the number of neighbor checks.
    """
    # Initialize the g-score, open list, closed set, and parent pointers.
    g = {start: 0}
    open = []
    closed = set()
    parents = {start: None}

    # Track nodes visited for comparison purposes
    nodes_visited = 0

    # Push the start node onto the open list with its f-score.
    heapq.heappush(open, (heuristic(start, end), start))

    while open:
        _, curr_node = heapq.heappop(open)

        # If the current node has already been processed, skip it because we are assuming
        # the heuristic is consistent, so a closed node doesn't need to be re-opened
        if curr_node in closed:
            continue

        # Mark the current node as processed by adding it to the closed set.
        closed.add(curr_node)

        # Check if we have reached the goal node
        if curr_node == end:
            # If so, reconstruct the path and return it along with the number of neighbor checks
            return reconstruct_path(parents, curr_node), nodes_visited

        # Explore the neighbors of the current node
        for neighbor in grid.neighbors(curr_node):
            nodes_visited += 1

            # If the neighbor has already been visited, skip it
            if neighbor in closed:
                continue

            # Compute the cost to move from the current node to the neighbor, to update g(n)
            dx, dy = neighbor[0] - curr_node[0], neighbor[1] - curr_node[1]
            dg = COSTS[(dx, dy)]

            # Calculate g(n) for the neighbor node
            new_g = dg + g[curr_node]

            # If we haven't seen this node before, then process it
            # If we have seen this node before, only process it if the
            # new g(n) is better than the previously found g(n) for this neighbor
            # meaning we have found a better path to it
            if new_g < g.get(neighbor, float("inf")):
                # Update the g(n) and parent pointer
                g[neighbor] = new_g
                parents[neighbor] = curr_node

                # Calculate the f(n) cost for the neighbor and add it to the priority queue
                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open, (f, neighbor))

    # If we have traversed the whole graph without finding the goal, then return none
    return None, nodes_visited


def dynamic_astar(grid: Grid, start, end, heuristic, density_constant=50.0):
    """
    Return the path and the number of neighbor checks during the search.

    Args:
        grid (Grid): The grid on which to perform the search.
        start: The starting node.
        end: The goal node.
        heuristic: The heuristic function to estimate the cost from a node to the goal.
        density_constant (float): A constant to weight the influence of dynamic obstacle density.

    Returns:
        tuple: A tuple containing the path from start to end (or None if no path is found)
        and the number of neighbor checks performed.
    """
    g = {start: 0}
    open = []
    closed = set()
    nodes_visited = 0
    parents = {start: None}

    # Start each prediction from the current obstacles, and keep forecast
    # diffusion separate from the live simulation's density map.
    prediction_grid = grid.clone()
    prediction_grid.reset_density()
    density_grids = [prediction_grid.density.copy()]

    # Keep track of the timestep for dynamic obstacles, so that we know the
    # density grid that corresponds to when the node is reached
    heapq.heappush(open, (heuristic(start, end), start, 0))

    while open:
        _, curr_node, timestep = heapq.heappop(open)

        if curr_node in closed:
            continue

        closed.add(curr_node)
        if curr_node == end:
            return reconstruct_path(parents, curr_node), nodes_visited

        for neighbor in grid.neighbors(curr_node):
            nodes_visited += 1
            if neighbor in closed:
                continue

            dx, dy = neighbor[0] - curr_node[0], neighbor[1] - curr_node[1]
            dg = COSTS[(dx, dy)]

            # Calculate the timestep for reaching the neighbor node based on the current node's timestep
            neighbor_timestep = timestep + 1
            # If we haven't found the density grid for this timestep, then generate it
            if neighbor_timestep >= len(density_grids):
                prediction_grid.diffuse_step()
                density_grids.append(prediction_grid.density.copy())

            # Update g(n) with the the cost of moving to the neighbor and the weighted obstacle density
            new_g = (
                dg
                + g[curr_node]
                + density_constant
                * density_grids[neighbor_timestep][neighbor[1], neighbor[0]]
            )

            if new_g < g.get(neighbor, float("inf")):
                g[neighbor] = new_g
                parents[neighbor] = curr_node

                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open, (f, neighbor, neighbor_timestep))

    return None, nodes_visited
