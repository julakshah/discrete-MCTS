import math
import heapq
from grid import Grid

# Directions as (dx, dy)
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


def reconstruct_path(parents, end):
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = parents[current]
    return path[::-1]


def astar(grid: Grid, start, end, heuristic):
    """Return the path and the number of neighbor checks during the search."""
    g = {start: 0}
    open = []
    closed = set()
    nodes_visited = 0
    parents = {start: None}

    heapq.heappush(open, (heuristic(start, end), start))

    while open:
        _, curr_node = heapq.heappop(open)

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

            new_g = dg + g[curr_node]

            if new_g < g.get(neighbor, float("inf")):
                g[neighbor] = new_g
                parents[neighbor] = curr_node

                f = new_g + heuristic(neighbor, end)
                heapq.heappush(open, (f, neighbor))

    return None, nodes_visited


def dynamic_astar(grid: Grid, start, end, heuristic, density_constant=50.0):
    """Return the path and the number of neighbor checks during the search."""
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

    # Keep track of the timestep for dynamic obstacles
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

            neighbor_timestep = timestep + 1
            if neighbor_timestep >= len(density_grids):
                prediction_grid.diffuse_step()
                density_grids.append(prediction_grid.density.copy())

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
