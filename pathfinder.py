import sys 
import numpy as np 
from collections import deque
import heapq
import math

STUDENT_ID = 'a1826142'
DEGREE = 'UG'

def parse_map(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    rows, cols = map(int, lines[0].split())
    start = tuple(map(lambda x: int(x) - 1, lines[1].split()))
    end = tuple(map(lambda x: int(x) - 1, lines[2].split()))
    grid = np.array([list(map(lambda x: int(x) if x.isdigit() else x, line.split())) for line in lines[3:]])
    
    return rows, cols, start, end, grid

def bfs(rows, cols, start, end, grid):
    queue = deque([(start, [])])  
    visited = set()
    visit_count = np.zeros((rows, cols), dtype=int)
    first_visit = np.full((rows, cols), '.', dtype=object)
    last_visit = np.full((rows, cols), '.', dtype=object)
    
    step = 1
    while queue:
        (current, path) = queue.popleft()
        
        if current in visited:
            visit_count[current] += 1
            last_visit[current] = step
            step += 1
            continue
        
        visited.add(current)
        visit_count[current] = 1
        first_visit[current] = step
        last_visit[current] = step
        step += 1
        
        new_path = path + [current]
        
        if current == end:
            return new_path, visit_count, first_visit, last_visit
        
        i, j = current
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
            ni, nj = i + di, j + dj
            
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != 'X':  
                queue.append(((ni, nj), new_path))
    
    return None, visit_count, first_visit, last_visit


def visualize_path(grid, path, visit_count, first_visit, last_visit):
    grid_with_path = np.copy(grid).astype(str)
    
    for (i, j) in path:
        grid_with_path[i][j] = '*'
    
    print("path:")
    for row in grid_with_path:
        print(" ".join(row))

    print("\n#visits:")
    for i in range(len(visit_count)):
        print(" ".join(str(visit_count[i][j]) if visit_count[i][j] > 0 else ('X' if grid[i][j] == 'X' else '.') for j in range(len(visit_count[i]))))

    print("\nfirst visit:")
    for i in range(len(first_visit)):
        print(" ".join(str(first_visit[i][j]) if first_visit[i][j] != '.' else ('X' if grid[i][j] == 'X' else '.') for j in range(len(first_visit[i]))))

    print("\nlast visit:")
    for i in range(len(last_visit)):
        print(" ".join(str(last_visit[i][j]) if last_visit[i][j] != '.' else ('X' if grid[i][j] == 'X' else '.') for j in range(len(last_visit[i]))))


def terrain_cost(a, b, c, d, grid):
    elevation_a = int(grid[a][b]) if grid[a][b] != 'X' else float('inf')
    elevation_c = int(grid[c][d]) if grid[c][d] != 'X' else float('inf')

    if elevation_c - elevation_a > 0:  
        return 1 + (elevation_c - elevation_a)
    return 1

def ucs(rows, cols, start, end, grid):
    pq = [(0, start)]  
    heapq.heapify(pq)
    came_from = {start: None}
    cost_so_far = {start: 0}
    visit_count = np.zeros((rows, cols), dtype=int)
    first_visit = np.full((rows, cols), '.', dtype=object)
    last_visit = np.full((rows, cols), '.', dtype=object)
    
    step = 1
    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == end:
            break  
        
        i, j = current
        visit_count[i][j] += 1
        
        if first_visit[i][j] == '.':
            first_visit[i][j] = step
        last_visit[i][j] = step
        step += 1
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != 'X':  
                new_cost = current_cost + terrain_cost(i, j, ni, nj, grid)

                if (ni, nj) not in cost_so_far or new_cost < cost_so_far[(ni, nj)]:
                    cost_so_far[(ni, nj)] = new_cost
                    priority = new_cost
                    heapq.heappush(pq, (priority, (ni, nj)))
                    came_from[(ni, nj)] = current  

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path if path[0] == start else None, visit_count, first_visit, last_visit

def euclidean_heuristic(current, end):
    return math.sqrt((current[0] - end[0])**2 + (current[1] - end[1])**2)

def manhattan_heuristic(current, end):
    return abs(current[0] - end[0]) + abs(current[1] - end[1])

def astar(rows, cols, start, end, grid, heuristic_type):
    print(f"A* search started. Heuristic: {heuristic_type}")
    if heuristic_type == "euclidean":
        heuristic = euclidean_heuristic
    elif heuristic_type == "manhattan":
        heuristic = manhattan_heuristic
    else:
        raise ValueError("Invalid heuristic type")

    pq = [(0 + heuristic(start, end), 0, start)]  # (f = g + h, g, current)
    heapq.heapify(pq)
    came_from = {start: None}
    cost_so_far = {start: 0}
    visit_count = np.zeros((rows, cols), dtype=int)
    first_visit = np.full((rows, cols), '.', dtype=object)
    last_visit = np.full((rows, cols), '.', dtype=object)
    
    step = 1
    while pq:
        _, current_cost, current = heapq.heappop(pq)

        if current == end:
            break  
        
        i, j = current
        visit_count[i][j] += 1
        
        if first_visit[i][j] == '.':
            first_visit[i][j] = step
        last_visit[i][j] = step
        step += 1
        
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] != 'X':  
                new_cost = current_cost + terrain_cost(i, j, ni, nj, grid)

                if (ni, nj) not in cost_so_far or new_cost < cost_so_far[(ni, nj)]:
                    cost_so_far[(ni, nj)] = new_cost
                    priority = new_cost + heuristic((ni, nj), end)
                    heapq.heappush(pq, (priority, new_cost, (ni, nj)))
                    came_from[(ni, nj)] = current  

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    path.reverse()

    return path if path[0] == start else None, visit_count, first_visit, last_visit


def main():
    if len(sys.argv) < 4:
        print("Usage: python pathfinder.py [mode] [map] [algorithm] [heuristic]")
        return
    
    mode = sys.argv[1]  
    map_file = sys.argv[2]  
    algorithm = sys.argv[3]  
    heuristic = sys.argv[4] if algorithm == "astar" else None  
    print(f"Mode: {mode}, Algorithm: {algorithm}, Heuristic: {heuristic}")

    rows, cols, start, end, grid = parse_map(map_file)
    print(f"Parsed Map:\nRows: {rows}, Cols: {cols}, Start: {start}, End: {end}")
    
    path, visits, first_visit, last_visit = None, None, None, None
    
    if algorithm == 'bfs':
        path, visits, first_visit, last_visit = bfs(rows, cols, start, end, grid)
    elif algorithm == 'ucs':
        path, visits, first_visit, last_visit = ucs(rows, cols, start, end, grid)
    elif algorithm == 'astar':
        if heuristic is None:
            print("Error: A* search requires a heuristic (euclidean or manhattan).")
    
    path, visits, first_visit, last_visit = astar(rows, cols, start, end, grid, heuristic)

    print("Path:", path)
    if path:
        print_output(mode, grid, path, visits, first_visit, last_visit)
    else:
        print("No path found.")

def print_output(mode, grid, path, visits, first_visit, last_visit):
    if mode == "debug":
        print("path:")
        print_grid_with_path(grid, path)
        print("\n#visits:")
        print_grid(visits)
        print("\nfirst visit:")
        print_grid(first_visit)
        print("\nlast visit:")
        print_grid(last_visit)
    elif mode == "release":
        print_grid_with_path(grid, path)

def print_grid_with_path(grid, path):
    grid_copy = np.array(grid, dtype=object)
    for (i, j) in path:
        grid_copy[i][j] = '*'
    for row in grid_copy:
        print(" ".join(str(cell) for cell in row))

def print_grid(grid):
    for row in grid:
        print(" ".join(str(cell) for cell in row))

if __name__ == "__main__":
    main()