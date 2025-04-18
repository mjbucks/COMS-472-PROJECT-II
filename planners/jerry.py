import numpy as np
import math

class Node:
    def __init__(self, pos, g_cost=0, h_cost=0, parent=None):
        self.pos = pos
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent

    def __lt__(self, other):
        return self.f_cost < other.f_cost

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors(pos, world):
    # Pre-compute all possible moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal
    neighbors = []
    for dx, dy in directions:
        new_x, new_y = pos[0] + dx, pos[1] + dy
        if (0 <= new_x < world.shape[0] and 
            0 <= new_y < world.shape[1] and 
            world[new_x, new_y] == 0):
            neighbors.append((new_x, new_y))
    return neighbors

def find_escape_paths(world, pos, pursuer, depth=3):
    """Find potential escape paths considering pursuer's position"""
    escape_paths = set()
    queue = [(pos, 0)]
    visited = {pos}
    
    while queue:
        current, current_depth = queue.pop(0)
        if current_depth >= depth:
            continue
            
        for next_pos in get_neighbors(current, world):
            if next_pos not in visited:
                visited.add(next_pos)
                queue.append((next_pos, current_depth + 1))
                # Only add positions that increase distance from pursuer
                if manhattan_distance(next_pos, pursuer) > manhattan_distance(current, pursuer):
                    escape_paths.add(next_pos)
    
    return escape_paths

def calculate_heuristic(current, pursued, pursuer, world):
    """
    Optimized heuristic that considers:
    1. Distance to target (pursued)
    2. Distance from pursuer
    3. Safety margin from pursuer
    """
    # Base distances
    dist_to_target = manhattan_distance(current, pursued)
    dist_from_pursuer = manhattan_distance(current, pursuer)
    
    # Safety factors
    SAFE_DISTANCE = 4
    DANGER_DISTANCE = 2
    
    # Calculate safety score
    if dist_from_pursuer < DANGER_DISTANCE:
        safety_penalty = 10
    else:
        safety_penalty = max(0, SAFE_DISTANCE - dist_from_pursuer) * 2
    
    # Strategic positioning - simplified
    strategic_score = 0
    if dist_from_pursuer > SAFE_DISTANCE:
        target_to_pursuer = manhattan_distance(pursued, pursuer)
        our_to_target = manhattan_distance(current, pursued)
        our_to_pursuer = dist_from_pursuer
        if our_to_target + our_to_pursuer < target_to_pursuer:
            strategic_score = 2
    
    # Combine factors with optimized weights
    return (dist_to_target * 0.7 - 
            dist_from_pursuer * 0.8 + 
            safety_penalty * 1.2 + 
            strategic_score)

def a_star_search(world, start, pursued, pursuer, max_steps=100):
    """
    Optimized A* search with early termination
    """
    start_node = Node(start, 0, calculate_heuristic(start, pursued, pursuer, world))
    open_list = [start_node]
    closed_set = set()
    node_dict = {start: start_node}
    steps = 0
    
    while open_list and steps < max_steps:
        steps += 1
        # Find node with minimum f_cost
        min_idx = 0
        for i in range(1, len(open_list)):
            if open_list[i].f_cost < open_list[min_idx].f_cost:
                min_idx = i
        current = open_list.pop(min_idx)
        
        if current.pos == pursued:
            path = []
            while current:
                path.append(current.pos)
                current = current.parent
            return path[::-1]
        
        closed_set.add(current.pos)
        
        for neighbor_pos in get_neighbors(current.pos, world):
            if neighbor_pos in closed_set:
                continue
                
            g_cost = current.g_cost + 1
            h_cost = calculate_heuristic(neighbor_pos, pursued, pursuer, world)
            neighbor = Node(neighbor_pos, g_cost, h_cost, current)
            
            if neighbor_pos not in node_dict or node_dict[neighbor_pos].f_cost > neighbor.f_cost:
                node_dict[neighbor_pos] = neighbor
                open_list.append(neighbor)
    
    return None

class PlannerAgent:
    # Class variables for cycle detection
    last_positions = []
    cycle_count = 0
    max_cycle_count = 3
    
    def __init__(self):
        pass
    
    def plan_action(world, current, pursued, pursuer):
        """
        Optimized action planning with strategic decision making
        """
        # Convert numpy arrays to tuples for the search
        current_pos = (int(current[0]), int(current[1]))
        pursued_pos = (int(pursued[0]), int(pursued[1]))
        pursuer_pos = (int(pursuer[0]), int(pursuer[1]))
        
        # Get the optimal path with limited steps
        path = a_star_search(world, current_pos, pursued_pos, pursuer_pos, max_steps=100)
        
        if path and len(path) > 1:
            next_pos = path[1]
            action = np.array([next_pos[0] - current_pos[0], 
                             next_pos[1] - current_pos[1]])
            
            # Simplified cycle detection
            if len(PlannerAgent.last_positions) >= 4:
                if (PlannerAgent.last_positions[-1] == PlannerAgent.last_positions[-3] and 
                    PlannerAgent.last_positions[-2] == PlannerAgent.last_positions[-4]):
                    PlannerAgent.cycle_count += 1
                    if PlannerAgent.cycle_count >= PlannerAgent.max_cycle_count:
                        # Force a different move
                        directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                             [-1, -1], [-1, 1], [1, -1], [1, 1]])
                        safe_moves = []
                        for direction in directions:
                            new_pos = current + direction
                            if (0 <= new_pos[0] < world.shape[0] and 
                                0 <= new_pos[1] < world.shape[1] and 
                                world[new_pos[0], new_pos[1]] == 0):
                                safe_moves.append(direction)
                        if safe_moves:
                            PlannerAgent.cycle_count = 0
                            return safe_moves[np.random.choice(len(safe_moves))]
                else:
                    PlannerAgent.cycle_count = 0
            
            # Update position history
            PlannerAgent.last_positions.append(current_pos)
            if len(PlannerAgent.last_positions) > 5:
                PlannerAgent.last_positions.pop(0)
            
            return action
        
        # Fallback to tactical move
        directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                             [-1, -1], [-1, 1], [1, -1], [1, 1]])
        
        # Score moves based on simple heuristic
        move_scores = []
        for direction in directions:
            new_pos = current + direction
            if (0 <= new_pos[0] < world.shape[0] and 
                0 <= new_pos[1] < world.shape[1] and 
                world[new_pos[0], new_pos[1]] == 0):
                # Simplified scoring
                score = (manhattan_distance(tuple(new_pos), pursued_pos) * 0.7 -
                        manhattan_distance(tuple(new_pos), pursuer_pos) * 0.8)
                move_scores.append((float(score), direction))
        
        if move_scores:
            # Choose the move with the best score
            move_scores.sort(key=lambda x: x[0], reverse=True)
            return move_scores[0][1]
        
        # If no safe moves, stay still
        return np.array([0, 0])


