import numpy as np
import math
from typing import List, Tuple, Optional, Set, Dict

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
    """Calculate Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors(pos, world):
    """Get valid neighboring positions."""
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
    """Find potential escape paths considering pursuer's position."""
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

def predict_spike_movement(world, spike_pos, tom_pos, jerry_pos):
    """
    Simplified prediction of Spike's next move based on immediate threats and opportunities.
    """
    possible_moves = get_neighbors(spike_pos, world)
    if not possible_moves:
        return [spike_pos]
    
    # Score each possible move
    move_scores = []
    for move in possible_moves:
        score = 0
        # Moving away from Tom is good
        if manhattan_distance(move, tom_pos) > manhattan_distance(spike_pos, tom_pos):
            score += 2
        # Moving toward Jerry is good
        if manhattan_distance(move, jerry_pos) < manhattan_distance(spike_pos, jerry_pos):
            score += 1
        move_scores.append((score, move))
    
    # Sort moves by score and return top 2
    move_scores.sort(reverse=True, key=lambda x: x[0])
    return [move for _, move in move_scores[:2]]

def find_interception_point(world, jerry_pos, spike_pos, tom_pos):
    """
    Simplified interception point calculation focusing on immediate tactical advantage.
    Only used when Tom is far enough away to make interception safe.
    """
    predicted_moves = predict_spike_movement(world, spike_pos, tom_pos, jerry_pos)
    
    if len(predicted_moves) == 1 and predicted_moves[0] == spike_pos:
        return spike_pos
    
    best_interception = None
    best_score = float('-inf')
    
    jerry_moves = get_neighbors(jerry_pos, world)
    if not jerry_moves:
        return jerry_pos
    
    for move in jerry_moves:
        score = 0
        # Score based on distance to Spike's predicted moves
        min_dist_to_spike = min(manhattan_distance(move, spike_move) for spike_move in predicted_moves)
        score -= min_dist_to_spike
        
        # Score based on distance from Tom
        dist_from_tom = manhattan_distance(move, tom_pos)
        score += dist_from_tom * 0.5
        
        if score > best_score:
            best_score = score
            best_interception = move
    
    return best_interception or jerry_pos

def calculate_heuristic(current, pursued, pursuer, world):
    """
    Enhanced heuristic that considers safety distance from Tom.
    """
    dist_to_target = manhattan_distance(current, pursued)
    dist_from_pursuer = manhattan_distance(current, pursuer)
    
    # Safety factors
    SAFE_DISTANCE = 4  # Minimum safe distance from Tom
    DANGER_DISTANCE = 2  # Distance at which we consider immediate danger
    
    # Calculate safety score
    if dist_from_pursuer < DANGER_DISTANCE:
        return float('inf')  # Avoid immediate danger
    
    # Heavily penalize being too close to Tom
    safety_penalty = max(0, SAFE_DISTANCE - dist_from_pursuer) * 3
    
    return float(dist_to_target - dist_from_pursuer * 1.2 - safety_penalty)

def a_star_search(world, start, pursued, pursuer):
    """
    Optimized A* search with safety-aware heuristic.
    """
    start_node = Node(start, 0, calculate_heuristic(start, pursued, pursuer, world))
    open_list = [start_node]
    closed_set = set()
    node_dict = {start: start_node}
    
    while open_list and len(closed_set) < 100:  # Limit search depth
        current = min(open_list, key=lambda x: x.f_cost)
        open_list.remove(current)
        
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
    # Class attributes
    last_positions = []
    cycle_count = 0
    max_cycle_count = 3
    last_spike_pos = None
    last_tom_pos = None
    
    def __init__(self):
        pass
    
    def plan_action(world, current, pursued, pursuer):
        """
        Enhanced action planning with safety-aware decision making.
        """
        current_pos = (int(current[0]), int(current[1]))
        pursued_pos = (int(pursued[0]), int(pursued[1]))
        pursuer_pos = (int(pursuer[0]), int(pursuer[1]))
        
        # Calculate distances
        dist_to_spike = manhattan_distance(current_pos, pursued_pos)
        dist_from_tom = manhattan_distance(current_pos, pursuer_pos)
        
        # Emergency escape if Tom is too close
        if dist_from_tom < 3:
            directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                 [-1, -1], [-1, 1], [1, -1], [1, 1]])
            safe_moves = []
            for direction in directions:
                new_pos = current + direction
                if (0 <= new_pos[0] < world.shape[0] and 
                    0 <= new_pos[1] < world.shape[1] and 
                    world[new_pos[0], new_pos[1]] == 0):
                    if manhattan_distance(tuple(new_pos), pursuer_pos) > dist_from_tom:
                        safe_moves.append(direction)
            if safe_moves:
                return safe_moves[np.random.choice(len(safe_moves))]
        
        # Direct capture opportunity (only if Tom is far enough)
        if dist_to_spike <= 2 and dist_from_tom > 4:
            path = a_star_search(world, current_pos, pursued_pos, pursuer_pos)
            if path and len(path) > 1:
                next_pos = path[1]
                return np.array([next_pos[0] - current_pos[0], 
                               next_pos[1] - current_pos[1]])
        
        # Use interception only when Tom is far enough away
        if dist_from_tom > 4 and PlannerAgent.last_spike_pos is not None:
            interception_point = find_interception_point(world, current_pos, pursued_pos, pursuer_pos)
            path = a_star_search(world, current_pos, interception_point, pursuer_pos)
            if path and len(path) > 1:
                next_pos = path[1]
                PlannerAgent.last_spike_pos = pursued_pos
                PlannerAgent.last_tom_pos = pursuer_pos
                return np.array([next_pos[0] - current_pos[0], 
                               next_pos[1] - current_pos[1]])
        
        # Standard pursuit with safety awareness
        path = a_star_search(world, current_pos, pursued_pos, pursuer_pos)
        if path and len(path) > 1:
            next_pos = path[1]
            PlannerAgent.last_spike_pos = pursued_pos
            PlannerAgent.last_tom_pos = pursuer_pos
            return np.array([next_pos[0] - current_pos[0], 
                           next_pos[1] - current_pos[1]])
        
        # If all else fails, stay still
        return np.array([0, 0])


