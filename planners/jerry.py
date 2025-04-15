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
    Enhanced heuristic that considers:
    1. Distance to target (pursued)
    2. Distance from pursuer
    3. Safety margin from pursuer
    4. Escape routes availability
    5. Strategic positioning
    """
    # Base distances
    dist_to_target = manhattan_distance(current, pursued)
    dist_from_pursuer = manhattan_distance(current, pursuer)
    
    # Safety factors
    SAFE_DISTANCE = 4  # Increased safe distance
    DANGER_DISTANCE = 2  # Distance at which we consider immediate danger
    
    # Calculate safety score
    if dist_from_pursuer < DANGER_DISTANCE:
        safety_penalty = 10  # High penalty for being very close to pursuer
    else:
        safety_penalty = max(0, SAFE_DISTANCE - dist_from_pursuer) * 2
    
    # Find escape routes
    escape_paths = find_escape_paths(world, current, pursuer)
    escape_score = len(escape_paths) * 0.5  # Bonus for having escape routes
    
    # Strategic positioning
    # Prefer positions that put us between target and pursuer when safe
    strategic_score = 0
    if dist_from_pursuer > SAFE_DISTANCE:
        target_to_pursuer = manhattan_distance(pursued, pursuer)
        our_to_target = manhattan_distance(current, pursued)
        our_to_pursuer = dist_from_pursuer
        if our_to_target + our_to_pursuer < target_to_pursuer:
            strategic_score = 2  # Bonus for good strategic position
    
    # Combine all factors
    return (dist_to_target * 0.7 -  # Minimize distance to target
            dist_from_pursuer * 0.8 +  # Maximize distance from pursuer
            safety_penalty * 1.2 -  # Avoid danger
            escape_score * 0.6 +  # Encourage positions with escape routes
            strategic_score)  # Encourage strategic positioning

def a_star_search(world, start, pursued, pursuer):
    """
    Enhanced A* search with sophisticated heuristic and tactical considerations
    """
    start_node = Node(start, 0, calculate_heuristic(start, pursued, pursuer, world))
    open_list = [start_node]
    closed_set = set()
    node_dict = {start: start_node}  # For efficient node lookup
    
    while open_list:
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
    def __init__(self):
        self.last_positions = []  # Store last few positions to detect cycles
        self.cycle_count = 0
        self.max_cycle_count = 3
    
    def plan_action(world, current, pursued, pursuer):
        """
        Computes an optimal action to take from the current position to capture the pursued while evading the pursuer
        """
        # Convert numpy arrays to tuples for the search
        current_pos = (int(current[0]), int(current[1]))
        pursued_pos = (int(pursued[0]), int(pursued[1]))
        pursuer_pos = (int(pursuer[0]), int(pursuer[1]))
        
        # Get the optimal path
        path = a_star_search(world, current_pos, pursued_pos, pursuer_pos)
        
        if path and len(path) > 1:
            next_pos = path[1]
            action = np.array([next_pos[0] - current_pos[0], 
                             next_pos[1] - current_pos[1]])
            
            # Store position for cycle detection
            if not hasattr(PlannerAgent, 'last_positions'):
                PlannerAgent.last_positions = []
            PlannerAgent.last_positions.append(current_pos)
            if len(PlannerAgent.last_positions) > 5:
                PlannerAgent.last_positions.pop(0)
            
            # Check for cycles
            if len(PlannerAgent.last_positions) >= 4:
                if (PlannerAgent.last_positions[-1] == PlannerAgent.last_positions[-3] and 
                    PlannerAgent.last_positions[-2] == PlannerAgent.last_positions[-4]):
                    PlannerAgent.cycle_count += 1
                    if PlannerAgent.cycle_count >= PlannerAgent.max_cycle_count:
                        # Force a different move to break the cycle
                        directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                                             [-1, -1], [-1, 1], [1, -1], [1, 1]])
                        safe_moves = []
                        for direction in directions:
                            new_pos = current + direction
                            if (0 <= new_pos[0] < world.shape[0] and 
                                0 <= new_pos[1] < world.shape[1] and 
                                world[new_pos[0], new_pos[1]] == 0 and
                                tuple(new_pos) not in PlannerAgent.last_positions[-3:]):
                                safe_moves.append(direction)
                        if safe_moves:
                            PlannerAgent.cycle_count = 0
                            return safe_moves[np.random.choice(len(safe_moves))]
                else:
                    PlannerAgent.cycle_count = 0
            
            return action
        
        # Fallback to tactical random move if no path is found
        directions = np.array([[0,0], [-1, 0], [1, 0], [0, -1], [0, 1],
                             [-1, -1], [-1, 1], [1, -1], [1, 1]])
        
        # Score each possible move
        move_scores = []
        for direction in directions:
            new_pos = current + direction
            if (0 <= new_pos[0] < world.shape[0] and 
                0 <= new_pos[1] < world.shape[1] and 
                world[new_pos[0], new_pos[1]] == 0):
                # Score based on distance to target and from pursuer
                score = (manhattan_distance(tuple(new_pos), pursued_pos) * 0.7 -
                        manhattan_distance(tuple(new_pos), pursuer_pos) * 0.8)
                move_scores.append((score, direction))
        
        if move_scores:
            # Choose the move with the best score
            move_scores.sort(reverse=True)
            return move_scores[0][1]
        
        # If no safe moves, stay still
        return np.array([0, 0])


