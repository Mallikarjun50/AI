from collections import deque

# Define the goal state
GOAL_STATE = (1, 2, 3,
              4, 5, 6,
              7, 8, 0)

# Define the initial state
INITIAL_STATE = (1, 2, 3,
                 0, 4, 6,
                 7, 5, 8)

# Possible moves: up, down, left, right
MOVES = {
    'up': -3,
    'down': 3,
    'left': -1,
    'right': 1
}

def is_valid_move(blank_index, move):
    if move == 'up' and blank_index < 3:
        return False
    if move == 'down' and blank_index > 5:
        return False
    if move == 'left' and blank_index % 3 == 0:
        return False
    if move == 'right' and blank_index % 3 == 2:
        return False
    return True

def move_blank(state, move):
    blank_index = state.index(0)
    if not is_valid_move(blank_index, move):
        return None
    new_index = blank_index + MOVES[move]
    new_state = list(state)
    new_state[blank_index], new_state[new_index] = new_state[new_index], new_state[blank_index]
    return tuple(new_state)

def depth_limited_search(state, depth, path, visited):
    if state == GOAL_STATE:
        return path
    if depth == 0:
        return None
    visited.add(state)
    for move in MOVES:
        new_state = move_blank(state, move)
        if new_state and new_state not in visited:
            result = depth_limited_search(new_state, depth - 1, path + [new_state], visited)
            if result:
                return result
    return None

def iterative_deepening_search(start_state):
    depth = 0
    while True:
        visited = set()
        print(f"Trying depth limit: {depth}")
        result = depth_limited_search(start_state, depth, [start_state], visited)
        if result:
            return result
        depth += 1

# Run the IDS algorithm
solution_path = iterative_deepening_search(INITIAL_STATE)

# Display the solution path
print("\nSolution found!")
for i, state in enumerate(solution_path):
    print(f"\nStep {i}:")
    print(state[0:3])
    print(state[3:6])
    print(state[6:9])
