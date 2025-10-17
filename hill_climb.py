import random

def random_state(n):
    # Randomly place one queen in each column
    return [random.randint(0, n - 1) for _ in range(n)]

def evaluate(state):
    # Count the number of attacking pairs of queens
    conflicts = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts

def generate_neighbors(state):
    neighbors = []
    n = len(state)
    for col in range(n):
        for row in range(n):
            if state[col] != row:
                new_state = state[:]
                new_state[col] = row
                neighbors.append(new_state)
    return neighbors

def hill_climb(n, max_iterations=1000):
    current = random_state(n)
    current_score = evaluate(current)

    for _ in range(max_iterations):
        neighbors = generate_neighbors(current)
        next_state = min(neighbors, key=evaluate)
        next_score = evaluate(next_state)

        if next_score < current_score:
            current = next_state
            current_score = next_score
        else:
            break

    return current, current_score

# Example usage
n = 8
solution, score = hill_climb(n)
print("Final State:", solution)
print("Conflicts:", score)
if score == 0:
    print("Solution found!")
else:
    print("Local optimum reached.")
