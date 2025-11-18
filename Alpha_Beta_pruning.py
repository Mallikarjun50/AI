# -------------------------------------------
# Alpha–Beta pruning for a full binary tree
# Works directly in Python IDLE
# -------------------------------------------

# Leaf values (A to P) – change if needed
LEAF_VALUES = [21, 5, 15, 11, 12, 8, 9, 13,
               5, 12, 13, 12, 13, 14, 7, 10]

LABELS = [chr(ord('A') + i) for i in range(16)]

import math

# Build full binary tree
DEPTH = 4
NUM_NODES = 2 ** (DEPTH + 1) - 1
nodes = [None] * NUM_NODES

# Insert leaves
leaf_start = 2 ** DEPTH - 1
for i, value in enumerate(LEAF_VALUES):
    nodes[leaf_start + i] = ("leaf", value, LABELS[i])

# Insert internal nodes
for i in range(leaf_start - 1, -1, -1):
    nodes[i] = ("internal", None, None)

# Track pruned leaves
pruned = set()

def collect_leaves(idx):
    """Return all leaves under a subtree."""
    result = []
    stack = [idx]
    while stack:
        n = stack.pop()
        if n >= len(nodes):
            continue
        if nodes[n][0] == "leaf":
            result.append(n)
        else:
            stack.append(2*n + 1)
            stack.append(2*n + 2)
    return result

# --------------------------
# Alpha–Beta Implementation
# --------------------------
def alpha_beta(i, depth, alpha, beta, maximizing):
    node_type, value, label = nodes[i]

    # LEAF NODE
    if node_type == "leaf":
        print(f"Visit leaf {label} = {value}")
        return value

    # INTERNAL NODE
    if maximizing:
        best = -99999
        print(f"MAX node at index {i}")

        for child in (2*i+1, 2*i+2):
            val = alpha_beta(child, depth+1, alpha, beta, False)
            best = max(best, val)
            alpha = max(alpha, best)

            if alpha >= beta:
                # prune siblings after this child
                sibling = 2*i+2 if child == 2*i+1 else None
                if sibling:
                    leaves = collect_leaves(sibling)
                    for li in leaves:
                        pruned.add(li)
                print("PRUNE at MAX node")
                break
        return best

    else:
        best = 99999
        print(f"MIN node at index {i}")

        for child in (2*i+1, 2*i+2):
            val = alpha_beta(child, depth+1, alpha, beta, True)
            best = min(best, val)
            beta = min(beta, best)

            if alpha >= beta:
                sibling = 2*i+2 if child == 2*i+1 else None
                if sibling:
                    leaves = collect_leaves(sibling)
                    for li in leaves:
                        pruned.add(li)
                print("PRUNE at MIN node")
                break
        return best

# Run alpha-beta from root (MAX)
root_value = alpha_beta(0, 0, -99999, 99999, True)

print("\n==========================")
print("Final Alpha–Beta Result")
print("==========================")
print("Root value:", root_value)

# ------------------------
# Determine Best Path
# ------------------------
def minimax(i, maximizing):
    node_type, value, _ = nodes[i]
    if node_type == "leaf":
        return value

    left = minimax(2*i+1, not maximizing)
    right = minimax(2*i+2, not maximizing)

    return max(left, right) if maximizing else min(left, right)

path = []
idx = 0
is_max = True

while nodes[idx][0] != "leaf":
    left = minimax(2*idx+1, not is_max)
    right = minimax(2*idx+2, not is_max)

    if is_max:
        idx = 2*idx+1 if left >= right else 2*idx+2
    else:
        idx = 2*idx+1 if left <= right else 2*idx+2

    path.append(idx)
    is_max = not is_max

print("Best path:")

for p in path:
    node_type, value, label = nodes[p]
    if node_type == "leaf":
        print(f" -> {label} = {value}")

print("\nPruned Leaves:")
for p in sorted(pruned):
    print(" ->", nodes[p][2], "=", nodes[p][1])
