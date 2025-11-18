# Define logical operations
def AND(p, q):
    return p and q

def OR(p, q):
    return p or q

def NOT(p):
    return not p

def IMPLIES(p, q):
    return (not p) or q

def BICONDITIONAL(p, q):
    return p == q

# Example usage
if __name__ == "__main__":
    # Define truth values for propositions
    P = True
    Q = False

    # Evaluate expressions
    print("P AND Q:", AND(P, Q))
    print("P OR Q:", OR(P, Q))
    print("NOT P:", NOT(P))
    print("P IMPLIES Q:", IMPLIES(P, Q))
    print("P BICONDITIONAL Q:", BICONDITIONAL(P, Q))
