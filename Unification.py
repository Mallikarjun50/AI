def is_variable(x):
    return isinstance(x, str) and x.islower()

def occurs_check(var, x, substitutions):
    if var == x:
        return True
    elif is_variable(x) and x in substitutions:
        return occurs_check(var, substitutions[x], substitutions)
    elif isinstance(x, list):
        return any(occurs_check(var, xi, substitutions) for xi in x)
    return False

def unify_var(var, x, substitutions):
    if var in substitutions:
        return unify(substitutions[var], x, substitutions)
    elif is_variable(x) and x in substitutions:
        return unify(var, substitutions[x], substitutions)
    elif occurs_check(var, x, substitutions):
        return None
    else:
        substitutions[var] = x
        return substitutions

def unify(x, y, substitutions=None):
    if substitutions is None:
        substitutions = {}

    if x == y:
        return substitutions

    if is_variable(x):
        return unify_var(x, y, substitutions)

    if is_variable(y):
        return unify_var(y, x, substitutions)

    if isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
        for xi, yi in zip(x, y):
            substitutions = unify(xi, yi, substitutions)
            if substitutions is None:
                return None
        return substitutions

    return None

# Example usage
if __name__ == "__main__":
    expr1 = ['f', 'x', ['g', 'y']]
    expr2 = ['f', ['g', 'z'], ['g', 'z']]

    result = unify(expr1, expr2)
    if result:
        print("Unification successful. Substitutions:")
        for k, v in result.items():
            print(f"{k} → {v}")
    else:
        print("Unification failed.")
