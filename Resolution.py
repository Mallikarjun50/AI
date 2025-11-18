from typing import List, Dict, Tuple, Set

def is_variable(term: str) -> bool:
    return term.islower()

def unify(term1: str, term2: str, subst: Dict[str, str]) -> Dict[str, str] or None:
    if term1 == term2:
        return subst
    if is_variable(term1):
        return unify_var(term1, term2, subst)
    if is_variable(term2):
        return unify_var(term2, term1, subst)
    return None

def unify_var(var: str, val: str, subst: Dict[str, str]) -> Dict[str, str] or None:
    if var in subst:
        return unify(subst[var], val, subst)
    elif val in subst:
        return unify(var, subst[val], subst)
    else:
        subst[var] = val
        return subst

def apply_subst(literal: str, subst: Dict[str, str]) -> str:
    pred, args = literal.split("(")
    args = args[:-1].split(",")
    new_args = [subst.get(arg, arg) for arg in args]
    return f"{pred}({','.join(new_args)})"

def resolve(c1: Set[str], c2: Set[str]) -> Set[str] or None:
    for l1 in c1:
        for l2 in c2:
            pred1, args1 = l1.replace("¬", "").split("(")
            pred2, args2 = l2.replace("¬", "").split("(")
            args1 = args1[:-1].split(",")
            args2 = args2[:-1].split(",")

            if pred1 == pred2 and l1.startswith("¬") != l2.startswith("¬"):
                subst = {}
                for a1, a2 in zip(args1, args2):
                    subst = unify(a1, a2, subst)
                    if subst is None:
                        break
                if subst is not None:
                    new_clause = set()
                    for lit in (c1 | c2):
                        if lit != l1 and lit != l2:
                            new_clause.add(apply_subst(lit, subst))
                    return new_clause
    return None

def resolution(kb: List[Set[str]], goal: str) -> bool:
    clauses = kb.copy()
    clauses.append({f"¬{goal}"})
    seen = set(frozenset(c) for c in clauses)

    print("Initial clauses:")
    for clause in clauses:
        print(clause)
    print("\nStarting resolution...\n")

    while True:
        new_clauses = []
        pairs = [(clauses[i], clauses[j]) for i in range(len(clauses)) for j in range(i+1, len(clauses))]
        for (ci, cj) in pairs:
            resolvent = resolve(ci, cj)
            if resolvent is not None:
                if not resolvent:
                    print(f"Resolved {ci} and {cj} → ∅ (empty clause)")
                    print("\n✅ Derived empty clause. Goal is proven: John likes peanuts.")
                    return True
                fr = frozenset(resolvent)
                if fr not in seen:
                    print(f"Resolved {ci} and {cj} → {resolvent}")
                    new_clauses.append(resolvent)
                    seen.add(fr)
        if not new_clauses:
            print("\n❌ No new clauses. Goal cannot be proven.")
            return False
        clauses.extend(new_clauses)

# Knowledge base with variables
kb = [
    {"¬Food(x)", "Likes(John,x)"},                      # John likes all food
    {"Food(Apple)"},                                    # Apple is food
    {"Food(Vegetables)"},                               # Vegetables are food
    {"Eats(x,y)", "¬Killed(x)", "Food(y)"},             # Anything eaten and not killed is food
    {"Eats(Anil,Peanuts)"},                             # Anil eats peanuts
    {"Alive(Anil)"},                                    # Anil is alive
    {"¬Eats(Anil,x)", "Eats(Harry,x)"},                 # Harry eats what Anil eats
    {"¬Alive(x)", "¬Killed(x)"},                        # Alive implies not killed
    {"Killed(x)", "Alive(x)"}                           # Not killed implies alive
]

# Run resolution to prove Likes(John,Peanuts)
resolution(kb, "Likes(John,Peanuts)")
