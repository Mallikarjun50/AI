"""
cnf_converter_idle.py

Simple IDLE-friendly CNF converter for propositional logic.

Syntax:
  - Variables: letters/digits/underscore starting with a letter:  p, q1, A
  - Operators:
      ~    NOT (unary)
      &    AND
      |    OR
      ->   IMPLIES
      <->  BICONDITIONAL
  - Parentheses: ( ... )

Examples you can try when running:
  (p -> q) & (q -> r)
  ~(p | q) -> (r & s)
  (p | (q & r)) -> s
  (a <-> b) | c
"""

import sys
import re
from copy import deepcopy

# ---------------------------
# AST node classes
# ---------------------------
class Expr:
    pass

class Var(Expr):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Var({self.name})"
    def __eq__(self, other):
        return isinstance(other, Var) and self.name == other.name
    def __hash__(self):
        return hash(("Var", self.name))

class Not(Expr):
    def __init__(self, arg):
        self.arg = arg
    def __repr__(self):
        return f"Not({self.arg})"
    def __eq__(self, other):
        return isinstance(other, Not) and self.arg == other.arg
    def __hash__(self):
        return hash(("Not", self.arg))

class And(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"And({self.left},{self.right})"
    def __eq__(self, other):
        return isinstance(other, And) and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash(("And", self.left, self.right))

class Or(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Or({self.left},{self.right})"
    def __eq__(self, other):
        return isinstance(other, Or) and self.left == other.left and self.right == other.right
    def __hash__(self):
        return hash(("Or", self.left, self.right))

class Implies(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Implies({self.left},{self.right})"

class Iff(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __repr__(self):
        return f"Iff({self.left},{self.right})"

# ---------------------------
# Lexer / Parser (recursive descent)
# ---------------------------
TOKEN_REGEX = re.compile(r'\s*(<->|->|[()~&|]|[A-Za-z][A-Za-z0-9_]*)')

def tokenize(s):
    tokens = TOKEN_REGEX.findall(s)
    if not tokens:
        return []
    return tokens

class Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.pos = 0

    def peek(self):
        if self.pos < len(self.toks):
            return self.toks[self.pos]
        return None

    def pop(self, expected=None):
        t = self.peek()
        if t is None:
            return None
        if expected and t != expected:
            raise SyntaxError(f"Expected token {expected} but found {t}")
        self.pos += 1
        return t

    # Grammar (precedence high->low):
    # atom      := VAR | ~ atom | ( expr )
    # conj      := atom ( & atom )*
    # disj      := conj ( | conj )*
    # impl      := disj ( -> disj )*
    # iff       := impl ( <-> impl )*
    # expr      := iff
    def parse(self):
        if not self.toks:
            raise SyntaxError("Empty input")
        node = self.parse_iff()
        if self.peek() is not None:
            raise SyntaxError("Unexpected token: " + str(self.peek()))
        return node

    def parse_iff(self):
        left = self.parse_impl()
        while self.peek() == "<->":
            self.pop("<->")
            right = self.parse_impl()
            left = Iff(left, right)
        return left

    def parse_impl(self):
        left = self.parse_disj()
        while self.peek() == "->":
            self.pop("->")
            right = self.parse_disj()
            left = Implies(left, right)
        return left

    def parse_disj(self):
        left = self.parse_conj()
        while self.peek() == "|":
            self.pop("|")
            right = self.parse_conj()
            left = Or(left, right)
        return left

    def parse_conj(self):
        left = self.parse_atom()
        while self.peek() == "&":
            self.pop("&")
            right = self.parse_atom()
            left = And(left, right)
        return left

    def parse_atom(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of input")
        if t == "(":
            self.pop("(")
            node = self.parse_iff()
            if self.peek() != ")":
                raise SyntaxError("Missing closing parenthesis")
            self.pop(")")
            return node
        if t == "~":
            self.pop("~")
            sub = self.parse_atom()
            return Not(sub)
        if re.match(r'[A-Za-z][A-Za-z0-9_]*', t):
            self.pop()
            return Var(t)
        raise SyntaxError("Unexpected token: " + t)

# ---------------------------
# Transformations to CNF
# ---------------------------

def eliminate_iff_implies(expr):
    """Return expression with no Iff or Implies (only Not, And, Or, Var)."""
    if isinstance(expr, Var):
        return expr
    if isinstance(expr, Not):
        return Not(eliminate_iff_implies(expr.arg))
    if isinstance(expr, And):
        return And(eliminate_iff_implies(expr.left), eliminate_iff_implies(expr.right))
    if isinstance(expr, Or):
        return Or(eliminate_iff_implies(expr.left), eliminate_iff_implies(expr.right))
    if isinstance(expr, Implies):
        # A -> B  ===  (~A) | B
        return Or(Not(eliminate_iff_implies(expr.left)), eliminate_iff_implies(expr.right))
    if isinstance(expr, Iff):
        # A <-> B  ===  (A -> B) & (B -> A)
        a = eliminate_iff_implies(expr.left)
        b = eliminate_iff_implies(expr.right)
        return And(Or(Not(a), b), Or(Not(b), a))
    raise ValueError("Unknown node type in eliminate_iff_implies")

def push_not_inwards(expr):
    """Transform to negation normal form (NNF): push NOT inwards, eliminate double negation."""
    if isinstance(expr, Var):
        return expr
    if isinstance(expr, Not):
        inner = expr.arg
        if isinstance(inner, Var):
            return expr  # Not over Var stays
        if isinstance(inner, Not):
            return push_not_inwards(inner.arg)  # double negation
        if isinstance(inner, And):
            # ~(A & B) = ~A | ~B
            return Or(push_not_inwards(Not(inner.left)), push_not_inwards(Not(inner.right)))
        if isinstance(inner, Or):
            # ~(A | B) = ~A & ~B
            return And(push_not_inwards(Not(inner.left)), push_not_inwards(Not(inner.right)))
        # shouldn't see Implies/Iff here if eliminated earlier
        return Not(push_not_inwards(inner))
    if isinstance(expr, And):
        return And(push_not_inwards(expr.left), push_not_inwards(expr.right))
    if isinstance(expr, Or):
        return Or(push_not_inwards(expr.left), push_not_inwards(expr.right))
    raise ValueError("Unknown node type in push_not_inwards")

def distribute_or_over_and(a, b):
    """
    Distribute Or over And to preserve equivalence:
      Or(A, And(B,C))  ->  And( Or(A,B), Or(A,C) )
      Or(And(B,C), A)  ->  And( Or(B,A), Or(C,A) )
    Operates on already NNF expressions.
    """
    # If either side is an And, distribute
    if isinstance(a, And):
        left = distribute_or_over_and(a.left, b)
        right = distribute_or_over_and(a.right, b)
        return And(left, right)
    if isinstance(b, And):
        left = distribute_or_over_and(a, b.left)
        right = distribute_or_over_and(a, b.right)
        return And(left, right)
    # otherwise just an Or
    return Or(a, b)

def to_cnf_expr(expr):
    """Full pipeline returning an expression in CNF (And of Ors)."""
    # 1. eliminate -> and <->
    no_imp = eliminate_iff_implies(expr)
    # 2. push NOT inwards -> NNF
    nnf = push_not_inwards(no_imp)
    # 3. recursively distribute OR over AND
    def distribute(node):
        if isinstance(node, Var) or isinstance(node, Not):
            return node
        if isinstance(node, And):
            return And(distribute(node.left), distribute(node.right))
        if isinstance(node, Or):
            left = distribute(node.left)
            right = distribute(node.right)
            return distribute_or_over_and(left, right)
        raise ValueError("Unexpected node in distribute: " + repr(node))
    cnf = distribute(nnf)
    return cnf

# ---------------------------
# Helpers: convert CNF expr -> list of clauses
# ---------------------------
def expr_to_clauses(expr):
    """
    Convert an expr that is an AND of ORs (CNF) into list of clauses.
    Each clause is represented as a frozenset of literals, where literal is a tuple ('p', True) or ('p', False)
    meaning positive or negated variable.
    """
    def clause_from_or(node):
        # returns set of literals for an OR-node (or single literal)
        if isinstance(node, Var):
            return { (node.name, True) }
        if isinstance(node, Not) and isinstance(node.arg, Var):
            return { (node.arg.name, False) }
        if isinstance(node, Or):
            left = clause_from_or(node.left)
            right = clause_from_or(node.right)
            return left.union(right)
        raise ValueError("Unexpected node in clause_from_or: " + repr(node))

    clauses = []
    if isinstance(expr, And):
        left_clauses = expr_to_clauses(expr.left)
        right_clauses = expr_to_clauses(expr.right)
        return left_clauses + right_clauses
    else:
        # single clause
        clause = frozenset(clause_from_or(expr))
        return [clause]

def clauses_to_string(clauses):
    parts = []
    for c in clauses:
        lits = []
        for (v, pos) in sorted(c):
            lits.append(v if pos else f"~{v}")
        parts.append("(" + " | ".join(lits) + ")")
    return " & ".join(parts)

# ---------------------------
# Pretty printer for expressions
# ---------------------------
def expr_to_string(e):
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Not):
        inner = e.arg
        if isinstance(inner, Var):
            return "~" + inner.name
        return "~(" + expr_to_string(inner) + ")"
    if isinstance(e, And):
        return "(" + expr_to_string(e.left) + " & " + expr_to_string(e.right) + ")"
    if isinstance(e, Or):
        return "(" + expr_to_string(e.left) + " | " + expr_to_string(e.right) + ")"
    if isinstance(e, Implies):
        return "(" + expr_to_string(e.left) + " -> " + expr_to_string(e.right) + ")"
    if isinstance(e, Iff):
        return "(" + expr_to_string(e.left) + " <-> " + expr_to_string(e.right) + ")"
    return str(e)

# ---------------------------
# Command-line / IDLE interaction
# ---------------------------
def convert_to_cnf_string(formula_str):
    toks = tokenize(formula_str)
    if not toks:
        raise SyntaxError("No tokens parsed; check formula.")
    parser = Parser(toks)
    ast = parser.parse()
    cnf_expr = to_cnf_expr(ast)
    clauses = expr_to_clauses(cnf_expr)
    return ast, cnf_expr, clauses

def demo_examples():
    examples = [
        "(p -> q) & (q -> r)",
        "~(p | q) -> (r & s)",
        "(p | (q & r)) -> s",
        "(a <-> b) | c",
        "~(p & q) | (r -> s)"
    ]
    for ex in examples:
        print("Input:", ex)
        ast, cnf_expr, clauses = convert_to_cnf_string(ex)
        print("Parsed AST:", expr_to_string(ast))
        print("CNF expression:", expr_to_string(cnf_expr))
        print("Clauses:", clauses_to_string(clauses))
        print("-" * 40)

def interactive():
    print("Propositional CNF converter (IDLE-friendly).")
    print("Enter a formula using ~ & | -> <-> and parentheses. Empty line quits.")
    while True:
        try:
            s = input("Formula> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not s:
            break
        try:
            ast, cnf_expr, clauses = convert_to_cnf_string(s)
            print("Parsed:", expr_to_string(ast))
            print("CNF expr:", expr_to_string(cnf_expr))
            print("Clauses:", clauses_to_string(clauses))
            print()
        except Exception as e:
            print("Error:", e)
            print("Make sure your syntax is correct. Example: (p -> q) & (q -> r)")
            print()

if __name__ == "__main__":
    # Demo on start, then interactive prompt
    print("Running demo examples...\n")
    demo_examples()
    interactive()
