class KnowledgeBase:
    def __init__(self):
        self.facts = set()
        self.rules = []

    def add_fact(self, fact):
        self.facts.add(fact)

    def add_rule(self, premises, conclusion):
        self.rules.append((premises, conclusion))

    def infer(self):
        inferred = True
        while inferred:
            inferred = False
            for premises, conclusion in self.rules:
                if all(p in self.facts for p in premises) and conclusion not in self.facts:
                    print(f"Inferred: {conclusion} from {premises}")
                    self.facts.add(conclusion)
                    inferred = True

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()

    # Add initial facts
    kb.add_fact("Human(Socrates)")
    kb.add_fact("Human(Plato)")

    # Add rules
    kb.add_rule(["Human(x)"], "Mortal(x)")  # This is a schema; we’ll simulate it manually

    # Simulate schema instantiation
    for fact in list(kb.facts):
        if fact.startswith("Human("):
            entity = fact[6:-1]
            kb.add_fact(f"Mortal({entity})")

    # Add a compound rule
    kb.add_rule(["Mortal(Socrates)", "Mortal(Plato)"], "AllPhilosophersAreMortal")

    # Run inference
    kb.infer()

    # Print final facts
    print("\nFinal Facts in Knowledge Base:")
    for fact in kb.facts:
        print(fact)
