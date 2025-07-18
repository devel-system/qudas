class QdAnnBlock:
    def __init__(self, qubo: dict, label: str = "block"):
        self.qubo = qubo
        self.label = label