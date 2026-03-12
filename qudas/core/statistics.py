import numpy as np

def energy_statistics(energies):
    return {
        "mean": float(np.mean(energies)),
        "variance": float(np.var(energies)),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
    }

def probability_statistics(counts):
    shots = sum(counts.values())
    probs = np.array([v / shots for v in counts.values()])
    return {
        "variance": float(np.var(probs)),
        "std": float(np.std(probs)),
    }