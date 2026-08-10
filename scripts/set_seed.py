"""Deterministic seeds (reproducibility gate)."""
import random, numpy as np, torch
from doc_agent import config

def main() -> None:
    s = config.load()["seed"]
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    main()
