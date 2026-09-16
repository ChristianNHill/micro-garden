"""Sensory encoder: scalar levels on named sets -> Poisson input spikes for one tick (PLAN.md Gate 2)."""
import numpy as np

from brain.lif import DT_MS

MAX_HZ = 50.0  # level 1.0 fires each neuron of the set at this rate (refractory caps it at 25 Hz at dt 10 ms)


def encode(rng: np.random.Generator, sets: dict[str, np.ndarray], levels: dict[str, np.ndarray | float], batch: int):
    """levels maps set name -> level in [0, 1], scalar or one per brain. Returns (brain idx, neuron idx) to kick."""
    bs, ns = [np.empty(0, np.int64)], [np.empty(0, np.int64)]
    for name, level in levels.items():
        targets = sets[name]
        for b, x in enumerate(np.broadcast_to(level, batch)):
            k = rng.binomial(len(targets), min(float(x) * MAX_HZ * DT_MS / 1000, 1.0))
            ns.append(rng.choice(targets, k, replace=False))
            bs.append(np.full(k, b))
    return np.concatenate(bs), np.concatenate(ns)
