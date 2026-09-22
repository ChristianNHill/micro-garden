"""Sensory encoder: scalar levels on named sets -> input drive for one tick.

Two ways to deliver the same mean drive. `encode` fires a random subset of a set each tick. `graded`
releases steadily, like the optic lobe (brain/vision.py), with none of the sampling noise, so the same
odor lights the same Kenyon cells every time. The server uses `graded`.
"""
import numpy as np

from brain.lif import DT_MS

MAX_HZ = 50.0  # level 1.0 fires each neuron of the set at this rate (refractory caps it at 25 Hz at dt 10 ms)
SENSE_NOISE = 0.3  # how much of a spiking receptor's shot noise to keep; see `graded`


def graded(sets: dict[str, np.ndarray], levels: dict[str, np.ndarray | float], batch: int, device: str,
           rng: np.random.Generator | None = None, noise: float = SENSE_NOISE):
    """levels -> (neuron indices, release per brain) for LIF.step's `graded` argument.

    Release is the spike probability `encode` would have used, so the mean current is unchanged.
    Sets must not overlap: release is assigned, not summed.

    `noise` mixes back a spiking receptor's shot noise: 0 releases the mean exactly, 1 is `encode`'s
    all-or-nothing draw. With none, every duck behaves identically and graded knobs act as switches;
    with too much, a smell is no longer recognisable from one whiff to the next.
    """
    import torch

    idx = np.concatenate([sets[name] for name in levels]) if levels else np.empty(0, np.int64)
    def column(level):
        """One level per brain as a (batch, 1) column, whether it came in as a scalar or one per duck."""
        lv = np.asarray(level, float)
        return np.broadcast_to(lv.reshape(-1, 1) if lv.ndim else lv, (batch, 1))

    out = np.concatenate([column(level).repeat(len(sets[name]), 1) for name, level in levels.items()],
                         axis=1) if levels else np.zeros((batch, 0))
    release = np.clip(out * MAX_HZ * DT_MS / 1000, 0, 1)
    if noise and rng is not None:
        release = release + noise * ((rng.random(release.shape) < release) - release)
    release = release.astype(np.float32)
    return torch.from_numpy(idx.astype(np.int64)).to(device), torch.from_numpy(release).to(device)


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


def demo() -> None:
    """A level may be one number or one per duck, and both must land the same way."""
    sets = {"a": np.arange(3), "b": np.arange(3, 7)}
    idx, rel = graded(sets, {"a": 0.5, "b": 0.1}, 2, "cpu")
    assert idx.tolist() == list(range(7)), idx
    assert np.allclose(rel.numpy()[:, :3], 0.5 * MAX_HZ * DT_MS / 1000), rel
    idx, rel = graded(sets, {"a": np.array([0.2, 0.8])}, 2, "cpu")
    per_duck = rel.numpy()[:, 0]
    assert np.allclose(per_duck, np.array([0.2, 0.8]) * MAX_HZ * DT_MS / 1000), per_duck
    assert graded(sets, {}, 2, "cpu")[0].numel() == 0
    print(f"ok  scalar and per-duck levels both released correctly, {len(idx)} neurons")


if __name__ == "__main__":
    demo()
