"""Gate 1: one brain ticks deterministically; five brains run faster than real time.

Run: uv run python -m gates.gate_01_tick
Drive: 20 Hz Poisson kicks into the sensory named sets (sugar, ORNs, bristle, JO, L1, L2).

Decision 2026-09-16 (Chris): torch-mps, dt 10 ms, SYN_GAIN 0.005, glutamate excitatory. Gate 4 model work
changed this to SYN_GAIN 0.01, glutamate inhibitory, and spike-frequency adaptation (5 brains: 2.6x).
Benchmark that informed it (5 brains, 10 simulated s): torch-mps 5.4x at 10 ms, 0.54x at 1 ms;
numpy event-driven 2.5x and 0.33x. numpy backend deleted.
"""
import sys
import time

import numpy as np

from brain.data import load_connectome, named_sets
from brain.encoder import MAX_HZ, encode
from brain.lif import DT_MS, LIF

INPUTS = ["sugar_grn", "orn_food", "orn_danger", "bristle", "johnstons_organ", "lamina_L1", "lamina_L2"]
DRIVE_HZ = 20.0
MIN_SPEED = 2.0  # x real time for 5 brains; margin for encoder, decoder and world


def run(W, sets, batch, ticks, seed):
    rng = np.random.default_rng(seed)
    brain = LIF(W, batch)
    levels = {k: DRIVE_HZ / MAX_HZ for k in INPUTS}
    t0 = time.perf_counter()
    for _ in range(ticks):
        brain.step(*encode(rng, sets, levels, batch))
    counts = brain.counts()  # syncs the device
    return counts, time.perf_counter() - t0


def main() -> int:
    W, ann = load_connectome()
    sets = named_sets(ann)
    targets = np.unique(np.concatenate([sets[k] for k in INPUTS]))
    driven = np.zeros(W.shape[0], bool)
    driven[targets] = True

    c1, _ = run(W, sets, 1, 100, seed=7)
    c2, _ = run(W, sets, 1, 100, seed=7)
    same = np.array_equal(c1, c2)
    print(f"determinism: repeat identical {same}  spikes {c1.sum():,}")

    sim_s = 10.0
    c, wall = run(W, sets, 5, int(sim_s * 1000 / DT_MS), seed=1)
    speed = sim_s / wall
    rate = c.sum() / c.size / sim_s
    und = c[:, ~driven].sum() / c[:, ~driven].size / sim_s
    print(f"5 brains, {sim_s:.0f} s at dt {DT_MS:.0f} ms: {speed:.2f}x real time  mean {rate:.2f} Hz  "
          f"undriven {und:.3f} Hz  ever spiked {(c > 0).any(axis=0).mean():.1%}")

    checks = {
        "deterministic": same,
        f"speed >= {MIN_SPEED}x": speed >= MIN_SPEED,
        "not silent or saturated": 0.01 < rate < 100,
        "activity propagates past kicked neurons": und > 0.1,
    }
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
