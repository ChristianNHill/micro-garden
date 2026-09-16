"""Gate 4: closed loop. Does one duck find food by smell, and does the real brain beat the shuffled one?

Run: uv run python -m gates.gate_04_seek [--episodes 20]
Each episode: one stub duck starts START_M from a single dish, facing a random way, and has MAX_S
to eat it. All episodes of one brain run side by side as one batch, in lockstep over the contract.

2026-09-16 record. First run (gain 0.005, no adaptation, DNa02 only): real 2/20, shuffled 0/20,
no odor steering. Chris chose deeper model work. Changes, each measured in scratch sweeps:
- spike-frequency adaptation (ADAPT_INC 1.0) and SYN_GAIN 0.01: activity stops after odor stops
- ORN_CONTRA 0.3: left/right odor difference now reaches PNs, lateral horn and 41 descending neurons
  (replicated on held-out seeds); steering readout adds DNb05 and DNp05 from that screen
- ODOR_CONTRAST 8 in the encoder: steering DNs need about 3:1, antennae see about 1.17:1
- escape needs 2 giant fiber spikes in 100 ms; turning gain 1.0 rad/s per Hz, wander 0.5
Result: real 20/20, median 25.2 s; shuffled 0/20. After lockstep began waiting for each new frame
(stale UDP frames made repeat runs differ), real 20/20, median 24.4 s, identical across runs. Caveat: the steering DNs were picked on the real
brain. The same screen on the shuffled brain finds only 2 descending neurons that respond to odor at all
(real: 364), so no readout choice would rescue it.
Decision (Chris, 2026-09-16): keep the "fly brain" label for smell, with the assumptions above stated.
"""
import argparse
import sys
import tempfile
import time

import numpy as np

from body.contract import Client
from body.stub2d.stub import DT, Stub
from brain.data import load_connectome, named_sets, shuffled
from brain.server import BrainServer

DISH = (2.0, 2.0)
START_M, MAX_S = 1.5, 120.0
PORT_BASE = 7700
BOUND_S = 40.0  # set once from the first passing full run (real median 25.2 s), 2026-09-16


def episodes(W, ann, sets, n: int, seed: int) -> np.ndarray:
    """Time to eat per episode, inf if never."""
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, n)
    poses = np.column_stack([DISH[0] + START_M * np.cos(angle), DISH[1] + START_M * np.sin(angle),
                             rng.uniform(-np.pi, np.pi, n)])
    dirs = [tempfile.TemporaryDirectory(prefix="mg") for _ in range(n)]
    stubs = [Stub(1, seed + e, d.name, food_xy=[DISH], frame_port=PORT_BASE + e, pose=poses[e])
             for e, d in enumerate(dirs)]
    ctls = [Client(f"{d.name}/control.sock") for d in dirs]
    server = BrainServer(W, ann, sets, [(f"{d.name}/duck-a.sock", PORT_BASE + e) for e, d in enumerate(dirs)], seed)
    for c in ctls:
        c.call("sim.step", n=0)
    ate = np.full(n, np.inf)
    for step in range(int(MAX_S / DT)):
        server.step(lockstep=True)
        for e, c in enumerate(ctls):
            if c.call("sim.step", n=1)["eaten"] and ate[e] == np.inf:
                ate[e] = (step + 1) * DT
        if np.isfinite(ate).all():
            break
    server.close()
    for c in ctls:
        c.close()
    for s in stubs:
        s.close()
    for d in dirs:
        d.cleanup()
    return ate


def summary(t: np.ndarray) -> str:
    found = np.isfinite(t)
    return (f"found {found.sum()}/{len(t)}  median {np.median(t):.1f} s  "
            f"times {np.round(np.sort(t[found]), 1).tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    W, ann = load_connectome()
    sets = named_sets(ann)
    results = {}
    for label, M in (("real", W), ("shuffled", shuffled(W, seed=0))):
        t0 = time.perf_counter()
        results[label] = episodes(M, ann, sets, args.episodes, seed=0)
        print(f"{label:9s} {summary(results[label])}  ({time.perf_counter() - t0:.0f} s wall)", flush=True)

    real = np.median(results["real"])
    if BOUND_S is None:
        print("BOUND_S not set yet: record the real median above and commit a bound")
        return 1
    ok = real < BOUND_S
    print(f"  {'ok  ' if ok else 'FAIL'} real median {real:.1f} s < bound {BOUND_S} s")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
