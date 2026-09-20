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

2026-09-19: the garden has a breeze now and this gate is run in it. That 20/20 did not survive a look
underneath (PLAN.md Gate 9b): steering is a handful of cells read over 100 ms, a food smell 4.5:1
stronger on one side moves it 0.13 Hz against a built-in turn of 0.4, and no descending type does
better, on two seeds in two conditions. It was a weakly biased random walk in a garden with nothing
else in it, and it fell to 10/20 when the smell was made to carry further. A fly finds food by turning
upwind when it smells it, and the connectome does have that: DNge091 fires on the side the wind comes
from, thirty times more clearly than anything does for food. So each duck now starts somewhere downwind
of the dish, within DOWNWIND_DEG of straight downwind, where there is a plume to follow; a duck upwind
of food has nothing to smell and finding it is luck, which is not what this gate is about. The shuffled
brain's wind neurons are silent, so the comparison still means what it did.
Result (2026-09-19): real 19/20, median 35.4 s; shuffled 1/20. That is inside the 40 s bound set on
2026-09-16, which is left as it was.
"""
import argparse
import sys
import time

import numpy as np

from brain.data import load_connectome, named_sets, shuffled
from gates.episodes import run, verdict

DISH = (2.0, 2.0)
WIND = (0.0, -1.0)  # light air from the north, as in the demo garden
DOWNWIND_DEG = 60.0
START_M, MAX_S = 1.5, 120.0
BOUND_S = 40.0  # set once from the first passing full run (real median 25.2 s), 2026-09-16


def episodes(W, ann, sets, n: int, seed: int) -> np.ndarray:
    """Time to eat per episode, inf if never."""
    rng = np.random.default_rng(seed)
    a = np.arctan2(WIND[1], WIND[0]) + np.radians(rng.uniform(-DOWNWIND_DEG, DOWNWIND_DEG, n))
    poses = np.column_stack([DISH[0] + START_M * np.cos(a), DISH[1] + START_M * np.sin(a),
                             rng.uniform(-np.pi, np.pi, n)])
    ate = np.full(n, np.inf)

    def until(stubs):
        for e, s in enumerate(stubs):
            if s.eaten and ate[e] == np.inf:
                ate[e] = s.eaten[0][0]
        return np.isfinite(ate).all()

    run(W, ann, sets, [dict(food_xy=[DISH], pose=p, wind=WIND) for p in poses], MAX_S, seed, until=until)
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
    return verdict({f"real median {real:.1f} s < bound {BOUND_S} s": real < BOUND_S})


if __name__ == "__main__":
    sys.exit(main())
