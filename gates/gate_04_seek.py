"""Gate 4: closed loop. Does one duck find food by smell, and does the real brain beat the shuffled one?

Run: uv run python -m gates.gate_04_seek [--episodes 20]
Each episode: one stub duck starts START_M from a single dish, facing a random way, and has MAX_S
to eat it. All episodes of one brain run side by side as one batch, in lockstep over the contract.
Asserts the real brain's median time to eat is under BOUND_S; the shuffled brain is printed only.

The garden has a breeze. The connectome has no usable left/right steering on food smell,
but it does steer upwind: DNge091 fires on the side the wind comes from. So each duck
starts downwind of the dish, within DOWNWIND_DEG of straight downwind, where there is a plume to
follow. Upwind of food there is nothing to smell. The shuffled brain's wind neurons are silent.
Caveat: the steering DNs were picked on the real brain, but the shuffled brain has almost no
descending neurons that respond to odor at all, so no readout choice would rescue it.
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
BOUND_S = 40.0  # set once from the first passing run (median 25.2 s) and not refitted since


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
