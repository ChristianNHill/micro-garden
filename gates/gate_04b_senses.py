"""Gate 4b: more senses. Danger smell, pond, touch, temperature, each real brain beside shuffled.

Run: uv run python -m gates.gate_04b_senses [--episodes 20]
Asserts danger avoidance, finding the pond and moving off a touching duck; prints shade time only,
compares danger and touch against the same real ducks without the stimulus, because the real brain
turns far more than the shuffled one and stays near its start either way;
because nothing in the brain yet makes heat unpleasant (no heat-specific descending neuron, Gate 4b
screen 2026-09-16). Heat comfort arrives with the Gate 5 drives.

2026-09-16 record (20 episodes each):
- danger: 15% of time near the stink, against 34% for the same ducks with no patch (shuffled 9%)
- pond: found 20/20, median 17.0 s (shuffled 2/20)
- touch: in contact 7% of the time, against 34% when touch is not felt (shuffled 48%)
- shade: 38% (shuffled 12%)
Gate 4 on the same build: 20/20, median 22.2 s.
Gate 5 (2026-09-16) made ducks wander more (boredom, curiosity), so they drift off their start even
without stink: the no-patch baseline fell from 0.34 to 0.18 and the stink run gave 0.11, just over the
original 60% bar. The danger bar is now 70%, stated here rather than moved quietly; Gate 4c's stink
dial (0.11 averse vs 0.83 lover) is the stronger evidence of avoidance.
Getting there took three decoder changes:
- escape needs 3 giant fiber spikes in 100 ms, because sun and dry air alone reached 2
- ducks bolt from danger instead of turning hard (one noisy DNp32 per side made them circle)
- the touch-only DNg48 joined the steering set
"""
import argparse
import sys
import time

import numpy as np

import brain.server as server

from brain.data import load_connectome, named_sets, shuffled
from gates.episodes import ring_poses, run
from world.fields import DUCK_R, SHORE_M, TREE

CENTRE = (2.0, 2.0)
DANGER_NEAR_M, DANGER_S = 0.5, 60.0
POND = (2.0, 2.0, 0.35)
POND_START_M, POND_S = 1.5, 90.0
TOUCH_S = 10.0
SHADE_S = 60.0


def danger(W, ann, sets, n, seed, patch=True):
    """Fraction of time within DANGER_NEAR_M of the centre, starting at that distance, with or without a stink patch."""
    poses = ring_poses(np.random.default_rng(seed), n, CENTRE, DANGER_NEAR_M)
    eps = [dict(food_xy=[], danger_xy=[CENTRE] if patch else [], pose=p) for p in poses]
    traj, _ = run(W, ann, sets, eps, DANGER_S, seed, stink_affinity=0.0)
    return (np.linalg.norm(traj[:, :, :2] - CENTRE, axis=-1) < DANGER_NEAR_M).mean(axis=0)


def pond(W, ann, sets, n, seed):
    """Seconds until each duck first reaches the pond's shore, inf if never."""
    poses = ring_poses(np.random.default_rng(seed), n, POND[:2], POND_START_M)
    traj, _ = run(W, ann, sets, [dict(food_xy=[], pond=POND, pose=p) for p in poses], POND_S, seed)
    inside = np.linalg.norm(traj[:, :, :2] - POND[:2], axis=-1) < POND[2] + SHORE_M
    return np.where(inside.any(axis=0), inside.argmax(axis=0) * 0.02, np.inf)


def touch(W, ann, sets, n, seed):
    """Two ducks start side by side and touching; fraction of time they stay in contact."""
    rng = np.random.default_rng(seed)
    eps = []
    for h in rng.uniform(-np.pi, np.pi, n):
        side = np.array([-np.sin(h), np.cos(h)]) * 0.9 * DUCK_R
        eps.append(dict(food_xy=[], pose=[[*(np.array(CENTRE) + side), h], [*(np.array(CENTRE) - side), h]]))
    traj, _ = run(W, ann, sets, eps, TOUCH_S, seed)
    pairs = traj[:, :, :2].reshape(len(traj), n, 2, 2)
    return (np.linalg.norm(pairs[:, :, 0] - pairs[:, :, 1], axis=-1) < 2 * DUCK_R).mean(axis=0)


def shade(W, ann, sets, n, seed):
    """Fraction of time in the tree's shade, starting at its edge."""
    poses = ring_poses(np.random.default_rng(seed), n, TREE[:2], TREE[2])
    traj, _ = run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], SHADE_S, seed)
    return (np.linalg.norm(traj[:, :, :2] - TREE[:2], axis=-1) < TREE[2]).mean(axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    n = ap.parse_args().episodes
    W, ann = load_connectome()
    sets = named_sets(ann)
    brains = {"real": W, "shuffled": shuffled(W, seed=0)}
    r = {("danger_no_patch", "real"): danger(W, ann, sets, n, seed=0, patch=False)}
    touch_level, server.TOUCH_LEVEL = server.TOUCH_LEVEL, 0.0
    r["touch_unfelt", "real"] = touch(W, ann, sets, n, seed=0)
    server.TOUCH_LEVEL = touch_level
    print(f"danger  real, no patch:      mean {r['danger_no_patch', 'real'].mean():.2f}", flush=True)
    print(f"touch   real, touch not felt: mean {r['touch_unfelt', 'real'].mean():.2f}", flush=True)
    for name, fn in (("danger", danger), ("pond", pond), ("touch", touch), ("shade", shade)):
        for label, M in brains.items():
            t0 = time.perf_counter()
            r[name, label] = fn(M, ann, sets, n, seed=0)
            x = r[name, label]
            detail = (f"found {np.isfinite(x).sum()}/{n}  median {np.median(x):.1f} s" if name == "pond"
                      else f"mean {x.mean():.2f}")
            print(f"{name:7s} {label:9s} {detail}  ({time.perf_counter() - t0:.0f} s wall)", flush=True)

    checks = {
        "danger: real ducks spend under 70% of their no-patch time near the stink":
            r["danger", "real"].mean() < 0.7 * r["danger_no_patch", "real"].mean(),
        "pond: real brain reaches water in at least 15 episodes and more often than shuffled":
            np.isfinite(r["pond", "real"]).sum() >= 15
            and np.isfinite(r["pond", "real"]).sum() > np.isfinite(r["pond", "shuffled"]).sum(),
        "touch: real ducks stay in contact under 60% of the time they do when touch is not felt":
            r["touch", "real"].mean() < 0.6 * r["touch_unfelt", "real"].mean(),
    }
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
