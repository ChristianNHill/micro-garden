"""Gate 8b: music and hats, liked or not according to the duck.

Run: uv run python -m gates.gate_08b_toys [--episodes 8]

Nothing here is scripted as "likes music" or "likes hats". A duck hears music through its Johnston's
organ neurons, with the sound falling off across the garden, and its music-affinity knob decides what it
does: a taste for it turns toward the louder ear, no taste turns away, half way does neither. A hat is
felt on all the bristles, the grooming neurons answer the itch, and vanity stops a duck shaking it off.

The ball and shiny rocks are not tested: both need vision, which runs at low gain so it does not drown
the nose. Music and hats use hearing and touch, which carry.

Each check is a knob turned two ways in the same garden, so a result that moves both ducks proves
nothing. The shuffled brain's music result is printed only.
"""
import argparse
import sys
import time

import numpy as np

from brain.data import load_connectome, named_sets, shuffled
from gates.episodes import run, verdict

MUSIC_AT = (3.2, 2.0)
MUSIC_S = 45.0
HAT_S = 60.0
START = [[1.0, 2.0, 0.0]]


def music_walk(W, ann, sets, n, seed, affinity):
    """Ducks start across the garden from a speaker. Returns how near they end up to it."""
    def play(stubs):
        for s in stubs:
            s.world.music = MUSIC_AT
        return False

    traj, _ = run(W, ann, sets, [dict(food_xy=[], pose=START) for _ in range(n)], MUSIC_S, seed,
                  until=play, music_affinity=affinity)  # half hungry: a sated duck stands and
    # listens, and this asks where music steers a walking duck
    tail = traj[-len(traj) // 3:]
    return float(np.linalg.norm(tail[:, :, :2] - np.array(MUSIC_AT), axis=-1).mean())


def hat_time(W, ann, sets, n, seed, vanity):
    """Every duck starts in a hat. Returns the share of the time the hats stay on."""
    worn = []

    started = []

    def watch(stubs):
        for s in stubs:
            if not started:  # the hook runs after the first step, so t is already past zero here
                s.hats[:] = True
            worn.append(s.hats.mean())
        started.append(True)
        return False

    run(W, ann, sets, [dict(food_xy=[], pose=START) for _ in range(n)], HAT_S, seed,
        until=watch, vanity=vanity)
    return float(np.mean(worn)) if worn else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    n = ap.parse_args().episodes
    t0 = time.perf_counter()
    W, ann = load_connectome()
    sets = named_sets(ann)

    loves = music_walk(W, ann, sets, n, 0, affinity=0.95)
    hates = music_walk(W, ann, sets, n, 0, affinity=0.05)
    shuf = music_walk(shuffled(W, seed=0), ann, sets, n, 0, affinity=0.95)
    print(f"music at {MUSIC_AT}, starting {np.linalg.norm(np.array(START[0][:2]) - MUSIC_AT):.1f} m away:")
    print(f"  ends up {loves:.2f} m from it liking music, {hates:.2f} m disliking it, {shuf:.2f} m shuffled")

    vain = hat_time(W, ann, sets, n, 1, vanity=0.95)
    plain = hat_time(W, ann, sets, n, 1, vanity=0.05)
    print(f"a hat stays on {100 * vain:.0f}% of the time on a vain duck, {100 * plain:.0f}% on one that is not")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    return verdict({
        "a duck that likes music ends up nearer it than one that does not": loves < hates - 0.2,
        "a hat stays on a vain duck longer": vain > plain + 0.1,
    })


if __name__ == "__main__":
    sys.exit(main())
