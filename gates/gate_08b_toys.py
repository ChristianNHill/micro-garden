"""Gate 8b: music and hats, liked or not according to the duck.

Run: uv run python -m gates.gate_08b_toys [--episodes 8]

Nothing here is scripted as "likes music" or "likes hats". A duck hears music through the 1,103
Johnston's organ neurons its connectome has, with the sound falling off across the garden, and what it
does about that is its music-affinity knob: a taste for it turns toward the louder ear, no taste turns
away, and half way does neither. A hat sits on the head and is felt on all 1,417 bristles, the grooming
neurons answer the itch, and vanity is what stops a duck shaking it off.

The ball and the shiny rocks wait. Both are things a duck has to *see*, and vision runs at a fifth of
the gain Gate 6 alone would want so that it does not drown the nose, which is the same reason the hand
is not asserted at Gate 8. Music and hats ride hearing and touch, which carry.

Each check is a knob turned two ways in the same garden, so a result that moves both ducks proves
nothing.

STATUS 2026-09-18: PASSING, both checks.
- A duck that likes music ends up 1.79 m from the speaker, one that does not 2.28 m, and the shuffled
  brain 1.90 m in between. It starts 2.2 m away.
- A hat stays on 100% of the time at vanity 0.95 and 28% at 0.05, and the knob grades in between
  (29, 37, 56, 100 across the dial) rather than switching.
- The hat check failed twice first, both times on how a decision is made rather than on the duck. The
  hats were never put on, because the `until` hook runs after the first step and the test looked for
  t == 0. Then no hat survived its first second, because the shed was rolled every tick: six thousand
  chances a minute makes any probability a certainty. It is now decided every five seconds, the way
  wading already was, with a chance that scales with vanity.
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
                  until=play, music_affinity=affinity)  # half hungry, like any duck: a sated one stands and
    # listens (brain/physiology.py restlessness, 2026-09-19), and this asks where music steers a walking duck
    tail = traj[-len(traj) // 3:]
    return float(np.linalg.norm(tail[:, :, :2] - np.array(MUSIC_AT), axis=-1).mean())


def hat_time(W, ann, sets, n, seed, vanity):
    """Every duck starts in a hat. Returns the share of the time the hats stay on."""
    worn = []

    started = []

    def watch(stubs):
        for s in stubs:
            if not started:  # the hook runs after the first step, so t is already past zero here
                s.hats[:] = True  # every duck starts in one; whether it stays is the duck's business
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
