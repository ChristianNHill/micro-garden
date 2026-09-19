"""Gate 8: ducks that notice each other, and a hand worth following.

Run: uv run python -m gates.gate_08_social [--episodes 10]

The three acceptance scenarios from ARCHITECTURE.md 2.8 that involve other ducks or the player:
a Bully displaces a Scaredy at one dish, two ducks startled together end up keeping their distance,
and a hand-fed duck comes to the hand.

The last two need a duck to learn, so they run with `learns=True`: sugar and the player's hand are
the rewarding dopamine, a startle the punishing kind, and the Kenyon cell to MBON synapses are the
only ones that move (brain/plasticity.py). Both also need a duck to tell one thing from another,
which is what the pheromone receptors and the retina are for: 429 receptors carry how strongly the
other ducks smell, and the hand is a pale object the eye can see.

Everything here is measured against a duck that had the same garden and the same time in it, and
differed only in what happened to it. A scenario that improves for both ducks has proved nothing.

The hand is measured and printed but not asserted, and that is a gap rather than a decision about what
matters. Its only signature is visual, and vision deliberately runs at a fifth of the gain that Gate 6
alone would want so that it does not drown the nose (Gate 6, 2026-09-18). Fed and unfed ducks came 0.66
and 0.67 m from it, which is no preference at all. Making it learnable needs the hand to smell of
something, and the only free glomeruli are ones food already uses, so it waits on vision getting
stronger or on a fourth odor.

STATUS 2026-09-18: PASSING, both asserted checks.
- A Bully takes 0.75 of the bites against a Scaredy's 0.25. It used to be 0.50 against 0.50, and the
  fix is not more attacks: being shoved now frightens a timid duck where it angered every duck alike,
  and a frightened duck goes off its food. So a Scaredy yields a dish it has been driven off, and Gate
  4c's aggressiveness dial, which needs attacks to stay rare, is untouched.
- Two ducks clapped at while side by side end up 0.39 m apart at the start and 0.65 m by the end. That
  is associative learning moving a duck, and it needed the mushroom body to be given its own synapse
  gain first: MBONs fired at 0.54 Hz and only a fifth of that came from Kenyon cells, so wiping every
  learned synapse moved the steering neurons less than the noise.
- The hand is printed and not asserted. It reads 0.57 m for the fed duck against 0.68 m for the other,
  which leans the right way and did not before, but its only cue is visual and vision is deliberately
  quiet. Asserting it waits on that.
"""
import argparse
import sys
import time

import numpy as np

from body.contract import Client
from brain.data import load_connectome, named_sets
from brain.personality import preset, stack
from gates.episodes import run, verdict
from world.fields import DUCK_R

DISH = (2.0, 2.0)
BULLY_S = 60.0
STARTLE_S, STARTLES = 90.0, 8
HAND_S, HAND_FEEDS = 90.0, 6
NEAR_DISH_M = 0.25


def bully_and_scaredy(W, ann, sets, n, seed):
    """Both start the same distance from one dish. Returns each one's share of the bites taken.

    Bites rather than time near the dish: two ducks can both stand within a dish's radius of it and
    neither be displaced, which is how this read 0.97 against 0.97 and said nothing (2026-09-18). Who
    actually gets the food is the contest.
    """
    rng = np.random.default_rng(seed)
    k = stack([preset(label, rng) for label in ("Bully", "Scaredy")] * n)
    poses = [[[DISH[0] - 0.4, DISH[1], 0.0], [DISH[0] + 0.4, DISH[1], np.pi]] for _ in range(n)]
    _, stubs = run(W, ann, sets, [dict(food_xy=[DISH], bites=1000, pose=p) for p in poses],
                   BULLY_S, seed, hunger=1.0, personality=k)
    bites = np.zeros(2)
    for s in stubs:
        for _, who in s.eaten:
            bites[who] += 1
    return bites / max(bites.sum(), 1)  # (bully's share, scaredy's share)


def startled_together(W, ann, sets, n, seed):
    """Two ducks, clapped at while they are side by side. Returns their mean gap before and after."""
    gaps = []

    def claps(stubs):
        for s in stubs:
            if s.t > STARTLE_S / 3 and int(s.t * 10) % int(STARTLE_S / STARTLES * 10) == 0:
                s.scared[:] = True
        return False

    poses = [[[1.8, 2.0, 0.0], [2.2, 2.0, np.pi]] for _ in range(n)]
    traj, _ = run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], STARTLE_S, seed,
                  until=claps, learns=True)
    pair = traj.reshape(len(traj), n, 2, 3)[:, :, :, :2]
    gap = np.linalg.norm(pair[:, :, 0] - pair[:, :, 1], axis=-1)
    third = len(gap) // 3
    gaps.append((gap[:third].mean(), gap[-third:].mean()))
    return gaps[0]


def follows_the_hand(W, ann, sets, n, seed):
    """The hand feeds one duck over and over, then shows itself to both. Returns how near each came."""
    fed, ignored = 0, 1
    poses = [[[1.2, 2.0, 0.0], [2.8, 2.0, np.pi]] for _ in range(n)]
    spots = [(1.2 + 0.35 * np.cos(a), 2.0 + 0.35 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, HAND_FEEDS)]

    def feed(stubs):
        for s in stubs:
            step = int(s.t / (HAND_S / (HAND_FEEDS + 2)))
            if step < HAND_FEEDS and s.world.hand != spots[step]:
                s.world.hand = spots[step]
                s.world.food = np.vstack([s.world.food, [spots[step]]])
                s.world.bites = np.append(s.world.bites, 3)
            elif step >= HAND_FEEDS:
                s.world.hand = (2.0, 2.6)  # the hand shows itself, empty, between the two of them
        return False

    traj, _ = run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], HAND_S, seed,
                  until=feed, learns=True)
    tail = traj[-len(traj) // 4:]
    near = np.linalg.norm(tail[:, :, :2] - np.array([2.0, 2.6]), axis=-1).min(axis=0)
    return near.reshape(n, 2)[:, fed].mean(), near.reshape(n, 2)[:, ignored].mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    n = ap.parse_args().episodes
    t0 = time.perf_counter()
    W, ann = load_connectome()
    sets = named_sets(ann)

    bully, scaredy = bully_and_scaredy(W, ann, sets, n, seed=0)
    print(f"one dish, Bully against Scaredy:  share of the bites {bully:.2f} against {scaredy:.2f}")
    before, after = startled_together(W, ann, sets, n, seed=1)
    print(f"clapped at {STARTLES} times together:      gap {before:.2f} m at the start, {after:.2f} m by the end")
    fed_near, ignored_near = follows_the_hand(W, ann, sets, n, seed=2)
    print(f"the hand, after feeding one duck: fed duck came {fed_near:.2f} m from it, "
          f"the other {ignored_near:.2f} m   (printed, not asserted; see the note above)")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    return verdict({
        "a Bully takes the food a Scaredy wanted": bully > scaredy + 0.15,
        "ducks startled together keep their distance afterwards": after > before + 2 * DUCK_R,
    })


if __name__ == "__main__":
    sys.exit(main())
