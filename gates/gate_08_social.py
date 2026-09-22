"""Gate 8: ducks that notice each other, and a hand worth following.

Run: uv run python -m gates.gate_08_social [--episodes 10]

The three acceptance scenarios that involve other ducks or the player:
a Bully displaces a Scaredy at one dish, two ducks startled together end up keeping their distance,
and a hand-fed duck comes to the hand.

The last two need a duck to learn, so they run with `learns=True`: sugar and the player's hand are
the rewarding dopamine, a startle the punishing kind, and the Kenyon cell to MBON synapses are the
only ones that move (brain/plasticity.py). Both also need a duck to tell one thing from another:
429 pheromone receptors carry how strongly the other ducks smell, and the hand is a pale object the
eye can see.

Everything is measured against a yoked duck that had the same garden and the same time in it, and
differed only in what happened to it. Two ducks set down face to face wander apart whatever happens,
so a pair compared with its own start proves nothing.

Asserted: the pair clapped at 8 times ends further apart than a yoked pair nobody clapped at, by more
than two duck radii. Last run: 1.57 m against 0.91 m. The learning reaches behavior
through two explicit readouts. Companionship turns a duck toward or away from the side other ducks
smell stronger on, by its sociability. Fondness is what it has learned about that smell, read from the
mushroom body's own synapses, and it adds to sociability. Nothing in the wiring itself carries
mushroom-body output to a left or a right. One smell stands for every other duck, so this is wariness
of ducks, not a grudge against one; a grudge needs each duck to smell of itself.

Printed, not asserted:
- The dish contest. A Bully that costs itself food, by turning on the other duck while the quicker
  Scaredy eats, is emergent behavior and stays. The share shows who a Bully is, not a fault to fix.
- The hand. That is a gap, not a decision about what matters. Its only cue is visual, and vision runs
  at a fifth of Gate 6's gain so it does not drown the nose; fed and unfed ducks come about equally
  near. It waits on stronger vision or a fourth odor (the free glomeruli are ones food already uses).
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


def bully_and_scaredy(W, ann, sets, n, seed):
    """Both start the same distance from one dish. Returns each one's share of the bites taken.

    Bites rather than time near the dish: two ducks can both stand within a dish's radius and neither
    be displaced. Who actually gets the food is the contest.
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


def startled_together(W, ann, sets, n, seed, clap=True):
    """Two ducks, clapped at while they are side by side. Returns their mean gap before and after.
    clap=False is the yoked pair: the same ducks, garden and time, and nothing happens to them."""
    gaps = []

    def claps(stubs):
        for s in stubs:
            if clap and s.t > STARTLE_S / 3 and int(s.t * 10) % int(STARTLE_S / STARTLES * 10) == 0:
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

    fed_at = {}  # garden -> the last spot it was fed at

    def feed(stubs):
        # The hand is put back every step, since a hand leaves the garden by itself after a few seconds, and
        # each spot is fed once: judged by where the hand was, a hand that had left looked new every step and
        # the gate put down a fruit a step, thousands of them (2026-09-22).
        for g, s in enumerate(stubs):
            step = int(s.t / (HAND_S / (HAND_FEEDS + 2)))
            if step < HAND_FEEDS:
                s.world.hand = spots[step]
                if fed_at.get(g) != step:
                    fed_at[g] = step
                    s.world.add_food(spots[step], 3)
            else:
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
    print(f"one dish, Bully against Scaredy:  share of the bites {bully:.2f} against {scaredy:.2f}"
          "   (printed, not asserted: Chris, 2026-09-20)")
    before, after = startled_together(W, ann, sets, n, seed=1)
    _, left_alone = startled_together(W, ann, sets, n, seed=1, clap=False)
    print(f"clapped at {STARTLES} times together:      gap {before:.2f} m at the start, {after:.2f} m by the end; "
          f"a pair nobody clapped at ends {left_alone:.2f} m apart")
    fed_near, ignored_near = follows_the_hand(W, ann, sets, n, seed=2)
    print(f"the hand, after feeding one duck: fed duck came {fed_near:.2f} m from it, "
          f"the other {ignored_near:.2f} m   (printed, not asserted; see the note above)")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    return verdict({
        "ducks startled together end up further apart than a pair left alone": after > left_alone + 2 * DUCK_R,
    })


if __name__ == "__main__":
    sys.exit(main())
