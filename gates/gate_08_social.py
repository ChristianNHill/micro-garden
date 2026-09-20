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

DECIDED 2026-09-20 (Chris): a Bully that costs itself food is emergent behaviour and stays. The dish
contest is printed, not asserted; what it shows is who a Bully is, not a fault to fix.

2026-09-20, later, on why the startled pair learns nothing that shows. Two things, found on a bench
(trained brains smell a duck while startled eight times, yoked controls only smell it, then the smell is
put on one side, mirrored). First, an event in the garden delivered dopamine for one body step, 20 ms, and
eight of them left the weights at 0.996 of baseline; `brain/plasticity.py` now gives each burst the second
a real one lasts, and the same eight startles cut the MBONs' answer to that duck's smell by a third, 1.17
to 0.78 Hz. So a duck does now learn who it was startled beside. Second, that lesson stops at the MBONs:
the turn toward the other duck reads +0.238 Hz trained against +0.234 control, and no descending type
changes beyond chance. Nothing in this wiring carries mushroom-body output to a left or a right. A duck
that acts on what it learned needs an explicit readout of MBON valence, as the stink and the music have,
and that is a feature for Chris to want or not. Until then this check fails and says so.

STATUS 2026-09-20: FAILING, both asserted checks, and neither of the 09-18 passes below survives a look.
- The Bully's 0.75 of the bites rested on a bug. A full duck kept tasting food (taste had a floor), so
  both stood at the 1000-bite dish for the whole minute and the Bully had all that time to drive the
  Scaredy off. With taste gated on hunger a starving duck eats its fill in about five seconds and walks
  away, the two are at the dish together for those seconds only, and nobody landed a headbutt in ten
  episodes: 0.47 against 0.53. Making the dish scarce (10 bites, one duck's fill) does not rescue it:
  the Scaredy takes 59 bites to the Bully's 30, because the Bully spends its seconds at the dish
  turning on the other duck (2 headbutts in ten episodes) while the quicker Scaredy eats. Whether and how
  a Bully gets food by bullying is a question about the aggression design, and it is Chris's.
- The startled pair's widening gap was never learning. It compared a pair with its own start, and two
  ducks set down face to face wander apart whatever happens: learning and clapped at +0.13 m, not
  learning and clapped at +0.17 m, learning and never clapped at +0.29 m (10 pairs each, s.e. 0.09 to
  0.19). The check is now against a yoked pair that is never clapped at, and it fails, which is the
  honest reading: there is no evidence yet that a duck learns to avoid the one it was startled beside.

STATUS 2026-09-18 (superseded): PASSING, both asserted checks.
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
