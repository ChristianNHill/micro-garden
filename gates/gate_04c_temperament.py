"""Gate 4c: temperament. Food defense, retaliation and stink lovers come from personality knobs.

Run: uv run python -m gates.gate_04c_temperament [--episodes 20]
Knobs are scales, not switches (Chris, 2026-09-16): the dial sweeps check in-between settings give
in-between behavior.
Two ducks start nose to nose on a full dish, or facing each other with no food. The aggressive duck
alternates sides across episodes. Aggressiveness feeds the pC1d/e mood (brain/physiology.py) and the
brain's aIPg turns it into attacks (brain/decoder.py). The shuffled brain is printed beside the first
scenario. Stink affinity is a readout knob (brain/decoder.py).

STATUS 2026-09-18: PASSING, all eight.
- The dial reads the first blow as a rate rather than a count. A count could not grade: the pair spawns
  inside contact range, so every episode offers one opportunity and "did that land" is near-certain.
- ATTACK_P dropped from 0.1 a tick to 0.01. Ten strikes a second is not a duck, and every setting above
  aggression 0.2 was landing inside the ~0.7 s the touch rate needs to climb past TOUCH_HZ, which
  squashed the top of the dial. First blow now falls at 20.0, 18.1, 15.2, 10.8 and 9.1 s across the
  knob, and the raw count grades too at 0.0, 0.1, 0.25, 0.5, 0.6 where it had been pinned at 1.0 for
  the top three. That the count recovered on its own says the mechanism was at fault, not the metric.
- AGGR_FULL_HZ was re-anchored 1.0 -> 4.4 earlier in the same sitting: graded senses quadrupled aIPg's
  rate and the decoder was still normalising against what it reached when the senses were noisy.
- Worth an eye at the blind test: a duck at full aggressiveness now lands 0.6 headbutts in 20 s where
  it landed 1.0, because the pair only touches 13-20% of an episode. The dial grades, but the top end
  is a milder bully than it was.
"""
import argparse
import sys
import time

import numpy as np

from brain.data import load_connectome, named_sets, shuffled
from gates.episodes import ring_poses, run, verdict

DISH = (2.0, 2.0)
PAIR_S = 20.0
STINK_S = 60.0
STINK_NEAR_M = 0.5
DIAL = (0.0, 0.25, 0.5, 0.75, 1.0)


def graded(values, slack) -> bool:
    """A scale, not a switch: climbs from the first to the last setting, never drops by more than slack,
    and the middle setting lands between 20% and 80% of the way up."""
    v = np.asarray(values, float)
    mid = (v[len(v) // 2] - v[0]) / max(v[-1] - v[0], 1e-9)
    return v[-1] > v[0] and bool((np.diff(v) >= -slack).all()) and 0.2 <= mid <= 0.8


def pair(W, ann, sets, n, roles, hunger, provoked=0.0, food=True):
    """roles: (aggressiveness of the first-listed duck, of its partner). Returns per role:
    headbutts given and fraction of time on the dish."""
    swap = np.arange(n) % 2 == 1
    # Every episode meets at the dish from a different angle, with the headings a little off true.
    # They used to start identically, so the only thing separating one episode from the next was
    # sensory noise; once the senses stopped being noisy, all 20 behaved alike and the aggressiveness
    # dial collapsed into a switch (Gate 4c, 2026-09-18). Randomness belongs in the world.
    rng = np.random.default_rng(7)
    eps = []
    for _ in range(n):
        a, jitter = rng.uniform(-np.pi, np.pi), rng.normal(0, 0.25, 2)
        step = 0.065 * np.array([np.cos(a), np.sin(a)])
        eps.append(dict(food_xy=[DISH] if food else [], bites=1000, pose=[
            [DISH[0] - step[0], DISH[1] - step[1], a + jitter[0]],
            [DISH[0] + step[0], DISH[1] + step[1], a + np.pi + jitter[1]],
        ]))
    aggr = np.where(swap[:, None], roles[::-1], roles).ravel()
    prov = np.where(swap[:, None], (0.0, provoked), (provoked, 0.0)).ravel()
    traj, stubs = run(W, ann, sets, eps, PAIR_S, seed=0, aggressiveness=aggr, hunger=hunger, provoked=prov)
    hits = np.zeros((n, 2))
    first = np.full((n, 2), PAIR_S)  # never struck counts as the whole episode
    for e, st in enumerate(stubs):
        for t, i, _ in st.headbutts:
            hits[e, i] += 1
            first[e, i] = min(first[e, i], t)
    on = (np.linalg.norm(traj[:, :, :2] - DISH, axis=-1) < 0.15).mean(axis=0).reshape(n, 2)
    role = lambda x: np.where(swap[:, None], x[:, ::-1], x)  # column 0 = first role
    return role(hits).sum(axis=0), role(on).mean(axis=0), role(first).mean(axis=0)


def stink(W, ann, sets, n, affinity):
    poses = ring_poses(np.random.default_rng(0), n, DISH, STINK_NEAR_M)
    traj, _ = run(W, ann, sets, [dict(food_xy=[], danger_xy=[DISH], pose=p) for p in poses], STINK_S, seed=0,
                  stink_affinity=affinity)
    return (np.linalg.norm(traj[:, :, :2] - DISH, axis=-1) < STINK_NEAR_M).mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    n = ap.parse_args().episodes
    W, ann = load_connectome()
    sets = named_sets(ann)
    t0 = time.perf_counter()

    hungry_hits, hungry_on, _ = pair(W, ann, sets, n, (1.0, 0.0), hunger=1.0)
    print(f"hungry, aggressive vs meek:   headbutts {hungry_hits.astype(int).tolist()}  dish time {hungry_on.round(2).tolist()}")
    shuf_hits, shuf_on, _ = pair(shuffled(W, seed=0), ann, sets, n, (1.0, 0.0), hunger=1.0)
    print(f"  shuffled brain:             headbutts {shuf_hits.astype(int).tolist()}  dish time {shuf_on.round(2).tolist()}")
    fed_hits, _, _ = pair(W, ann, sets, n, (1.0, 0.0), hunger=0.0)
    print(f"fed, aggressive vs meek:      headbutts {fed_hits.astype(int).tolist()}")
    fight_hits, _, _ = pair(W, ann, sets, n, (1.0, 1.0), hunger=0.0, provoked=1.0, food=False)
    meek_hits, _, _ = pair(W, ann, sets, n, (1.0, 0.0), hunger=0.0, provoked=1.0, food=False)
    print(f"provoked attacker, no food:   vs aggressive {fight_hits.astype(int).tolist()}  vs meek {meek_hits.astype(int).tolist()}")
    averse, lover = stink(W, ann, sets, n, 0.0), stink(W, ann, sets, n, 1.0)
    print(f"time near stink:              averse {averse:.2f}  lover {lover:.2f}")
    dial_hits, dial_on, dial_first = zip(*[pair(W, ann, sets, n, (a, 0.0), hunger=1.0) for a in DIAL])
    dial_hits = [h[0] / n for h in dial_hits]
    dial_on = [o[0] for o in dial_on]
    # How soon the first blow lands, not how many land. The pair spawns inside contact range and the
    # first shove puts them out of it for the rest of the episode, so every episode offers exactly one
    # opportunity and a count can only ask whether that one landed: near-certain above about knob 0.2.
    # Latency has no such ceiling (Gate 4c, measured 2026-09-18).
    # As a rate, not a waiting time: the knob sets a chance of attacking per tick, and rate is its
    # linear image where latency is its reciprocal. Measured latencies 20.0, 13.4, 2.8, 0.8, 0.7 s put
    # the middle knob 89% of the way up as a speed but 22% as a rate, and a scale wants it in the middle.
    dial_rate = [1.0 / f[0] for f in dial_first]
    dial_stink = [stink(W, ann, sets, n, a) for a in DIAL]
    print(f"aggressiveness {DIAL}: first headbutt after {np.round([f[0] for f in dial_first], 1).tolist()} s"
          f"  headbutts per episode {np.round(dial_hits, 2).tolist()}  dish time {np.round(dial_on, 2).tolist()}")
    print(f"stink affinity {DIAL}: time near stink {np.round(dial_stink, 2).tolist()}")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    checks = {
        "hungry aggressive duck attacks more than the meek one": hungry_hits[0] > 2 * hungry_hits[1],
        "hungry aggressive duck holds the dish longer": hungry_on[0] > hungry_on[1],
        "fed aggressive duck attacks under a third as often as hungry": fed_hits[0] < hungry_hits[0] / 3,
        "provoked duck attacks": fight_hits[0] > 0 and meek_hits[0] > 0,
        "aggressive target hits back, meek target does not": fight_hits[1] > 0 and meek_hits[1] == 0,
        "stink lovers spend at least twice as long near stink": lover > 2 * averse,
        "attacks come faster as the aggressiveness dial rises": graded(dial_rate, slack=0.05),
        "time near stink grades with the stink-affinity dial": graded(dial_stink, slack=0.1),
    }
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
