"""Gate 9b: the whole garden, left alone for two days, and nobody gets stuck.

Run: uv run python -m gates.gate_09b_soak [--minutes 20] [--gardens 3] [--labels Bully,Napper,...]

Every other gate proves one piece in a garden built for that piece. This one puts a pond, a dish, a
stink patch, a tree and five different ducks in one place, which is the garden anyone will look at. It
catches "stay here" states that no other need can break, such as a sated duck that still tastes the
shore as food and parks at the pond. No single-piece gate can see one.

This is the demo garden (`body/stub2d/stub.py` DEMO_GARDEN) with the demo's five labels, eyes open,
for two whole days at DAY_S. It asserts what a garden worth watching cannot do without:

- every duck eats, drinks and sleeps at some point
- no duck is stuck: awake, in real need (hunger or thirst over NEEDY), and going nowhere (within
  STUCK_M for STUCK_S) without a bite or a sip to show for it
- no label starves or parches: hunger or thirst above PINNED for more than PINNED_SHARE of the run

and prints where each duck's time went, which is the table to read for "does anything happen".

Several gardens run side by side, the same five ducks starting in different places. One garden is one
draw from a chaotic system (sips have gone 155 to 48 between near-identical runs), and nothing can be
tuned against a number that loose. Eating, drinking, sleeping and being stuck are checked for every
duck in every garden. Starving and parching are checked on each label's mean over the gardens, because
a "no single duck" bar gets stricter with every garden added; the single ducks over the limit are
printed beside it. The mean still fails a label that starves as a rule.

The Bully is exempt from the starving limit by name (HUNGRY_BY_NATURE): its hunger is character, not a
fault. It still has to eat, drink and sleep like everyone else.
"""
import argparse
import sys
import time

import numpy as np

from body.stub2d.stub import DEMO_GARDEN, DT
from brain.data import load_connectome, named_sets
from brain.personality import preset, stack
from gates.episodes import run, verdict

LABELS = "Bully,Napper,Carefree,Chatty,Scaredy"  # the demo's
NEEDY = 0.8
STUCK_S, STUCK_M = 120, 0.2
PINNED, PINNED_SHARE = 0.95, 0.25
# The Bully goes hungry because it would rather fight than eat: it turns on a rival at the food and needs
# 40% more than anyone. That is emergent and it stays, so the Bully is held to eating at all, not to the
# starving limit.
HUNGRY_BY_NATURE = {"Bully"}
MOVING_MS = 0.02


def soak(W, ann, sets, labels, minutes, seed, gardens):
    """One row per simulated second, per duck; ducks of garden g are columns g * len(labels) onward."""
    n = len(labels) * gardens
    rng = np.random.default_rng(seed)
    pose = np.column_stack([rng.uniform(0.5, DEMO_GARDEN["size"] - 0.5, (n, 2)), rng.uniform(-np.pi, np.pi, n)])
    log = {k: [] for k in ("xy", "asleep", "hunger", "thirst", "swimming", "sips", "fear", "following")}
    sips, steps = np.zeros(n), [0]

    def watch(server, stubs):
        f = server.frames
        sips[:] += [x["drank"] > 0 for x in f]
        steps[0] += 1
        if steps[0] % int(1 / DT):
            return
        b = server.body
        log["xy"].append(np.concatenate([s.pose[:, :2] for s in stubs]))
        log["swimming"].append(np.array([x["swimming"] > 0 for x in f]))
        for k, v in (("asleep", b.asleep), ("hunger", b.hunger), ("thirst", b.thirst), ("sips", sips),
                     ("fear", b.fear), ("following", server.following)):
            log[k].append(np.array(v, float))

    per = len(labels)
    _, stubs = run(W, ann, sets, [dict(DEMO_GARDEN, pose=pose[g * per:(g + 1) * per]) for g in range(gardens)],
                   minutes * 60, seed, watch=watch, personality=stack([preset(x, rng) for x in labels * gardens]))
    out = {k: np.array(v) for k, v in log.items()}
    bites = np.zeros((len(out["xy"]), n))
    for g, stub in enumerate(stubs):
        for t, duck in stub.eaten:
            bites[min(int(t), len(bites) - 1):, g * per + duck] += 1
    out["bites"] = bites
    return out


def stuck_seconds(d) -> np.ndarray:
    """Per duck: seconds spent awake, needy, and within STUCK_M of where it was STUCK_S ago, with
    nothing eaten or drunk in between."""
    xy, fed = d["xy"], d["bites"] + d["sips"]
    went = np.linalg.norm(xy[STUCK_S:] - xy[:-STUCK_S], axis=-1) < STUCK_M
    unfed = fed[STUCK_S:] == fed[:-STUCK_S]
    need = np.maximum(d["hunger"], d["thirst"])
    # needy and awake for the whole window, not just at its end
    bad = (need > NEEDY) & (d["asleep"] == 0)
    whole = np.array([bad[i:i + STUCK_S].all(axis=0) for i in range(len(bad) - STUCK_S)])
    return (went & unfed & whole).sum(axis=0)


def describe_stuck(d, labels, i) -> str:
    """Where a stuck duck was and what state it was in, over its first stuck two minutes."""
    xy, fed = d["xy"], d["bites"] + d["sips"]
    bad = (np.maximum(d["hunger"], d["thirst"]) > NEEDY) & (d["asleep"] == 0)
    for t in range(len(xy) - STUCK_S):
        if (np.linalg.norm(xy[t + STUCK_S, i] - xy[t, i]) < STUCK_M and fed[t + STUCK_S, i] == fed[t, i]
                and bad[t:t + STUCK_S, i].all()):
            w = slice(t, t + STUCK_S)
            path = np.linalg.norm(np.diff(xy[w, i], axis=0), axis=1).sum()
            return (f"{labels[i % len(labels)]} in garden {i // len(labels)} from {t} s at ({xy[t, i, 0]:.2f}, {xy[t, i, 1]:.2f}): "
                    f"walked {path:.1f} m going nowhere, hunger {d['hunger'][w, i].mean():.2f}, thirst "
                    f"{d['thirst'][w, i].mean():.2f}, fear {d['fear'][w, i].mean():.2f}, following a plume "
                    f"{d['following'][w, i].mean():.2f}")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=20.0, help="simulated; 10 is one day and night")
    ap.add_argument("--labels", default=LABELS)
    ap.add_argument("--gardens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    labels = args.labels.split(",")
    t0 = time.perf_counter()
    W, ann = load_connectome()
    d = soak(W, ann, named_sets(ann), labels, args.minutes, args.seed, args.gardens)
    per, G = len(labels), args.gardens
    mean = lambda x: np.asarray(x, float).reshape(G, per).mean(0)  # per label, over gardens

    speed = np.linalg.norm(np.diff(d["xy"], axis=0), axis=-1)
    shade = np.linalg.norm(d["xy"] - np.array(DEMO_GARDEN["tree"][:2]), axis=-1) < DEMO_GARDEN["tree"][2]
    print(f"{args.minutes:g} simulated minutes in {G} demo gardens, share of each duck's time, mean over gardens:")
    print(f"  {'':10s} {'asleep':>7s} {'moving':>7s} {'swims':>7s} {'shade':>7s} {'hungry':>7s} {'thirsty':>7s}"
          f" {'bites':>6s} {'sips':>6s} {'stuck s':>8s}")
    stuck = stuck_seconds(d) if len(d["xy"]) > STUCK_S else np.zeros(per * G)
    cols = [d["asleep"].mean(0), (speed > MOVING_MS).mean(0), d["swimming"].mean(0), shade.mean(0),
            (d["hunger"] > PINNED).mean(0), (d["thirst"] > PINNED).mean(0)]
    for i, label in enumerate(labels):
        print(f"  {label:10s} " + " ".join(f"{mean(c)[i]:7.2f}" for c in cols)
              + f" {mean(d['bites'][-1])[i]:6.1f} {mean(d['sips'][-1])[i]:6.1f} {mean(stuck)[i]:8.0f}")
    print(f"soak: {d['bites'][-1].sum() / G:.0f} bites and {d['sips'][-1].sum() / G:.0f} sips a garden, "
          f"{int(stuck.sum())} stuck seconds, {int((d['bites'][-1] == 0).sum())} ducks never ate and "
          f"{int((d['sips'][-1] == 0).sum())} never drank, of {per * G}")
    print(f"({time.perf_counter() - t0:.0f} s wall)")
    for i in np.flatnonzero(np.maximum(cols[4], cols[5]) > PINNED_SHARE):
        print(f"  over the limit: {labels[i % per]} in garden {i // per}: starving {cols[4][i]:.2f}, parched "
              f"{cols[5][i]:.2f} of the time, {d['bites'][-1, i]:.0f} bites, {d['sips'][-1, i]:.0f} sips")

    held = np.array([label not in HUNGRY_BY_NATURE for label in labels])
    for i in np.flatnonzero(stuck > 0):
        print(f"  stuck: {describe_stuck(d, labels, i)}")

    pinned = np.maximum(np.where(held, mean(cols[4]), 0), mean(cols[5]))  # per label, over the gardens
    return verdict({
        "every duck eats": bool((d["bites"][-1] > 0).all()),
        "every duck drinks": bool((d["sips"][-1] > 0).all()),
        "every duck sleeps": bool((d["asleep"].sum(axis=0) > 0).all()),
        f"no duck is stuck in need for {STUCK_S} s": bool((stuck == 0).all()),
        f"no label starves or parches more than {PINNED_SHARE:.0%} of the time": bool((pinned <= PINNED_SHARE).all()),
    })


if __name__ == "__main__":
    sys.exit(main())
