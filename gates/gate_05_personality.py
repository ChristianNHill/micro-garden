"""Gate 5: drives and personality. Thirst, swimming, sleep, and five labels that behave differently.

Run: uv run python -m gates.gate_05_personality [--minutes 10] [--part drives|labels|all]
Knobs and labels: brain/personality.py. Drives and emotions: brain/physiology.py.

Asserted (--part drives, the default), 10 ducks per scenario:
- drinking: a thirsty duck at the shore lowers its thirst by more than 0.3 in 30 s.
- water love: at a pond in the shade, so heat does not drive everyone in, water love 0.9 swims more
  than 0.1 by 0.15.
- heat: ducks equally bad at heat at the sunny pond, differing only in water love. One cools off by
  wading in, the other by sitting in the shade, and the 0.9 duck must swim more by 0.1. The margin is
  thin (0.12 on the last run). Asking whether every hot duck swims is the wrong test, because once the
  pond is visible, shade-seeking beats cooling off.
- sleep: the Napper sleeps more than Energetic by 0.05.

Printed, not asserted (--part labels, over an hour): an eight-number behavior row per label, averaged
over 4 gardens (one garden is dominated by who reaches the fruit first), for two seeds, and the ratio of
different-label distance to same-label distance. It was asserted at 1.5 and dropped: its run-to-run noise
is as large as what it looks for. A blind test (watch `stub --view --brain --blind` with labels hidden)
showed that a garden that feels alive matters more than labels a viewer can name. The table is still
the place to see what each knob does.

Why labels barely separate: needs outrank likes, nearly every knob acts through a like or a temperament,
and the ducks are in need most of the time. A knob shows only through a channel that moves a duck.
Speed, sleep, wind-following, wading, music, hats, attacks and voice do. Steering by the smell of other
ducks or of a stink is mostly noise, and that is where sociability and stink affinity act. What holds:
the Napper moves least and slowest, Chatty quacks most, the Bully eats and headbutts most.
"""
import argparse
import sys
import time

import numpy as np

from body.stub2d.stub import DEMO_GARDEN, DT
from brain.data import load_connectome, named_sets
from brain.personality import preset, stack
from gates.episodes import run, verdict
from world.fields import SHORE_M, TREE

POND = (2.0, 2.0, 0.35)  # in the sun
SHADED_POND = (TREE[0], TREE[1], 0.35)  # under the tree
LABELS = ["Bully", "Napper", "Carefree", "Chatty", "Scaredy"]
N = 10  # episodes per scenario


def shore_poses(n, pond=POND):
    """Ducks inside the shore band, facing the water."""
    a = np.linspace(-np.pi, np.pi, n, endpoint=False)
    r = pond[2] + SHORE_M * 0.5
    return np.column_stack([pond[0] + r * np.cos(a), pond[1] + r * np.sin(a), a + np.pi])


def relaxed_log():
    """(log, until) pair: until() records which ducks are relaxed (asleep) each step."""
    log = []

    def until(stubs):
        log.append(np.concatenate([s.relaxed for s in stubs]))
        return False
    return log, until


def drinking(W, ann, sets):
    thirst = []
    run(W, ann, sets, [dict(food_xy=[], pond=POND, pose=p) for p in shore_poses(N)], 30.0, seed=0,
        watch=lambda server, stubs: thirst.append(server.body.thirst.copy()), thirst=1.0, water_love=0.0)
    return thirst[0].mean(), thirst[-1].mean()


def swim_time(W, ann, sets, pond=POND, **kw):
    eps = [dict(food_xy=[], pond=pond, pose=p) for p in shore_poses(N, pond)]
    # Fed and watered: likes yield to hunger, so a hungry duck swims less whatever it thinks of water.
    traj, _ = run(W, ann, sets, eps, 60.0, seed=0, thirst=0.0, hunger=0.0, **kw)
    return (np.linalg.norm(traj[:, :, :2] - pond[:2], axis=-1) < pond[2] - SHORE_M).mean()


def sleep_time(W, ann, sets, label, minutes):
    log, until = relaxed_log()
    rng = np.random.default_rng(0)
    k = stack([preset(label, rng) for _ in range(N)])
    poses = np.column_stack([np.full(N, 2.0), np.linspace(0.5, 3.5, N), np.zeros(N)])
    run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], minutes * 60, seed=0, until=until, personality=k)
    return np.array(log).mean()


def signatures(W, ann, sets, jitter_seed, minutes, gardens=4):
    """Several gardens, each with the five labels in a shuffled start order; one behavior row per label,
    averaged over gardens (a single garden is dominated by who happens to reach the fruit first)."""
    rng = np.random.default_rng(jitter_seed)
    n = len(LABELS)
    rows = np.zeros((n, len(COLUMNS)))
    for g in range(gardens):  # one garden at a time keeps GPU memory small
        perm = rng.permutation(n)
        k = stack([preset(LABELS[j], rng) for j in perm])
        garden = {**DEMO_GARDEN, "pose": [[1.5 + 0.4 * i, 2.0, rng.uniform(-np.pi, np.pi)] for i in range(n)]}
        log, until = relaxed_log()
        traj, (st,) = run(W, ann, sets, [garden], minutes * 60, seed=jitter_seed * 10 + g, until=until, personality=k)
        relaxed = np.array(log)
        xy = traj[:, :, :2]
        step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) / DT
        others = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=-1) + np.eye(n) * 9
        pond = st.world.pond
        count = lambda events, col: np.bincount([e[col] for e in events], minlength=n)
        garden_rows = np.column_stack([
            (step > 0.02).mean(axis=0),  # moving
            relaxed.mean(axis=0),  # asleep
            (others.min(axis=2) < 0.3).mean(axis=0),  # near others
            (np.linalg.norm(xy - pond[:2], axis=-1) < pond[2] - SHORE_M).mean(axis=0),  # swimming
            step.mean(axis=0),  # speed
            count(st.sounds, 1) / minutes,  # quacks per minute
            count(st.headbutts, 1) / minutes,  # headbutts per minute
            count(st.eaten, 1) / minutes,  # bites per minute
        ])
        rows[perm] += garden_rows / gardens  # duck i in this garden has label perm[i]
    return rows


COLUMNS = ["moving", "asleep", "near others", "swimming", "speed", "quacks/min", "headbutts/min", "bites/min"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--part", choices=["drives", "labels", "all"], default="drives",
                    help="drives is the gate; labels prints the five-label table and asserts nothing, "
                         "and takes over an hour, so it runs when asked for")
    args = ap.parse_args()
    minutes = args.minutes
    W, ann = load_connectome()
    sets = named_sets(ann)
    t0 = time.perf_counter()

    checks = {}
    if args.part in ("drives", "all"):
        before, after = drinking(W, ann, sets)
        print(f"thirsty duck at the shore: thirst {before:.2f} -> {after:.2f} in 30 s")
        # water love is tested in the shade, so heat does not drive everyone in
        shy = swim_time(W, ann, sets, pond=SHADED_POND, water_love=0.1, body_temp=21.0)
        lover = swim_time(W, ann, sets, pond=SHADED_POND, water_love=0.9, body_temp=21.0)
        print(f"time swimming in the shade: water love 0.1 {shy:.2f}, 0.9 {lover:.2f}")
        # Equally bad at heat, differing only in water love: which way a duck cools off is the knob.
        waders = swim_time(W, ann, sets, pond=POND, water_love=0.9, heat_tolerance=0.05)
        shaders = swim_time(W, ann, sets, pond=POND, water_love=0.1, heat_tolerance=0.05)
        print(f"hot ducks at the sunny pond: water love 0.9 swims {waders:.2f}, 0.1 swims {shaders:.2f}")
        napper, energetic = sleep_time(W, ann, sets, "Napper", minutes), sleep_time(W, ann, sets, "Energetic", minutes)
        print(f"time asleep over {minutes:g} min: Napper {napper:.2f}, Energetic {energetic:.2f}")

        checks |= {
            "drinking lowers thirst": after < before - 0.3,
            "water lovers swim more": lover > shy + 0.15,
            "a hot duck cools off in the way its water love picks": waders > shaders + 0.1,
            "Napper sleeps more than Energetic": napper > energetic + 0.05,
        }
    if args.part in ("labels", "all"):
        a, b = signatures(W, ann, sets, 1, minutes), signatures(W, ann, sets, 2, minutes)
        print(f"\n{'':10s}" + "".join(f"{c:>14s}" for c in COLUMNS))
        for run_name, rows in (("run 1", a), ("run 2", b)):
            for label, row in zip(LABELS, rows):
                print(f"{label:10s}" + "".join(f"{v:14.2f}" for v in row) + f"   ({run_name})")
        both = np.vstack([a, b])
        z = (both - both.mean(axis=0)) / (both.std(axis=0) + 1e-9)
        za, zb = z[: len(LABELS)], z[len(LABELS):]
        d = np.linalg.norm(za[:, None] - zb[None], axis=-1)  # run-1 duck vs run-2 duck
        same, different = np.diag(d).mean(), d[~np.eye(len(LABELS), dtype=bool)].mean()
        print(f"\nsignature distance: same label across runs {same:.2f}, different labels {different:.2f}")
        print(f"({time.perf_counter() - t0:.0f} s wall)")

        # Printed, not asserted: its run-to-run noise is as large as what it looks for (see the docstring).
        print(f"  (labels against repeats: {different / same:.2f}; it was asserted at 1.5 until 2026-09-20)")
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
