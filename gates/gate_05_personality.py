"""Gate 5: drives and personality. Thirst, swimming, sleep, and five labels that behave differently.

Run: uv run python -m gates.gate_05_personality [--minutes 10]
Knobs and labels: brain/personality.py, ARCHITECTURE.md §2.4. Drives and emotions: brain/physiology.py.
Blind test (manual, Chris): watch the 2D view for 5 minutes with labels hidden and guess. Score: not yet run.

2026-09-17 record:
- drives: thirst 1.00 -> 0.01 in 30 s at the shore; swimming in the shade, water love 0.1 vs 0.9: 0.19 vs 0.37;
  at the sunny pond, heat tolerance 0.05 vs 0.95: 0.36 vs 0.24; asleep over 10 min, Napper 0.14 vs Energetic 0.00.
- labels (4 gardens per run): same label across runs 1.90, different labels 4.19. Consistent traits:
  Bully fights most, Napper sleeps and moves least, Chatty quacks most, Scaredy fastest and swims least.
- Getting there:
  - one garden per run was luck-dominated (ratio 1.06), so labels are averaged over 4 gardens
  - energy's speed range was widened
  - water love now pulls toward the pond
  - Bully seeks company and gets hungry fast
  - a duck at the shore reconsiders wading every 5 s
  - a swimmer paddles nearly in place while one that chose not to walks out
- Timidity, sociability and curiosity have little to react to until vision (Gate 6), other ducks' smell
  (Gate 8) and novel objects (Gate 8b); re-judge Scaredy, Loner and Curious then.
Decision (Chris, 2026-09-17): keep all 21 labels and the Bully change; re-judge the quiet labels after
Gates 6 and 8.

2026-09-17, later: the labels check now FAILS. Re-run after the clean-code pass: same label across runs
2.78, different labels 4.10, ratio 1.47 against the 1.5 bar. Drives still pass with the recorded numbers.
Not the refactor: the same 10-minute garden is bit-identical before and after it (trajectories, sounds,
headbutts), and the behavior code has not changed since 0dc72a7, on the same torch and numpy. The
recorded 1.90/4.19 does not reproduce. The rows are still luck-dominated at 4 gardens (Carefree 0.88
bites/min in run 1, 0.00 in run 2; Chatty 0.00 then 0.38 headbutts/min), so the measure was marginal
when it passed. Decision (Chris, 2026-09-17): leave it failing and re-judge after Gates 6 and 8, when
timidity, sociability and curiosity have something to react to. Untried fix: average over 8 gardens,
which is what took the ratio from 1.06 to 1.90 when gardens went 1 -> 4.

Re-run with eyes open, 2026-09-17 (Gate 6 wired vision into the closed loop at VIS_GAIN 0.05):
- **The labels check now passes.** Same label across runs 2.10, different labels 4.23, a ratio of 2.01
  against the 1.5 bar, where blind it failed at 1.47. This is what Chris held the gate open for:
  timidity, sociability and curiosity finally have something to react to. Still 4 gardens, so this is
  not more averaging, it is more signal.
- **Both swimming checks now fail, on margin and not direction.** Water love 0.1 against 0.9 swims
  0.55 against 0.68, needing +0.15; heat tolerance 0.05 against 0.95 swims 0.35 against 0.30, needing
  +0.10. Blind, the water-love pair read 0.19 against 0.37. So eyes roughly triple how much every duck
  swims and the measure saturates, squeezing the knobs together rather than reversing them.
- Why: the pond is not drawn into the retina at all (body/stub2d/retina.py renders dishes, ducks and
  the tree, on the grounds that water is flat on the ground). A duck cannot see water, so it wanders
  in rather than choosing it, and a knob about wanting water cannot show through. Drawing the pond is
  the obvious next thing to try, and is a design call: a real pond is a bright reflective surface.
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
    import brain.server as server
    orig = server.BrainServer.step

    def step(self, lockstep):
        out = orig(self, lockstep)
        thirst.append(self.body.thirst.copy())
        return out
    server.BrainServer.step = step
    try:
        run(W, ann, sets, [dict(food_xy=[], pond=POND, pose=p) for p in shore_poses(N)], 30.0, seed=0,
            thirst=1.0, water_love=0.0)
    finally:
        server.BrainServer.step = orig
    return thirst[0].mean(), thirst[-1].mean()


def swim_time(W, ann, sets, pond=POND, **kw):
    eps = [dict(food_xy=[], pond=pond, pose=p) for p in shore_poses(N, pond)]
    traj, _ = run(W, ann, sets, eps, 60.0, seed=0, thirst=0.0, **kw)
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
    ap.add_argument("--part", choices=["drives", "labels", "all"], default="all",
                    help="run half the gate (each half takes about 20 minutes)")
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
        # same sunny pond: heat-intolerant ducks overheat sooner and go back in to cool off (swimming cools a
        # duck, so a hot duck's swims are short; comparing two ponds also mixed in their positions)
        intolerant = swim_time(W, ann, sets, pond=POND, water_love=0.3, heat_tolerance=0.05)
        tolerant = swim_time(W, ann, sets, pond=POND, water_love=0.3, heat_tolerance=0.95)
        print(f"time swimming at the sunny pond: heat tolerance 0.05 {intolerant:.2f}, 0.95 {tolerant:.2f}")
        napper, energetic = sleep_time(W, ann, sets, "Napper", minutes), sleep_time(W, ann, sets, "Energetic", minutes)
        print(f"time asleep over {minutes:g} min: Napper {napper:.2f}, Energetic {energetic:.2f}")

        checks |= {
            "drinking lowers thirst": after < before - 0.3,
            "water lovers swim more": lover > shy + 0.15,
            "heat-intolerant ducks swim more in the sun": intolerant > tolerant + 0.1,
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

        checks["labels differ more than repeats of the same label"] = different > 1.5 * same
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
