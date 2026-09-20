"""Gate 5: drives and personality. Thirst, swimming, sleep, and five labels that behave differently.

Run: uv run python -m gates.gate_05_personality [--minutes 10]
Knobs and labels: brain/personality.py, ARCHITECTURE.md §2.4. Drives and emotions: brain/physiology.py.
Blind test (manual, Chris): watch the 2D view for 5 minutes with labels hidden and guess. Score: not yet run.

2026-09-20 record: the labels check FAILS, 3.37 for the same label across runs against 4.09 for different
labels (1.21, bar 1.5; it was 2.29 on 09-18), in the demo garden as it now is (breeze, day-scale hunger,
fruit every 10 s). The table is the finding, not the ratio: every label moves 0.63 to 0.70 of the time,
sleeps 0.14 to 0.18, walks 0.06 m/s and swims under 0.05. The Napper naps 0.18 against everyone's 0.16,
because night puts every duck to sleep and daylight cancels an ordinary duck's sleepiness, leaving the
Napper about one extra nap in two days. Chatty quacks no more than the Napper, since most quacks are
events (a bite, a sip) that every duck has. What still shows: the Bully headbutts (0.15 a minute against
0.03), the Napper keeps to itself (near others 0.3 against 0.5). The likely reason is structural. Needs
outrank likes now, which is right, nearly every knob acts through a like or a temperament, and the ducks
are in need most of the time, so there is rarely a comfortable duck for a personality to show in. That is
a question about the economy and about which knobs should be visible, and it is Chris's; nothing was
tuned to move this number.

2026-09-20, after making room (a find is a ten-bite meal; a need is ignored below 0.4 and total by 0.8, so
ducks have both needs low 32% of their waking time where it was 11%): 3.53 against 4.28, the same 1.21. What
holds across both runs: the Napper moves least (0.55, 0.54 against about 0.65) and slowest, Chatty quacks
most (3.4 and 4.0 a minute against 2.0 to 2.9), the Bully eats most (1.8, 2.0 bites a minute). Swimming
does not pick out Carefree because in the day's heat every duck's swim urge is at its ceiling. A knob shows
only through a channel that moves a duck: speed, sleep, wind-following, wading, music, hats, attacks and
voice do; steering by the smell of other ducks or of a stink is mostly noise, which is where sociability
and stink affinity live. The blind test is the better instrument and is Chris's to run next.

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

Re-run with the pond drawn into the retina, 2026-09-18:
- **Water lovers now pass.** Swim time 0.41 against 0.64 for water love 0.1 against 0.9, a gap of 0.23
  against the 0.15 wanted, where a blind-to-water duck read 0.55 against 0.68. Both numbers fell: ducks
  stopped blundering into water they could not see and started choosing it, which is what the knob means.
- **Heat tolerance now fails the other way round**, 0.18 for an intolerant duck against 0.29 for a
  tolerant one, where it wants the intolerant one higher. This looks like a real conflict rather than
  noise. `physiology.sense_gains` turns a hot duck's cold sense up (`"cold": 1 + 2 * hot`) so it steers
  toward cold to find shade, and POND sits in the sun on purpose. Now that water is visible and steering
  can act, shade-seeking beats cooling off: the hot duck walks away from the sunny pond. Blind, it
  wandered in regardless. SHADED_POND already exists in this file if the intent is to test the swim urge
  without the steering fighting it.
- The labels ratio fell from 2.01 to 1.52 against the 1.5 bar (same label 2.73, different labels 4.16).
  Still passing, but back to the thin margin this check has always had.

Re-run with graded senses, 2026-09-18: four of five pass.
- **The labels check is comfortable at last**, 4.28 against 1.87, a ratio of 2.29 where it read 1.47
  failing and then 1.52. Graded senses sharpened the signatures: Scaredy moves 0.87 of the time at
  nearly twice everyone's speed and keeps the least company, Napper is the only one that sleeps,
  Chatty quacks 6.5 a minute against Scaredy's 2.7.
- Water lovers pass more clearly too: 0.15 against 0.43 swimming, a gap of 0.28 where 0.15 is wanted.
- **The shade-or-water knob came out inverted** (0.21 for a water lover, 0.34 for a water-shy duck).
  First explanation, that cold-seeking walks a duck into the pond, was wrong: `temperature_at` depends
  only on distance from the tree, so the pond is not cold to sense and cold-seeking does steer to shade.
  The real fault was arithmetic. `swim_urge` was `clip(water_love + hot)`, and heat alone pinned it at
  1.00 for every duck, so one that hates water waded in exactly as readily as one that loves it. Heat
  now multiplies the taste for water instead of standing in for it, `clip(water_love * (1 + hot))`,
  which leaves the shade case untouched (0.10 and 0.90) and separates the sunny one (0.20 against 1.00).
  Re-run: the drives half passes all four. A hot duck with water love 0.9 swims 0.24 against 0.12 for
  one with 0.1, the right way round where it read 0.21 against 0.34, and the shade figures are
  bit-identical at 0.15 and 0.43. The margin is 0.12 against a 0.10 bar, so it is thin.
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
    # Fed as well as watered: a like is tested on a duck with nothing pressing, since likes yield to
    # hunger now and a half-hungry duck swims about half as much whatever it thinks of water (2026-09-20).
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
        # Two ducks equally bad at heat, at the same sunny pond, differing only in water love: one
        # cools off by wading in and the other by sitting in the shade. Which it picks is the knob, not
        # the temperature (Chris, 2026-09-18). Asking instead whether every hot duck swims had it
        # backwards, because shade-seeking beat cooling off once the water became visible.
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

        checks["labels differ more than repeats of the same label"] = different > 1.5 * same
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
