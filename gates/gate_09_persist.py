"""Gate 9: a garden put away and picked up again is the same garden, older.

Run: uv run python -m gates.gate_09_persist [--minutes 10]

Two things have to be true. Saving and loading must lose nothing, and the catch-up that ages a duck
over the gap must land where actually living through the gap would have landed it. The second is the
real check: `brain/save.py` integrates the drives in 5 s steps with no body and no garden, and
this compares that against the same stretch lived properly in the stub.

The garden is empty on purpose. Catch-up cannot know that a duck found a dish while nobody was
watching, so the honest comparison is a stretch where there was nothing to find. Ducks run blind here
too, since none of this is about vision and it halves the wall time.

Measured 2026-09-18 over the full ten minutes, which is one whole day at DAY_S: hunger and thirst come
back off by 0.000 and sleep pressure within tolerance. Two things had to be fixed to get there, and the
suite runner found both the first time it ran this gate at its default length. Catch-up sat in permanent
darkness, because an all-zero frame has no daylight in it, and a duck alone in the dark gets three times
as sleepy as it should; it now follows the sun across the gap. And its steps were 60 s, which is far too
coarse for a drive that switches sharply between asleep and awake: sleep pressure drifted 0.37 against a
1 s reference at 60 s steps, 0.08 at 10 s and 0.04 at 5 s, where hunger and thirst were exact at every
size. Three days at 5 s steps costs 3.5 s of the 10 s budget. Fatigue, boredom and temperature do not match and should not. A duck away does not walk, so it
does not tire (0.27 lived against 0.00); nothing happens to it, so it gets thoroughly bored (0.23
against 1.00); and it is not standing in the sun, so it sits at a neutral 24 degrees rather than the
garden's 29. Those three are printed rather than asserted.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from brain.data import load_connectome, named_sets
from brain.physiology import Physiology
from brain.save import MAX_GAP_S, catch_up, load, save
from gates.episodes import run, verdict

# What only watches the clock, and so must come back exactly where living through it would have left it.
CLOCK_DRIVES = ("hunger", "thirst", "sleep_pressure")
# What needs a garden, and so cannot: a duck away does not walk, so it does not tire; nothing happens
# to it, so it gets thoroughly bored; and there is no sun on it, so it sits at a neutral temperature.
# These are printed and not asserted, which is the same thing brain/save.py says it does.
WORLD_DRIVES = ("fatigue", "boredom", "body_temp")
DRIVES = CLOCK_DRIVES + WORLD_DRIVES
TOLERANCE = 0.05  # per drive, on a 0-1 scale
THREE_DAY_BUDGET_S = 10.0


def lived(W, ann, sets, minutes, n):
    """Drives after really living through the stretch, in an empty garden."""
    poses = [[[1.0 + 0.4 * i, 2.0, 0.0] for i in range(n)]]
    kept = {}

    def remember(stubs):
        return False

    import brain.server as server
    original = server.BrainServer.step

    def step(self, lockstep):
        out = original(self, lockstep)
        kept["body"] = self.body
        return out

    server.BrainServer.step = step
    try:
        run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], minutes * 60, seed=0,
            until=remember, eyes=False, hunger=0.2, thirst=0.2, body_temp=24.0)
    finally:
        server.BrainServer.step = original
    return {d: np.asarray(getattr(kept["body"], d), float).copy() for d in DRIVES}


def caught_up(minutes, n):
    """Drives after the same stretch spent away, aged by brain/save.py instead."""
    body = Physiology(n, {}, hunger=0.2, thirst=0.2, body_temp=24.0)
    catch_up(body, minutes * 60, since=0.0)  # the same stretch of day the lived duck saw
    return {d: np.asarray(getattr(body, d), float).copy() for d in DRIVES}


def round_trip(n):
    """Save a duck, load it into a different one, and see whether anything was lost."""
    body = Physiology(n, {}, hunger=0.31, thirst=0.62, body_temp=26.5)
    body.fatigue[:] = 0.44
    body.boredom[:] = 0.17
    path = Path(tempfile.mkdtemp()) / "garden.npz"
    save(path, body, when=1000.0)
    other = Physiology(n, {}, hunger=0.99, thirst=0.99, body_temp=10.0)
    load(path, other, now=1000.0)
    worst = max(float(np.max(np.abs(np.asarray(getattr(body, d), float)
                                    - np.asarray(getattr(other, d), float)))) for d in DRIVES)
    return worst, path


def garden_round_trip(n) -> bool:
    """The garden comes back too: where the ducks stood, who wore a hat, what was on the ground, what
    was playing, and the time of day, moved on by the gap."""
    from body.stub2d.stub import DEMO_GARDEN, Stub
    from gates.episodes import free_port_base
    port = free_port_base(7760, n)
    a = Stub(n, 0, tempfile.mkdtemp(), **DEMO_GARDEN, frame_port=port)
    a.hats[n - 1], a.t, a.world.music = True, 123.0, (1.0, 2.0)
    a.world.eat(0)
    path = Path(tempfile.mkdtemp()) / "garden.npz"
    save(path, Physiology(n, {}), stub=a, when=1000.0)
    a.close()
    b = Stub(n, 9, tempfile.mkdtemp(), frame_port=port)  # a different garden, different ducks
    load(path, Physiology(n, {}), stub=b, now=1060.0)
    b.close()
    return bool(np.allclose(a.pose, b.pose) and (a.hats == b.hats).all() and b.t == 183.0
                and b.world.music == (1.0, 2.0) and np.allclose(a.world.food, b.world.food)
                and (a.world.bites == b.world.bites).all())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--ducks", type=int, default=5)
    args = ap.parse_args()
    t0 = time.perf_counter()
    W, ann = load_connectome()
    sets = named_sets(ann)

    worst, path = round_trip(args.ducks)
    print(f"save and load, nothing else: worst drive off by {worst:.4f}")

    real = lived(W, ann, sets, args.minutes, args.ducks)
    away = caught_up(args.minutes, args.ducks)
    print(f"after {args.minutes:g} minutes, lived against spent away:")
    gaps = {}
    for d in DRIVES:
        gaps[d] = float(np.max(np.abs(real[d] - away[d])))
        tag = "" if d in CLOCK_DRIVES else "   (needs the garden; not asserted)"
        print(f"  {d:15s} {np.mean(real[d]):6.3f} against {np.mean(away[d]):6.3f}"
              f"   worst duck off by {gaps[d]:.3f}{tag}")

    body = Physiology(args.ducks, {}, hunger=0.2, thirst=0.2)
    t1 = time.perf_counter()
    catch_up(body, MAX_GAP_S)
    three_days = time.perf_counter() - t1
    print(f"three days away catches up in {three_days:.2f} s")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    return verdict({
        "saving and loading loses nothing": worst < 1e-6,
        "the garden comes back as it was left, a minute later in its day": garden_round_trip(args.ducks),
        "hunger, thirst and sleep age exactly as they would have": all(gaps[d] <= TOLERANCE
                                                                      for d in CLOCK_DRIVES),
        f"three days away loads in under {THREE_DAY_BUDGET_S:g} s": three_days < THREE_DAY_BUDGET_S,
    })


if __name__ == "__main__":
    sys.exit(main())
