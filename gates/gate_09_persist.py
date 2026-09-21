"""Gate 9: a garden put away and picked up again is the same garden, older.

Run: uv run python -m gates.gate_09_persist [--minutes 10]

Two things have to be true. Saving and loading must lose nothing, and the catch-up that ages a duck
over the gap must land where actually living through the gap would have landed it. The second is the
real check: `brain/save.py` integrates the drives in 5 s steps with no body and no garden, and
this compares that against the same stretch lived properly in the stub.

The garden is empty on purpose. Catch-up cannot know that a duck found a dish while nobody was
watching, so the honest comparison is a stretch where there was nothing to find. Each duck also lives
its stretch alone, in a garden of its own: five in a row used to drift apart, but a contented duck can
stand still now, so they dozed in a huddle and a neighbour's nudge fired a sleeper's escape and woke it
at 537 s, which put its sleep 0.66 out against a catch-up that had no neighbour to reckon with
(2026-09-20). Being woken is something that happens in a garden, like finding a dish.

Alone is not enough either, and why is worth knowing: the giant fiber fires at about 0.6 Hz whatever the
duck senses, asleep and blind with nothing but dry air included, and at that rate three spikes land inside
the decoder's 100 ms escape window by chance about once in six duck-minutes (8 escapes among 5 lone blind
ducks in 10 minutes). Two of those woke a sleeper. So sleep is asserted on the ducks nothing disturbed, no
escape while asleep or in the minute before dropping off, which is the stretch catch-up claims to describe;
LIVED_DUCKS of them live it so that enough are left, and the disturbed ones are printed. The false startles
themselves are left alone here: the escape threshold sits at the giant fiber's refractory limit, so moving
it is a decision about the model rather than a fix to this gate.

Later the same day the detector was reset from measurement (5 spikes in 300 ms, brain/decoder.py) and false
startles fell from about 57 an hour to 3. The undisturbed-duck check stays: a startle in the night is rare
now, not impossible. Ducks run blind here
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
LIVED_DUCKS = 8  # lone ducks in the lived comparison; about a quarter get startled in their sleep
UNDISTURBED_MIN = 3  # and the sleep check needs at least this many that were not
SETTLE_S = 60.0  # an escape this soon before dropping off delays it (fear has to fade first)


def lived(W, ann, sets, minutes, n):
    """Drives after really living through the stretch, in an empty garden."""
    poses = [[[2.0, 2.0, 0.0]]] * n  # one duck to a garden
    kept = {"asleep": np.zeros(n, bool), "disturbed": np.zeros(n, bool), "last_startle": np.full(n, -np.inf)}

    def watch(server, stubs):
        body, startled = server.body, np.asarray(server.escaped, bool)
        dropped_off = body.asleep & ~kept["asleep"]
        kept["disturbed"] |= (startled & kept["asleep"]) | (dropped_off & (server.t - kept["last_startle"] < SETTLE_S))
        kept["last_startle"][startled] = server.t
        kept["asleep"], kept["body"] = body.asleep.copy(), body

    run(W, ann, sets, [dict(food_xy=[], pose=p) for p in poses], minutes * 60, seed=0, watch=watch,
        eyes=False, hunger=0.2, thirst=0.2, body_temp=24.0)
    return {d: np.asarray(getattr(kept["body"], d), float).copy() for d in DRIVES}, kept["disturbed"]


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

    real, disturbed = lived(W, ann, sets, args.minutes, LIVED_DUCKS)
    away = caught_up(args.minutes, LIVED_DUCKS)
    calm = ~disturbed
    print(f"{int(disturbed.sum())} of {LIVED_DUCKS} lone ducks were startled in or just before their sleep")
    print(f"after {args.minutes:g} minutes, lived against spent away:")
    gaps = {}
    for d in DRIVES:
        gaps[d] = float(np.max(np.abs(real[d] - away[d])[calm if d == "sleep_pressure" else slice(None)], initial=0))
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
        "hunger, thirst and undisturbed sleep age exactly as they would have": all(gaps[d] <= TOLERANCE
                                                                                  for d in CLOCK_DRIVES),
        f"at least {UNDISTURBED_MIN} ducks slept undisturbed": int(calm.sum()) >= UNDISTURBED_MIN,
        f"three days away loads in under {THREE_DAY_BUDGET_S:g} s": three_days < THREE_DAY_BUDGET_S,
    })


if __name__ == "__main__":
    sys.exit(main())
