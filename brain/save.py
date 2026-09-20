"""Saving a garden and catching it up on launch (PLAN.md Gate 9).

What is worth keeping is what a duck cannot be given back: how hungry and tired it is, what it has
learned, who it is, and where everything sits. The connectome is not saved; it is the same 2.7 million
edges every time, and loading it from `data/` takes longer than everything here put together.

Catching up is deliberately coarse. A duck left overnight should be hungry, thirsty and rested when
you come back, and it should not cost eight hours of simulation to work that out: the drives are
integrated in CATCH_UP_S steps with no body and no world, which is exact for anything that only
decays or rises on a clock. Anything that needs the garden, hunger from actually eating or what a duck
learns, does not happen while nobody is watching; what it had already learned fades on the two clocks
in `brain/plasticity.py`. Three days is the cap, past which a duck is as
hungry as it is ever going to get.
"""
import time

import numpy as np

from body import frames
from world.fields import daylight

AMBIENT_C = 24.0  # the garden while nobody is watching: no sun, no shade, no pond
CATCH_UP_S = 5.0  # sleep pressure switches too sharply for coarse steps: 60 s drifts 0.37, 5 s drifts 0.04
MAX_GAP_S = 3 * 24 * 3600.0  # three days; longer is the same duck


def save(path, body, plastic=None, stub=None, when=None) -> None:
    """Write physiology, personality, learned weights and the garden (a body/stub2d Stub) to a .npz."""
    state = {f"body.{k}": np.asarray(v) for k, v in vars(body).items() if isinstance(v, (np.ndarray, float, int))}
    state |= {f"knob.{k}": np.asarray(v) for k, v in body.k.items()}
    state["when"] = np.asarray(time.time() if when is None else when)
    # the garden's own clock, so the sun is where it should be on return (this read a `t` the world
    # never had, and every load restarted the day at dawn)
    state["garden_t"] = np.asarray(float(stub.t) if stub is not None else 0.0)
    if plastic is not None:
        state["weights"] = plastic.w.detach().cpu().numpy()
        state["slow_weights"] = plastic.slow.detach().cpu().numpy()
    if stub is not None:
        world, nowhere = stub.world, [np.nan, np.nan]
        state["pose"], state["hats"] = stub.pose, stub.hats
        state["food"] = np.asarray(world.food, float).reshape(-1, 2)
        state["bites"] = np.asarray(world.bites).reshape(-1)
        state["hand"] = np.asarray(world.hand if world.hand is not None else nowhere, float)
        state["music"] = np.asarray(world.music if world.music is not None else nowhere, float)
    np.savez(path, **state)


def load(path, body, plastic=None, stub=None, now=None) -> float:
    """Put a saved garden back and run the drives forward over the gap. Returns the gap in seconds."""
    z = np.load(path, allow_pickle=False)
    for key in z.files:
        head, _, name = key.partition(".")
        if head == "body" and hasattr(body, name):
            setattr(body, name, z[key].copy() if z[key].ndim else z[key].item())
        elif head == "knob":
            body.k[name] = z[key].copy()
    if plastic is not None and "weights" in z.files:
        plastic.w[:] = plastic.w.new_tensor(z["weights"])
        if "slow_weights" in z.files:
            plastic.slow[:] = plastic.slow.new_tensor(z["slow_weights"])
    gap = min(max((time.time() if now is None else now) - float(z["when"]), 0.0), MAX_GAP_S)
    since = float(z["garden_t"]) if "garden_t" in z.files else 0.0
    if stub is not None and "pose" in z.files:
        world = stub.world
        stub.pose[:], stub.hats[:] = z["pose"], z["hats"]
        stub.t = since + gap
        world.food, world.bites = z["food"].copy(), z["bites"].copy()
        world.hand = None if np.isnan(z["hand"]).any() else tuple(z["hand"])
        world.music = None if np.isnan(z["music"]).any() else tuple(z["music"])
        world.odor[:] = 0
        world.diffuse(2000)  # the smell of the food that is there now, not of what was there at start
    catch_up(body, gap, plastic, since=since)
    return gap


def catch_up(body, gap_s: float, plastic=None, since: float = 0.0) -> None:
    """Age the drives over a gap with no garden to react to: hungrier, thirstier, and rested.

    The sun still rises while nobody is watching, so each step carries its own daylight. Without that
    a duck left alone sat in permanent darkness and got three times as sleepy as it should have, which
    is what Gate 9 caught the first time the suite ran it over a whole day.
    """
    n = len(body.hunger)
    quiet = np.array([frames.blank()] * n, frames.FRAME)
    quiet["temp_left"] = quiet["temp_right"] = AMBIENT_C  # an all-zero frame would be freezing
    still, nothing = np.zeros(n, bool), np.zeros(n)
    for step in range(int(gap_s // CATCH_UP_S)):
        quiet["light"] = daylight(since + step * CATCH_UP_S)
        body.step(CATCH_UP_S, quiet, still, nothing)
    if plastic is not None and gap_s:
        plastic.rest(gap_s)  # the fast part of a lesson fades overnight; the consolidated part mostly stays
