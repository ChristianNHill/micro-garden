"""Saving a garden and catching it up on launch (PLAN.md Gate 9).

What is worth keeping is what a duck cannot be given back: how hungry and tired it is, what it has
learned, who it is, and where everything sits. The connectome is not saved; it is the same 2.7 million
edges every time, and loading it from `data/` takes longer than everything here put together.

Catching up is deliberately coarse. A duck left overnight should be hungry, thirsty and rested when
you come back, and it should not cost eight hours of simulation to work that out: the drives are
integrated in CATCH_UP_S steps with no body and no world, which is exact for anything that only
decays or rises on a clock. Anything that needs the garden, hunger from actually eating or what a duck
learns, does not happen while nobody is watching. Three days is the cap, past which a duck is as
hungry as it is ever going to get.
"""
import time

import numpy as np

from body import frames
from brain.plasticity import RECOVER_S
from world.fields import daylight

AMBIENT_C = 24.0  # the garden while nobody is watching: no sun, no shade, no pond
CATCH_UP_S = 5.0  # sleep pressure switches too sharply for coarse steps: 60 s drifts 0.37, 5 s drifts 0.04
MAX_GAP_S = 3 * 24 * 3600.0  # three days; longer is the same duck


def save(path, body, plastic=None, world=None, when=None) -> None:
    """Write physiology, personality, learned weights and the garden's objects to a .npz."""
    state = {f"body.{k}": np.asarray(v) for k, v in vars(body).items() if isinstance(v, (np.ndarray, float, int))}
    state |= {f"knob.{k}": np.asarray(v) for k, v in body.k.items()}
    state["when"] = np.asarray(time.time() if when is None else when)
    state["garden_t"] = np.asarray(float(getattr(world, "t", 0.0)) if world is not None else 0.0)
    if plastic is not None:
        state["weights"] = plastic.w.detach().cpu().numpy()
    if world is not None:
        state["food"] = np.asarray(world.food, float).reshape(-1, 2)
        state["bites"] = np.asarray(world.bites).reshape(-1)
        state["hand"] = np.asarray(world.hand if world.hand is not None else [np.nan, np.nan], float)
    np.savez(path, **state)


def load(path, body, plastic=None, world=None, now=None) -> float:
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
    if world is not None and "food" in z.files:
        world.food, world.bites = z["food"].copy(), z["bites"].copy()
        world.hand = None if np.isnan(z["hand"]).any() else tuple(z["hand"])
    gap = min(max((time.time() if now is None else now) - float(z["when"]), 0.0), MAX_GAP_S)
    catch_up(body, gap, plastic, since=float(z.get("garden_t", 0.0)))
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
        # weights creep back toward baseline the whole time, which is the forgetting curve
        plastic.w += (plastic.base - plastic.w) * min(gap_s / RECOVER_S, 1.0)
