"""Saving a garden and catching it up on launch.

Saved: physiology, learned weights, personality, and where everything sits. The connectome is not
saved; it is the same every time and reloads from `data/`.

Catching up is coarse on purpose: the drives are integrated in CATCH_UP_S steps with no body and no
world, which is exact for anything that only rises or decays on a clock. Nothing that needs the garden
(eating, learning) happens during the gap; learned weights fade on the clocks in `brain/plasticity.py`.
The gap is capped at MAX_GAP_S.
"""
import time

import numpy as np

from body import frames
from world.fields import daylight

AMBIENT_C = 24.0  # the garden while nobody is watching: no sun, no shade, no pond
CATCH_UP_S = 5.0  # sleep pressure switches sharply: 60 s steps drift 0.37, 5 s drift 0.04
MAX_GAP_S = 3 * 24 * 3600.0  # three days; past this the drives are saturated


def save(path, body, plastic=None, stub=None, when=None) -> None:
    """Write physiology, personality, learned weights and the garden (a body/stub2d Stub) to a .npz."""
    state = {f"body.{k}": np.asarray(v) for k, v in vars(body).items() if isinstance(v, (np.ndarray, float, int))}
    state |= {f"knob.{k}": np.asarray(v) for k, v in body.k.items()}
    state["when"] = np.asarray(time.time() if when is None else when)
    # the garden's own clock, so the sun is in the right place on return
    state["garden_t"] = np.asarray(float(stub.t) if stub is not None else 0.0)
    if plastic is not None:
        state["weights"] = plastic.w.detach().cpu().numpy()
        state["slow_weights"] = plastic.slow.detach().cpu().numpy()
    if stub is not None:
        world, nowhere = stub.world, [np.nan, np.nan]
        state["pose"], state["hats"] = stub.pose, stub.hats
        state["duck_names"] = np.asarray(stub.duck_names)
        state["hat_style"] = stub.hat_style
        state["hat_items"] = np.asarray(stub.hat_items, float).reshape(-1, 3)
        state["food"] = np.asarray(world.food, float).reshape(-1, 2)
        state["bites"] = np.asarray(world.bites).reshape(-1)
        state["kinds"] = np.asarray(world.kinds).reshape(-1)
        state["hand"] = np.asarray(world.hand if world.hand is not None else nowhere, float)
        state["music"] = np.asarray(world.music if world.music is not None else nowhere, float)
    np.savez(path, **state)


def load(path, body, plastic=None, stub=None, now=None) -> float:
    """Put a saved garden back and run the drives forward over the gap. Returns the gap in seconds."""
    z = np.load(path, allow_pickle=False)
    for key in z.files:
        head, _, name = key.partition(".")
        name = {"run_skill": "walk_skill"}.get(name, name)  # saves from before walking was the skill
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
        if "duck_names" in z.files:  # saves from before the ducks had names keep the new ones
            stub.duck_names = [str(name) for name in z["duck_names"]]
        if "hat_style" in z.files:  # older saves lack these
            stub.hat_style[:] = z["hat_style"]
            stub.hat_items = [[float(x), float(y), int(k)] for x, y, k in z["hat_items"]]
        stub.t = since + gap
        world.set_food(z["food"], z["bites"], z["kinds"] if "kinds" in z.files else None)  # saves from before fruit had kinds have none
        world.hand = None if np.isnan(z["hand"]).any() else tuple(z["hand"])
        world.music = None if np.isnan(z["music"]).any() else tuple(z["music"])
        world.odor[:] = 0
        world.diffuse(2000)  # rebuild the smell of the food that is there now
    catch_up(body, gap, plastic, since=since)
    return gap


def catch_up(body, gap_s: float, plastic=None, since: float = 0.0) -> None:
    """Age the drives over a gap with no garden to react to: hungrier, thirstier, and rested.

    Each step carries its own daylight, or a duck left alone gets far too sleepy in the dark.
    """
    n = len(body.hunger)
    quiet = np.array([frames.blank()] * n, frames.FRAME)
    quiet["temp_left"] = quiet["temp_right"] = AMBIENT_C  # an all-zero frame would be freezing
    still, nothing = np.zeros(n, bool), np.zeros(n)
    for step in range(int(gap_s // CATCH_UP_S)):
        quiet["light"] = daylight(since + step * CATCH_UP_S)
        body.step(CATCH_UP_S, quiet, still, nothing)
    if plastic is not None and gap_s:
        plastic.rest(gap_s)  # the fast weights fade; the consolidated ones mostly stay
