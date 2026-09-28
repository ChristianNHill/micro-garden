"""Cases from the Python modules for web/check.js to hold the browser garden to: the same inputs through the Python
and the JavaScript, and the answers compared. Only what does not roll dice is compared exactly; the browser's dice
are its own. web/pack.py writes these into web/data/parity.json.gz.
"""
from types import SimpleNamespace

import numpy as np

from body.stub2d import retina
from body.stub2d.stub import DEMO_GARDEN
from brain import emotes, reactions, social
from brain.decoder import Decoder
from brain.personality import LABELS, KNOBS, label_of, preset, stack
from brain.physiology import Physiology, aggression_tone, pressing
from brain.save import catch_up
from brain.server import BrainServer, bilateral, sense_levels
from viewer.snapshot import readout
from world import fields
from world.fields import World, contacts, daylight, temperature_at, wind_on

FIVE = ["Bully", "Napper", "Carefree", "Chatty", "Scaredy"]
PLAIN = lambda x: x.tolist() if isinstance(x, np.ndarray) else x


def frame_fields(n: int, rng) -> dict:
    """A made-up frame, one value a duck for every field the body and the server read."""
    f = {name: rng.uniform(0, 1, n) for name in (
        "odor_left", "odor_right", "danger_left", "danger_right", "humidity_left", "humidity_right", "duck_left", "duck_right",
        "music_left", "music_right", "ball_left", "ball_right", "drum_left", "drum_right", "pond_left", "pond_right",
        "shade_left", "shade_right", "near_left", "near_right", "cry_left", "cry_right", "hand_left", "hand_right")}
    f |= {name: rng.uniform(15, 33, n) for name in ("temp_left", "temp_right")}
    f |= {name: (rng.random(n) < 0.3).astype(float) for name in (
        "touch_left", "touch_right", "sugar", "water", "bumped", "petted", "scared", "hat", "ball_near", "kicked", "drum_near",
        "drummed", "swimming", "ate", "drank", "show", "hand_fed", "held", "thrown", "heard_alarm", "heard_joy")}
    f["light"] = np.full(n, rng.uniform(0, 1))
    f["wind"], f["wind_from"] = rng.uniform(0, 1, n), rng.uniform(-np.pi, np.pi, n)
    for name in ("near_id", "bumped_by", "saw_shove_by", "saw_shove_of", "show_by", "hat_taken_by", "comforted_by"):
        f[name] = np.where(rng.random(n) < 0.3, rng.integers(0, n, n), -1).astype(float)
    f["saw_fall_by"] = np.full(n, -1.0)  # a fall seen is answered with dice
    f["ate_kind"] = np.where(rng.random(n) < 0.3, rng.integers(0, 10, n), -1).astype(float)
    return f


def body_state(b) -> dict:
    keep = ("hunger", "thirst", "fatigue", "sleep_pressure", "boredom", "body_temp", "asleep", "alone_s", "joy", "fear",
            "sorrow", "anger", "scent", "scent_was", "swim_skill", "walk_skill", "eat_skill", "dance_skill", "fight_skill",
            "fashion_skill", "music_skill")
    out = {k: PLAIN(np.asarray(getattr(b, k), float)) for k in keep}
    if hasattr(b, "bond"):
        out |= {"bond": b.bond.tolist(), "hand_trust": b.hand_trust.tolist()}
    return out


def five(**kw) -> Physiology:
    b = Physiology(5, stack([preset(x) for x in FIVE]), **kw)
    social.init(b)
    return b


def world_case() -> dict:
    """The demo garden in a steady wind, stepped a while: the smells and what the senses read of them."""
    g = {k: v for k, v in DEMO_GARDEN.items() if k in ("size", "tree", "food_xy", "bites", "danger_xy", "pond", "rocks")}
    w = World(wind=(0.4, -0.8), **g)
    for t in range(60):
        w.step(t * fields.DT)
    rng = np.random.default_rng(3)
    at = rng.uniform(0, g["size"], (40, 2))
    return {"garden": {k: PLAIN(np.asarray(v, float)) if not isinstance(v, (int, float)) else v for k, v in g.items()},
            "wind": [0.4, -0.8], "steps": 60, "at": at.tolist(),
            "odor": [float(w.odor_at(p)) for p in at], "danger": [float(w.odor_at(p, w.danger_odor)) for p in at],
            "humidity": [float(w.humidity_at(p)) for p in at], "temp": [float(temperature_at(p, 0.7, w.tree)) for p in at],
            "grid_sum": [float(w.odor.sum()), float(w.danger_odor.sum()), float(w.damp.sum())],
            "daylight": [daylight(t) for t in np.linspace(0, 1300, 37)],
            "wind_on": [[float(x[0]) for x in wind_on(np.array([h]), np.array([0.3, -0.9]))]
                        for h in np.linspace(-3, 3, 7)],
            "contacts": [list(map(PLAIN, contacts(at[:6], rng.uniform(-3, 3, 6), at[6:9], 1.2)))]}


def retina_case() -> dict:
    w = World(food_xy=[(2.0, 2.2), (3.1, 1.0)], pond=(4.0, 4.0, 1.0), rocks=[(5.0, 1.0, 0.4)], size=6.0, tree=(1.5, 4.3, 0.9))
    w.balls = np.array([[2.6, 2.0, 0, 0]])
    w.hand = (2.2, 1.6)
    xy = np.array([[2.0, 1.8], [2.3, 2.1], [4.0, 2.4]])
    h = np.array([1.2, -2.0, 1.6])
    lum = retina.luminance(xy, h, w, 0.8)
    return {"food": w.food.tolist(), "pond": list(w.pond), "rocks": w.rocks.tolist(), "balls": w.balls.tolist(), "hand": list(w.hand),
            "tree": list(w.tree), "size": 6.0, "xy": xy.tolist(), "heading": h.tolist(), "light": 0.8, "lum": lum.ravel().tolist()}


def body_case() -> dict:
    """Twenty steps of five ducks' bodies and social lives on made-up frames, and every readout of them."""
    rng = np.random.default_rng(5)
    b = five(hunger=0.6, thirst=0.3)
    b.body_temp[:] = [22, 29, 31, 25, 19]
    steps = []
    for _ in range(20):
        f = frame_fields(5, rng)
        escaped, speed = rng.random(5) < 0.2, rng.uniform(0, 0.3, 5)
        falls, wakes = b.step(0.5, f, escaped, speed)
        social.update(b, f, 0.5, speed, np.random.default_rng(0))  # no fall is seen, so the dice decide nothing
        scent = (rng.uniform(0, 0.3, (5, 5)), rng.uniform(0, 0.3, (5, 5)))
        steer = social.steering(b, f, scent)
        wants, damp = rng.uniform(0, 1, 5), rng.uniform(0, 1, 5)
        server = SimpleNamespace(body=b, clap_left=np.zeros(5), plastic=None)
        levels = BrainServer._levels(server, f)
        steps.append({
            "frame": {k: PLAIN(v) for k, v in f.items()}, "escaped": escaped.tolist(), "speed": speed.tolist(),
            "scent": [s.tolist() for s in scent], "wants": wants.tolist(), "damp": damp.tolist(),
            "falls": falls.tolist(), "wakes": wakes.tolist(), "state": body_state(b),
            "gains": {k: PLAIN(v) for k, v in b.sense_gains().items()},
            "motor": {k: PLAIN(np.asarray(v, float)) for k, v in b.motor(wants=wants, damp=damp).items()},
            "cooling": [c.tolist() for c in b.cooling()], "play": b.play().tolist(),
            "steering": {k: PLAIN(np.asarray(v, float)) for k, v in steer.items()},
            "levels": {k: PLAIN(np.asarray(v, float)) for k, v in levels.items()},
            "following": server.following.tolist(),
            "feelings": [emotes.feelings(b, i) for i in range(5)],
            "readout": [readout(b, i) for i in range(5)],
            "aggression": [float(aggression_tone(b.k["aggressiveness"][i], b.hunger[i], 0.4, b.anger[i], b.k["kindness"][i])) for i in range(5)],
        })
    return {"labels": FIVE, "start": {"hunger": 0.6, "thirst": 0.3, "body_temp": [22, 29, 31, 25, 19]}, "steps": steps}


def decoder_case() -> dict:
    """The decoder's intents from set rates and body inputs, where no dice decide anything."""
    rng = np.random.default_rng(9)
    names = ["DNa02", "odor_steer", "moist_steer", "DNp09", "moonwalker", "giant_fiber", "proboscis_mn",
             "danger_valence", "touch_steer", "aIPg", "grooming_dn", "wind_steer"]
    sets = {name: np.array([2 * i, 2 * i + 1]) for i, name in enumerate(names)}
    import pandas as pd
    ann = pd.DataFrame({"side": ["left", "right"] * len(names)})
    import torch
    out = []
    for _ in range(30):
        dec = Decoder(ann, sets, 1)
        dec.wander = np.array([rng.uniform(-0.5, 0.5)])
        dec.ticks = 1
        rates = rng.uniform(0, 3, 14).astype(np.float32)
        rates[[6, 7]] = rng.uniform(0, 0.25, 2)  # no stink met and no aggression, which roll dice
        rates[10] = rng.uniform(0, 0.04)
        dec.rates[0] = rates
        body = {"restlessness": rng.uniform(0, 1), "speed": rng.uniform(0.3, 1.5), "wander": rng.uniform(0.5, 2),
                "surge": rng.uniform(0, 1), "fear": rng.uniform(0, 1), "duck_left": rng.uniform(0, 1), "duck_right": rng.uniform(0, 1),
                "music_affinity": rng.uniform(0, 1), "music_left": rng.uniform(0, 1), "music_right": rng.uniform(0, 1),
                "sociability": rng.uniform(0, 1), "fondness": rng.uniform(-0.5, 0.5), "at_ease": rng.uniform(0, 1),
                "near_left": rng.uniform(0, 0.5), "near_right": rng.uniform(0, 0.5), "bond_near": rng.uniform(-1, 1),
                "bond_turn": rng.uniform(-1, 1), "hand_trust": rng.uniform(-1, 1), "hand_left": rng.uniform(0, 1),
                "hand_right": rng.uniform(0, 1), "comfort": rng.uniform(0, 1), "cry_left": rng.uniform(0, 1),
                "cry_right": rng.uniform(0, 1), "play": rng.uniform(0, 1), "ball_left": rng.uniform(0, 1),
                "ball_right": rng.uniform(0, 1), "cool_left": rng.uniform(0, 1), "cool_right": rng.uniform(0, 1),
                "intent_turn": rng.uniform(-1, 1), "intent": float(rng.random() < 0.3), "hunger": rng.uniform(0, 1),
                "walk_skill": rng.uniform(0, 1), "tasting": bool(rng.random() < 0.5), "zoomies": bool(rng.random() < 0.2),
                "social": rng.uniform(0, 1), "sleepy_together": rng.uniform(0, 0.5)}
        dec.body = body
        from brain.lif import DEVICE
        spikes = torch.zeros((1, 24), dtype=torch.bool, device=DEVICE)
        it = dec.update(spikes)[0]
        out.append({"rates": rates.tolist(), "wander": float(dec.wander[0]), "body": body,
                    "rates_after": dec.rates[0].tolist(), "intent": {k: float(v) for k, v in it.items()}})
    return {"sizes": Decoder(ann, sets, 1).size.tolist(), "cases": out}


def social_case() -> dict:
    b = five(hunger=0.2, thirst=0.2)
    rng = np.random.default_rng(2)
    b.bond = rng.uniform(-1, 1, (5, 5))
    np.fill_diagonal(b.bond, 0)
    w = [{kind: reactions.weights(kind, b, wd, a, t) for kind in ("shove", "fall", "laugh", "comfort", "angry", "sad")}
         for wd, a, t in [(0, 1, 2), (2, 0, 2), (3, 4, 1), (1, 3, -1)]]
    prac = [[s, d, st, idle, float(social.practised(s, d, st, idle))] for s, d, st, idle in
            [(0.0, True, 0.01, 0.0), (0.5, True, 0.03, 0.0), (0.9, False, 0.0, 1e5), (0.2, False, 0.0, 5000.0), (0.6, False, 0.0, 300.0)]]
    return {"bond": b.bond.tolist(), "weights": w, "practised": prac,
            "friends": [list(social.friends(b, i, FIVE)) for i in range(5)],
            "label_of": [label_of(stack([preset(x) for x in LABELS]), i) for i in range(len(LABELS))],
            "labels": list(LABELS)}


def save_case() -> dict:
    b = five(hunger=0.3, thirst=0.4)
    catch_up(b, 7 * 3600.0, since=250.0)
    return {"gap": 7 * 3600.0, "since": 250.0, "state": body_state(b)}


def cases() -> dict:
    return {"world": world_case(), "retina": retina_case(), "body": body_case(), "decoder": decoder_case(),
            "social": social_case(), "save": save_case(), "knobs": KNOBS}
