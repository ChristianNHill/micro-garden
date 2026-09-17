"""Brain server: sensory frames in, microduck intents out, one brain per body (PLAN.md Gate 4).

Each 20 ms body step: newest frame per duck -> encode -> 2 brain ticks -> decode -> robot.move, plus
robot.do ground_pick while feeding. Encoding: each directional sense feeds its own side's neurons
(food and danger smell, moist and dry air, heat and cold, touch); sugar or pond water on contact feeds
the sugar/water taste neurons, which FlyWire does not split.

Drive the free-running stub:  uv run python -m body.stub2d.stub --view   then
                              uv run python -m brain.server
"""
import argparse
import os
import string
import time

import numpy as np

from body import frames
from body.contract import Client
from brain.data import load_connectome, named_sets, shuffled
from brain.decoder import Decoder
from brain.encoder import encode
from brain.lif import DT_MS, LIF
from brain.physiology import Physiology, aggression_tone

BODY_DT_MS = 20.0
TICKS_PER_STEP = int(BODY_DT_MS / DT_MS)
ODOR_HALF = 0.5  # odor concentration that gives input level 0.5; the dish itself is about 3
# Antennae 10 cm apart see about 1.17:1; steering DNs need about 3:1 (Gate 4 model work). This gain on
# the normalized left/right difference is a modelling assumption standing in for peripheral sharpening.
# Real and shuffled brains get the same input.
ODOR_CONTRAST = 8.0
HUMID_HALF = 0.1  # humidity 1 m from the pond edge is about 0.08
DRY_LEVEL = 0.2  # dry-air neurons at full dryness; kept low so dry air does not swamp other senses
TEMP_COMFORT_C, TEMP_SPAN_C, TEMP_LEVEL = 25.0, 5.0, 0.5
TOUCH_LEVEL = 0.2  # bristle input per side while another duck touches that side
VOICE_COOLDOWN_S = 3.0
SIDED = ["orn_food", "orn_danger", "moist_air", "dry_air", "heat", "cold", "bristle"]


def bilateral(left, right, half):
    """Per-side input levels: saturating overall level, left/right difference amplified by ODOR_CONTRAST."""
    mean = (left + right) / 2
    d = ODOR_CONTRAST * (left - right) / np.maximum(left + right, 1e-9)
    level = mean / (mean + half)
    return level * np.clip(1 + d, 0, 2), level * np.clip(1 - d, 0, 2)


def sense_levels(f: np.ndarray) -> dict:
    """Frame records (structured array, one per brain) -> encoder levels."""
    lv = {}
    for name, key, half in (("orn_food", "odor", ODOR_HALF), ("orn_danger", "danger", ODOR_HALF),
                            ("moist_air", "humidity", HUMID_HALF)):
        lv[f"{name}_left"], lv[f"{name}_right"] = bilateral(f[f"{key}_left"], f[f"{key}_right"], half)
    for side in ("left", "right"):
        temp = f[f"temp_{side}"]
        lv[f"heat_{side}"] = TEMP_LEVEL * np.clip((temp - TEMP_COMFORT_C) / TEMP_SPAN_C, 0, 1)
        lv[f"cold_{side}"] = TEMP_LEVEL * np.clip((TEMP_COMFORT_C - temp) / TEMP_SPAN_C, 0, 1)
        lv[f"dry_air_{side}"] = DRY_LEVEL * (1 - f[f"humidity_{side}"])
        lv[f"bristle_{side}"] = TOUCH_LEVEL * np.minimum(f[f"touch_{side}"], 1)
    lv["sugar"], lv["water_taste"] = f["sugar"], f["water"]  # combined into sugar_grn after gains
    return lv


class BrainServer:
    def __init__(self, W, ann, sets, bodies, seed: int, personality: dict | None = None,
                 hunger=0.5, thirst=0.5, provoked=0.0, body_temp=24.0, **knobs):
        """bodies: list of (robot socket path, frame UDP port), one brain each. personality and knobs
        (brain/personality.py names) and starting physiology are scalars or one value per body; unset
        knobs are 0.5."""
        side = ann["side"].to_numpy()
        self.sets = dict(sets)
        for name in SIDED:
            for s in ("left", "right"):
                self.sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
        self.n = len(bodies)
        self.k = {**(personality or {}), **knobs}
        self.brain = LIF(W, self.n)
        self.decoder = Decoder(ann, self.sets, self.n, seed, self.k.get("stink_affinity", 0.5))
        self.body = Physiology(self.n, self.k, hunger, thirst, provoked, body_temp)
        self.chattiness = np.broadcast_to(np.asarray(self.k.get("chattiness", 0.5), float), self.n)
        self.rng = np.random.default_rng(seed)
        self.voice_rng = np.random.default_rng(seed + 1)
        self.robots = [Client(path) for path, _ in bodies]
        self.rx = [frames.receiver(port) for _, port in bodies]
        self.frames = [None] * self.n
        self.last_vx = np.zeros(self.n)
        self.escaped = np.zeros(self.n, bool)
        self.quiet_until = np.zeros(self.n)
        self.t = 0.0

    def step(self, lockstep: bool) -> list[dict]:
        """One body step. lockstep=True sends intents as answered requests so a stepped body sees them."""
        read = frames.newer if lockstep else frames.latest
        self.frames = [read(r, f) for r, f in zip(self.rx, self.frames)]
        f = np.array([x if x is not None else np.zeros((), frames.FRAME) for x in self.frames], frames.FRAME)
        body = self.body
        self.t += BODY_DT_MS / 1000
        falls_asleep, wakes = body.step(BODY_DT_MS / 1000, f, escaped=self.escaped, speed=self.last_vx)

        levels = sense_levels(f)
        wet = f["swimming"] > 0  # flies do not swim: wet reads as saturated humidity and touch all over
        for s in ("left", "right"):
            levels[f"moist_air_{s}"] = np.where(wet, 1.0, levels[f"moist_air_{s}"])
            levels[f"bristle_{s}"] = np.where(wet, TOUCH_LEVEL, levels[f"bristle_{s}"])
        gains = body.sense_gains()
        for key in levels:
            base = key.removesuffix("_left").removesuffix("_right")
            if base in gains:
                levels[key] = levels[key] * gains[base]
        levels["sugar_grn"] = np.maximum(levels.pop("sugar"), levels.pop("water_taste"))
        food_odor = (f["odor_left"] + f["odor_right"]) / 2
        levels["pC1_aggr"] = aggression_tone(body.k["aggressiveness"], body.hunger, food_odor, body.anger,
                                             body.k["kindness"])

        self.decoder.body = {**body.motor(), "swimming": wet, "at_shore": f["water"] > 0, "thirst": body.thirst}
        self.escaped[:] = False
        for _ in range(TICKS_PER_STEP):
            intents = self.decoder.update(self.brain.step(*encode(self.rng, self.sets, levels, self.n)))
            self.escaped |= [it["escape"] for it in intents]
        self.last_vx = np.array([it["vx"] for it in intents])

        for i, (robot, it) in enumerate(zip(self.robots, intents)):
            send = robot.call if lockstep else robot.notify
            if falls_asleep[i]:
                send("robot.relax")
            if wakes[i]:
                send("robot.init")
            send("robot.move", vx=it["vx"], vy=it["vy"], vyaw=it["vyaw"])
            if it["feed"]:
                send("robot.do", skill="drink" if f["water"][i] > 0 else "ground_pick")
            if it["attack"]:
                send("robot.do", skill="headbutt")
            tag = self._voice(i, f[i], falls_asleep[i], wakes[i])
            if tag:
                send("robot.sound", tag=tag)
        return intents

    def _voice(self, i, f, falls_asleep, wakes) -> str | None:
        """A quack for this step, if any: events pick the tag, chattiness picks whether to speak."""
        if self.t < self.quiet_until[i]:
            return None
        tag = self._pick_voice(i, f, falls_asleep, wakes)
        if tag:
            self.quiet_until[i] = self.t + VOICE_COOLDOWN_S
        return tag

    def _pick_voice(self, i, f, falls_asleep, wakes) -> str | None:
        b = self.body
        events = [
            (self.escaped[i], "alarm"), (wakes, "greet"), (falls_asleep, "coo"),
            (f["ate"] > 0 or f["drank"] > 0, "chirp"), (f["bumped"] > 0, "alarm"),
        ]
        tag = next((t for happened, t in events if happened), None)
        if tag is None and not b.asleep[i]:
            # idle chatter, about once a minute at chattiness 1, colored by mood
            if self.voice_rng.random() < self.chattiness[i] * BODY_DT_MS / 1000 / 60 * (1 + 3 * b.boredom[i]):
                tag = "peck" if b.sorrow[i] > 0.5 else "inquire" if b.boredom[i] > 0.5 else "chirp"
            return tag
        return tag if self.voice_rng.random() < 0.2 + 0.8 * self.chattiness[i] else None

    def close(self) -> None:
        for c in self.robots:
            c.close()
        for r in self.rx:
            r.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ducks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sock-dir", default=os.path.expanduser("~/.cache/micro-garden"))
    ap.add_argument("--shuffled", action="store_true", help="drive the ducks with the shuffled control brain")
    args = ap.parse_args()
    W, ann = load_connectome()
    if args.shuffled:
        W = shuffled(W, args.seed)
    bodies = [(os.path.join(args.sock_dir, f"duck-{c}.sock"), frames.FRAME_PORT + i)
              for i, c in enumerate(string.ascii_lowercase[:args.ducks])]
    server = BrainServer(W, ann, named_sets(ann), bodies, args.seed)
    print(f"driving {args.ducks} ducks{' with shuffled brains' if args.shuffled else ''}; Ctrl-C to stop")
    next_t = time.monotonic()
    while True:
        server.step(lockstep=False)
        next_t += BODY_DT_MS / 1000
        lag = next_t - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            next_t = time.monotonic()  # brain fell behind real time; don't try to catch up


if __name__ == "__main__":
    main()
