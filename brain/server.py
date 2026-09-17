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
    lv["sugar_grn"] = np.maximum(f["sugar"], f["water"])
    return lv


class BrainServer:
    def __init__(self, W, ann, sets, bodies, seed: int, aggressiveness=0.0, stink_affinity=0.0,
                 hunger=0.0, provoked=0.0):
        """bodies: list of (robot socket path, frame UDP port), one brain each. Personality knobs and
        starting physiology are scalars or one value per body."""
        side = ann["side"].to_numpy()
        self.sets = dict(sets)
        for name in SIDED:
            for s in ("left", "right"):
                self.sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
        self.n = len(bodies)
        self.brain = LIF(W, self.n)
        self.decoder = Decoder(ann, self.sets, self.n, seed, stink_affinity)
        self.aggressiveness = np.broadcast_to(np.asarray(aggressiveness, float), self.n).copy()
        self.body = Physiology(self.n, hunger, provoked)
        self.rng = np.random.default_rng(seed)
        self.robots = [Client(path) for path, _ in bodies]
        self.rx = [frames.receiver(port) for _, port in bodies]
        self.frames = [None] * self.n

    def step(self, lockstep: bool) -> list[dict]:
        """One body step. lockstep=True sends intents as answered requests so a stepped body sees them."""
        read = frames.newer if lockstep else frames.latest
        self.frames = [read(r, f) for r, f in zip(self.rx, self.frames)]
        f = np.array([x if x is not None else np.zeros((), frames.FRAME) for x in self.frames], frames.FRAME)
        levels = sense_levels(f)
        food_odor = (f["odor_left"] + f["odor_right"]) / 2
        levels["pC1_aggr"] = aggression_tone(self.aggressiveness, self.body.hunger, food_odor, self.body.provoked)
        for _ in range(TICKS_PER_STEP):
            intents = self.decoder.update(self.brain.step(*encode(self.rng, self.sets, levels, self.n)))
        self.body.step(BODY_DT_MS / 1000, ate=f["ate"] > 0, bumped=f["bumped"] > 0)
        for robot, it in zip(self.robots, intents):
            send = robot.call if lockstep else robot.notify
            send("robot.move", vx=it["vx"], vy=it["vy"], vyaw=it["vyaw"])
            if it["feed"]:
                send("robot.do", skill="ground_pick")
            if it["attack"]:
                send("robot.do", skill="headbutt")
        return intents

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
