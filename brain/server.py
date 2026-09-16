"""Brain server: sensory frames in, microduck intents out, one brain per body (PLAN.md Gate 4).

Each 20 ms body step: newest frame per duck -> encode (odor per antenna into same-side food ORNs,
sugar on contact) -> 2 brain ticks -> decode -> robot.move, plus robot.do ground_pick while feeding.

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

BODY_DT_MS = 20.0
TICKS_PER_STEP = int(BODY_DT_MS / DT_MS)
ODOR_HALF = 0.5  # odor concentration that gives input level 0.5; the dish itself is about 3
# Antennae 10 cm apart see about 1.17:1; steering DNs need about 3:1 (Gate 4 model work). This gain on
# the normalized left/right difference is a modelling assumption standing in for peripheral sharpening.
# Real and shuffled brains get the same input.
ODOR_CONTRAST = 8.0


def odor_levels(left, right):
    """Per-antenna input levels: saturating overall level, left/right difference amplified."""
    mean = (left + right) / 2
    d = ODOR_CONTRAST * (left - right) / np.maximum(left + right, 1e-9)
    level = mean / (mean + ODOR_HALF)
    return level * np.clip(1 + d, 0, 2), level * np.clip(1 - d, 0, 2)


class BrainServer:
    def __init__(self, W, ann, sets, bodies, seed: int):
        """bodies: list of (robot socket path, frame UDP port), one brain each."""
        side = ann["side"].to_numpy()
        orn = sets["orn_food"]
        self.sets = {**sets, "orn_food_left": orn[side[orn] == "left"], "orn_food_right": orn[side[orn] == "right"]}
        self.n = len(bodies)
        self.brain = LIF(W, self.n)
        self.decoder = Decoder(ann, self.sets, self.n, seed)
        self.rng = np.random.default_rng(seed)
        self.robots = [Client(path) for path, _ in bodies]
        self.rx = [frames.receiver(port) for _, port in bodies]
        self.frames = [None] * self.n

    def step(self, lockstep: bool) -> list[dict]:
        """One body step. lockstep=True sends intents as answered requests so a stepped body sees them."""
        read = frames.newer if lockstep else frames.latest
        self.frames = [read(r, f) for r, f in zip(self.rx, self.frames)]
        f = np.array([(x["odor_left"], x["odor_right"], x["sugar"]) if x is not None else (0, 0, 0)
                      for x in self.frames])
        left, right = odor_levels(f[:, 0], f[:, 1])
        levels = {"orn_food_left": left, "orn_food_right": right, "sugar_grn": f[:, 2]}
        for _ in range(TICKS_PER_STEP):
            intents = self.decoder.update(self.brain.step(*encode(self.rng, self.sets, levels, self.n)))
        for robot, it in zip(self.robots, intents):
            send = robot.call if lockstep else robot.notify
            send("robot.move", vx=it["vx"], vy=it["vy"], vyaw=it["vyaw"])
            if it["feed"]:
                send("robot.do", skill="ground_pick")
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
