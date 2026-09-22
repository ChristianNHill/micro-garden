"""Gate 12: five simulated microducks, five fly brains, the whole garden, one day, left alone.

Run: uv run python -m gates.gate_12_sim_five [--minutes 10]
Needs five simulated ducks up, and nothing else running on the machine (the simulator is wall-clock):
    cd ~/.cache/micro-garden/spike/microduck && PATH="$HOME/.cargo/bin:$PATH" \\
        DUCK_SIM_RL=~/.cache/micro-garden/spike/microduck_rl DUCK_SIM_VIEWER=0 DUCK_SIM_DUCKS=5 scripts/duck-sim

Gate 9b's soak on robots: the demo garden and its five labels, for one garden day on the wall clock.
Asserted: every duck eats, drinks and sleeps, and sleeps sitting down rather than face down with its
motors off; every robot's control loop holds 50 Hz (upstream's measure of a healthy duck); nobody walks
out of the garden, which has no walls in MuJoCo; and no duck that is awake ends the day on the floor.

The ducks march and pivot rather than walk, because the upstream walking policy does not track a
velocity (body/mujoco/adapter.py `snap`). That shapes every number here, so the 2D soak's numbers are
printed for scale and not compared.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

from body.mujoco.adapter import SIM_STATE, MujocoBody, robot_state
from body.stub2d.stub import DEMO_GARDEN, DT
from brain.data import load_connectome, named_sets
from brain.personality import preset, stack
from brain.server import BrainServer
from gates.episodes import free_port_base, verdict

LABELS = "Bully,Napper,Carefree,Chatty,Scaredy".split(",")
OUTSIDE_M = 0.5  # further out of the garden than this and the fence has failed
HEALTHY_HZ = 45.0  # upstream's own gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10.0, help="on the wall clock; 10 is one garden day and night")
    args = ap.parse_args()
    n = len(LABELS)
    if not os.path.exists(os.path.join(SIM_STATE, "duck-e.sock")):
        print(f"five simulated ducks are not up; start them first:\n{__doc__.split('the simulator is wall-clock):')[1].split('Gate 9b')[0]}")
        return 2
    port, d = free_port_base(7880, n), tempfile.mkdtemp(prefix="mg")
    body = MujocoBody(n, 0, d, frame_port=port, **DEMO_GARDEN)
    W, ann = load_connectome()
    rng = np.random.default_rng(0)
    server = BrainServer(W, ann, named_sets(ann), [(f"{d}/{nm}.sock", port + i) for i, nm in enumerate(body.names)], 0,
                         personality=stack([preset(x, rng) for x in LABELS]))
    sips, asleep_s, sat_asleep, out_most, late = np.zeros(n), np.zeros(n), np.zeros(n, bool), np.zeros(n), 0
    moved = np.zeros(n)
    steps, next_t, w0, last_xy = int(args.minutes * 60 / DT), time.monotonic(), time.monotonic(), body.pose[:, :2].copy()
    try:
        for k in range(steps):
            with body.lock:
                body.step()
            server.step(lockstep=False)
            sips += [f is not None and f["drank"] > 0 for f in server.frames]
            asleep = server.body.asleep
            asleep_s += asleep * DT
            xy = body.pose[:, :2]
            moved += np.linalg.norm(xy - last_xy, axis=1)
            last_xy = xy.copy()
            out_most = np.maximum(out_most, np.maximum(-xy, xy - body.world.size).max(1))
            if k % 250 == 0:  # every five seconds: is a sleeping duck sitting?
                for i in np.flatnonzero(asleep):
                    sat_asleep[i] |= robot_state(body.robot_paths[i])["policy"] == "sit"
            next_t += DT
            slack = next_t - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                late, next_t = late + 1, time.monotonic()
        wall = time.monotonic() - w0
        states = [robot_state(p) for p in body.robot_paths]
        bites = np.bincount([who for _, who in body.eaten], minlength=n)
        print(f"{args.minutes:g} minutes on the wall clock ({wall:.0f} s; the garden's loop ran late on {100 * late / steps:.0f}% of ticks)")
        print(f"  {'':9s} {'bites':>5s} {'sips':>5s} {'asleep':>7s} {'sat':>4s} {'path':>7s} {'out by':>7s} {'loop Hz':>8s} {'missed':>7s} {'ends':>10s}")
        for i, label in enumerate(LABELS):
            st = states[i]
            print(f"  {label:9s} {bites[i]:5d} {sips[i]:5.0f} {asleep_s[i] / (steps * DT):7.2f} {'yes' if sat_asleep[i] else 'no':>4s}"
                  f" {moved[i]:6.1f}m {max(out_most[i], 0):6.2f}m {st['loop']['hz']:8.1f} {st['loop']['missed']:7d}"
                  f" {'asleep' if server.body.asleep[i] else 'fallen' if st['safety']['fallen'] else st['policy']:>10s}")
        print(f"sim five: {bites.sum()} bites and {sips.sum():.0f} sips in a day; the 2D soak, for scale and not for comparison, "
              f"has about 65 and 95 a garden day")
        slept = asleep_s > 0
        return verdict({
            "every duck eats": bool((bites > 0).all()),
            "every duck drinks": bool((sips > 0).all()),
            "every duck sleeps": bool(slept.all()),
            "and sleeps sitting down": bool(sat_asleep[slept].all()) and bool(slept.any()),
            f"every robot's control loop holds {HEALTHY_HZ:g} Hz": all(st["loop"]["hz"] >= HEALTHY_HZ for st in states),
            f"nobody gets more than {OUTSIDE_M} m outside the garden": bool((out_most <= OUTSIDE_M).all()),
            # a seated duck leans forward far enough that robotd calls it fallen, so a sleeper does not count
            "no duck that is awake ends the day on the floor": not any(
                st["safety"]["fallen"] and not server.body.asleep[i] for i, st in enumerate(states)),
        })
    finally:
        server.close()
        body.close()


if __name__ == "__main__":
    sys.exit(main())
