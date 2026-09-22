"""Gate 13: the Godot garden draws what the garden publishes, and the player's click reaches the garden.

Run: uv run python -m gates.gate_13_godot [--seconds 20]     (needs Godot 4; set GODOT if it is not in /Applications)

A garden of five wandering ducks publishes its world snapshot (viewer/snapshot.py) at the body's own rate
while the Godot project runs headless beside it. Godot prints how many snapshots it took in each second and,
told to with --feed-at, clicks food into the garden the way a player would. The gate asserts both ends:
every full second carried at least 30 snapshots, and the garden has a dish it did not have before.

What is under test is the pipe, which is the same whichever body fills it, so this uses the 2D body
with no brain: no simulator, no GPU, and 20 s is enough to measure a rate. Run by number only, since
it needs Godot installed.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from body.stub2d.stub import DEMO_GARDEN, DT, Stub
from gates.episodes import verdict
from viewer.snapshot import SNAPSHOT_PORT, Snapshot

GODOT = os.environ.get("GODOT", "/Applications/Godot.app/Contents/MacOS/Godot")
PROJECT = os.path.join(os.path.dirname(__file__), "..", "viewer", "godot")
MIN_HZ = 30
FEED_AT = (2.5, 1.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    args = ap.parse_args()
    if not os.path.exists(GODOT):
        print(f"no Godot at {GODOT}; install Godot 4 or set GODOT")
        return 1
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        stub, out = Stub(5, 0, d, **DEMO_GARDEN), Snapshot()
        dishes = len(stub.world.food)
        godot = subprocess.Popen([GODOT, "--headless", "--path", PROJECT, "--", f"--port={SNAPSHOT_PORT}", "--mute",
                                  f"--feed-at={FEED_AT[0]},{FEED_AT[1]}",
                                  f"--quit-after={args.seconds}"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        next_t, fed_at = time.monotonic(), None
        while godot.poll() is None:
            with stub.lock:
                if int(stub.t / DT) % 25 == 0:
                    stub.cmd = np.column_stack([rng.uniform(-0.1, 0.3, 5), np.zeros(5), rng.uniform(-1.5, 1.5, 5)])
                stub.step()
                out.step(stub)
                if fed_at is None and any(np.allclose(f, FEED_AT, atol=1e-6) for f in stub.world.food):
                    fed_at = stub.t
            next_t += DT
            time.sleep(max(0.0, next_t - time.monotonic()))
        log = godot.stdout.read()
        out.close()
    hz = [int(x) for x in re.findall(r"snapshots hz=(\d+)", log)][1:]  # the first second includes Godot starting
    errors = [line for line in log.splitlines() if "ERROR" in line]
    print(f"Godot took {min(hz, default=0)} to {max(hz, default=0)} snapshots a second over {len(hz)} s; "
          f"the garden went from {dishes} dishes to a fed one at t = {fed_at if fed_at is None else round(fed_at, 2)} s")
    for line in errors[:5]:
        print("  " + line)
    return verdict({
        f"at least {MIN_HZ} snapshots in every second": len(hz) >= args.seconds - 3 and min(hz) >= MIN_HZ,
        "a click in Godot puts food in the garden": fed_at is not None,
        "Godot reports no script errors": not errors,
    })


if __name__ == "__main__":
    sys.exit(main())
