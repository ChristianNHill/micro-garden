"""Gate 10: one simulated microduck, a real robot daemon on a body in MuJoCo, living in our garden.

Run: uv run python -m gates.gate_10_sim_one [--no-brain]
Needs the simulator up first (PLAN.md Gate 10 says how it runs on a Mac):
    cd ~/.cache/micro-garden/spike/microduck && PATH="$HOME/.cargo/bin:$PATH" \\
        DUCK_SIM_RL=~/.cache/micro-garden/spike/microduck_rl DUCK_SIM_VIEWER=0 scripts/duck-sim
The simulator is wall-clock and has to hold 1.0x real time, so run nothing else on the machine with it.

What is asserted is the plumbing, because that is what is ours: the garden's frames reach the brain at the
robot's rate; the pose in them is where MuJoCo says the duck is; what the brain asks for reaches robotd;
the duck walks when told to; and a duck standing on a dish that is told to eat gets its bite.

What is printed and not asserted is the real brain walking the duck up the wind to a dish. The walking
policy upstream ships does not track a velocity (pollen-robotics/microduck_rl issue 46), so
`body/mujoco/adapter.py` snaps the brain's intent to the four commands that do something: a duck that can
only march or pivot, at 0.12 m/s and about 0.1 rad/s. Asserting an arrival on that would be asserting the
workaround. When a policy that walks lands, delete `snap` and move this check up.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

from body import frames
from body.contract import Client
from body.mujoco.adapter import FORWARD, SIM_STATE, STAND, MujocoBody, robot_state
from body.stub2d.stub import DT
from gates.episodes import free_port_base, verdict

WALK_S, WALK_M = 6.0, 0.3  # told to walk for this long, it should get at least this far (it does about 0.7 m)
BRAIN_S, DISH_UPWIND_M = 90.0, 1.0


def run_for(body, seconds: float, each=None) -> None:
    """The garden on the wall clock, as the robot is."""
    next_t = time.monotonic()
    for _ in range(int(seconds / DT)):
        with body.lock:
            body.step()
        if each:
            each()
        next_t += DT
        time.sleep(max(0.0, next_t - time.monotonic()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-brain", action="store_true", help="skip the printed brain-driven walk (saves two minutes)")
    args = ap.parse_args()
    robotd = os.path.join(SIM_STATE, "duck-a.sock")
    if not os.path.exists(robotd):
        print(f"no simulated duck at {robotd}; start one first:\n{__doc__.split('Needs the simulator up first')[1].split('The simulator is')[0]}")
        return 2

    port = free_port_base(7820, 1)
    d = tempfile.mkdtemp(prefix="mg")
    body = MujocoBody(1, 0, d, food_xy=[], wind=(0.0, -1.0), frame_port=port)
    ours, rx = Client(f"{d}/duck-a.sock"), frames.receiver(port)
    checks = {}
    try:
        # frames, at the robot's rate, carrying where MuJoCo says the duck is
        # by the frames' own clock against the wall's: counting arrivals undercounts, since a frame still
        # in flight when the socket is read is merged with the next one
        run_for(body, DT)  # the first frame went out before this socket was listening
        first, w0 = frames.newer(rx), time.monotonic()
        run_for(body, 2.0)
        f = frames.newer(rx, first)
        rate = (f["t"] - first["t"]) / DT / (time.monotonic() - w0)
        x, y, h = body.truth[0].pose()
        off = float(np.hypot(f["x"] - (x + body.origin[0]), f["y"] - (y + body.origin[1])))
        print(f"frames: {rate:.0f} a second; the pose in them is {1000 * off:.1f} mm from MuJoCo's, heading off by {abs(f['heading'] - h):.3f} rad")
        checks["the garden's frames arrive at the robot's 50 Hz"] = rate >= 45
        checks["the pose in a frame is where MuJoCo says the duck is"] = off < 0.01 and abs(f["heading"] - h) < 0.05

        # what the brain asks for reaches robotd, snapped to what this policy obeys
        ours.notify("robot.move", vx=0.1, vy=0.0, vyaw=0.0)
        time.sleep(0.3)
        asked = robot_state(robotd)["move"]["requested"]
        print(f"asked for vx 0.1: robotd has {asked}")
        checks["a move the brain asks for reaches the robot"] = bool(np.allclose(asked[:1] + asked[2:], FORWARD))

        # and the duck walks
        x0, y0, _ = body.truth[0].pose()
        run_for(body, WALK_S, each=lambda: ours.notify("robot.move", vx=0.1, vy=0.0, vyaw=0.0))
        x1, y1, _ = body.truth[0].pose()
        went = float(np.hypot(x1 - x0, y1 - y0))
        ours.notify("robot.move", vx=0.0, vy=0.0, vyaw=0.0)
        time.sleep(0.3)
        print(f"told to walk for {WALK_S:g} s: went {went:.2f} m ({went / WALK_S:.3f} m/s); then told to stand: robotd has {robot_state(robotd)['move']['requested']}")
        checks[f"the duck walks when told to (at least {WALK_M} m)"] = went > WALK_M
        checks["and stands when told to"] = bool(np.allclose(robot_state(robotd)["move"]["requested"], [STAND[0], 0.0, STAND[1]]))

        # a bite: a dish under the duck, and the brain's ground_pick
        with body.lock:
            body.world.food = body.pose[:1, :2].copy()
            body.world.bites = np.array([3])
        ours.call("robot.do", skill="ground_pick")
        run_for(body, 0.1)
        print(f"a dish under the duck and a ground_pick: {len(body.eaten)} bite, {int(body.world.bites.sum())} left in the dish")
        checks["a duck on a dish that is told to eat gets its bite"] = len(body.eaten) == 1 and int(body.world.bites.sum()) == 2

        if not args.no_brain:
            rx.close()  # the brain listens on this port itself
            from brain.data import load_connectome, named_sets
            from brain.server import BrainServer
            W, ann = load_connectome()
            with body.lock:
                body._read_poses()
                px, py, _ = body.pose[0]
                body.world.food = np.array([[px, py + DISH_UPWIND_M]])  # the wind blows toward -y, so this is upwind
                body.world.bites = np.array([10])
                body.world.odor[:] = 0
                body.world.diffuse(2000)
            server = BrainServer(W, ann, named_sets(ann), [(f"{d}/duck-a.sock", port)], 0, hunger=0.9)
            nearest = [DISH_UPWIND_M]

            def think():
                server.step(lockstep=False)
                nearest[0] = min(nearest[0], float(np.hypot(*(body.pose[0, :2] - body.world.food[0]))) if len(body.world.food) else 0.0)
            run_for(body, BRAIN_S, each=think)
            server.close()
            print(f"the real brain, hungry, a dish {DISH_UPWIND_M:g} m upwind, {BRAIN_S:g} s: came within {nearest[0]:.2f} m, "
                  f"{len(body.eaten) - 1} bites   (printed, not asserted: waits on a policy that walks)")
    finally:
        ours.notify("robot.move", vx=0.0, vy=0.0, vyaw=0.0)
        ours.close()
        body.close()
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
