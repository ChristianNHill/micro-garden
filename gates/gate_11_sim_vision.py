"""Gate 11: a simulated microduck sees through its own camera, and the player can take its wheel.

Run: uv run python -m gates.gate_11_sim_vision
Needs the simulator up with duck-a's camera on. `scripts/duck-sim` refuses to finish with cameras on and
no GStreamer (that is for the robot's WebRTC stream, which this does not use), but by then the body server
and the daemons are running, so the ducks only need standing, which this gate does itself:
    cd ~/.cache/micro-garden/spike/microduck && PATH="$HOME/.cargo/bin:$PATH" \\
        DUCK_SIM_RL=~/.cache/micro-garden/spike/microduck_rl DUCK_SIM_VIEWER=0 DUCK_SIM_CAMERAS=a scripts/duck-sim

Asserted: camera frames reach the fly's retina at a usable rate, through the columns that look where the
camera looks and no others; the view is the duck's own, so it changes when the duck turns; and with the
player at the wheel the robot does what the player asked while the brain goes on sensing and deciding.

Not asserted: the giant fiber firing to something approaching. Vision runs at a twentieth of its natural
gain so it does not drown the nose, and at that gain the escape detector catches few swooping hands
(brain/decoder.py). The camera sees only MuJoCo's world (floor and other ducks, none of the garden's
dishes, tree or pond), so a simulated duck sees the garden drawn from above unless asked to use its
camera (`body/mujoco/adapter.py --cameras a`). The camera is for the real robot, where it is the only eye.
"""
import json
import os
import socket
import sys
import tempfile
import time

import numpy as np

from body import frames
from body.mujoco.adapter import FORWARD, LEFT, SIM_STATE, MujocoBody, robot_state
from body.mujoco.camera import SEEN
from body.stub2d.stub import DT
from gates.episodes import free_port_base, verdict
from gates.gate_10_sim_one import run_for


def stand(sock_path: str) -> None:
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(5)
    s.connect(sock_path)
    f = s.makefile("rw")
    f.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "robot.enable", "params": {"on": True}}) + "\n")
    f.flush()
    f.readline()
    s.close()


def main() -> int:
    robotd = os.path.join(SIM_STATE, "duck-a.sock")
    if not os.path.exists(robotd):
        print(f"no simulated duck at {robotd}; start one with its camera on first:\n{__doc__.split('this gate does itself:')[1].split('Asserted:')[0]}")
        return 2
    stand(robotd)
    time.sleep(7)  # the sit-to-stand policy takes about six seconds

    port = free_port_base(7870, 1)
    d = tempfile.mkdtemp(prefix="mg")
    body = MujocoBody(1, 0, d, food_xy=[], frame_port=port, cameras=(0,))
    cam = body.cameras[0]
    checks = {}
    try:
        run_for(body, 3.0)
        fps = cam.frames / 3.0
        lum = body.seen[0]
        inside, outside = lum[SEEN], lum[~SEEN]
        print(f"camera: {fps:.0f} frames a second; seen through {SEEN.sum(1).tolist()} of 721 columns an eye; "
              f"those columns vary by {inside.std():.3f}, the rest by {np.ptp(outside):.4f}")
        checks["camera frames reach the retina at 10 a second or more"] = fps >= 10
        checks["the columns that look where the camera looks carry the scene, and only those"] = (
            inside.std() > 0.01 and np.ptp(outside) < 1e-6)

        before = body.seen[0].copy()
        run_for(body, 6.0, each=lambda: body.robots[0].notify("robot.move", vx=LEFT[0], vy=0.0, vyaw=LEFT[1]))
        body.robots[0].notify("robot.move", vx=0.0, vy=0.0, vyaw=0.0)
        changed = float(np.abs(body.seen[0] - before)[SEEN].mean())
        print(f"pivoting for 6 s changed what it sees by {changed:.3f} a column")
        checks["the view is the duck's own: it changes when the duck turns"] = changed > 0.01

        from brain.data import load_connectome, named_sets
        from brain.server import BrainServer
        W, ann = load_connectome()
        server = BrainServer(W, ann, named_sets(ann), [(f"{d}/duck-a.sock", port)], 0, hunger=0.9)
        server.possessed[0] = True
        t_first, intents = [None], []

        def think():
            body._move(0, {"vx": 0.2, "vy": 0.0, "vyaw": 0.0})  # the player holds W
            intents.append(server.step(lockstep=False)[0])
            if t_first[0] is None and server.frames[0] is not None:
                t_first[0] = float(server.frames[0]["t"])
        run_for(body, 4.0, each=think)
        asked = robot_state(robotd)["move"]["requested"]
        sensed = float(server.frames[0]["t"]) - t_first[0]
        brain_wanted = float(np.std([it["vyaw"] for it in intents]))
        server.close()
        print(f"player at the wheel holding W for 4 s: robotd has {asked}; the brain's frames moved on {sensed:.1f} s "
              f"and it went on deciding (its turning varied by {brain_wanted:.2f} rad/s), unsent")
        checks["with the player at the wheel the robot does what the player asked"] = bool(np.allclose([asked[0], asked[2]], FORWARD))
        checks["and the brain goes on sensing and deciding"] = sensed > 3.0 and brain_wanted > 0.05
    finally:
        body.close()
    return verdict(checks)


if __name__ == "__main__":
    sys.exit(main())
