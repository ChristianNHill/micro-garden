"""Record the microduck's own motions, once, for the Godot garden to replay: clips.json.

Run with one simulated robot up (`DUCK_SIM_DUCKS=1 scripts/duck-sim`):
    uv run python viewer/godot/record_clips.py

The microduck's motions (sit, stand, walk, turn, kick, roll, peck, go limp, get up) are its control policies
running in the simulator, not animation files. With the MuJoCo body Godot draws the live joints. The 2D
body has no robot, so this plays each motion on the simulated robot and samples its joints, height and lean
at RATE_HZ. Each frame is [joints in robotd's order, trunk height, lean].
"""
import json
import os
import sys
import time
from pathlib import Path

from body.contract import Client
from body.mujoco.adapter import SIM_STATE, TRUTH_PORT, Truth, robot_state

OUT = Path(__file__).parent / "clips.json"
RATE_HZ = 15  # Godot eases between samples; 20 made the file too big
WALK, LEFT, RIGHT = dict(vx=0.30, vy=0.0, vyaw=0.0), dict(vx=0.0, vy=0.0, vyaw=2.0), dict(vx=0.30, vy=0.0, vyaw=-1.5)


def record(truth: Truth, seconds: float, keep=None) -> list:
    """Samples at RATE_HZ for `seconds`; `keep` is called every sample to keep the robot moving."""
    frames, next_t = [], time.monotonic()
    for _ in range(int(seconds * RATE_HZ)):
        if keep:
            keep()
        truth.pose()
        a = truth.articulation()
        frames.append([[round(v, 2) for v in a["joints"]], round(a["z"], 3), [round(v, 3) for v in a["tilt"]]])
        next_t += 1 / RATE_HZ
        time.sleep(max(0.0, next_t - time.monotonic()))
    return frames


def main() -> int:
    sock = os.path.join(SIM_STATE, "duck-a.sock")
    robot, truth = Client(sock), Truth(TRUTH_PORT)
    move = lambda **v: (lambda: robot.notify("robot.move", **v))
    stop = move(vx=0.0, vy=0.0, vyaw=0.0)
    do = lambda skill: robot.notify("robot.do", skill=skill)
    clips = {}

    def settle(seconds=2.0):
        record(truth, seconds, stop)

    settle(3.0)
    clips["stand"] = {"loop": True, "frames": record(truth, 2.0, stop)}
    record(truth, 2.0, move(**WALK))  # get up to speed first
    clips["walk"] = {"loop": True, "frames": record(truth, 3.0, move(**WALK))}
    settle()
    clips["turn_left"] = {"loop": True, "frames": record(truth, 2.0, move(**LEFT))}
    settle()
    clips["turn_right"] = {"loop": True, "frames": record(truth, 2.0, move(**RIGHT))}
    settle()
    do("ground_pick")
    clips["peck"] = {"loop": False, "frames": record(truth, 3.5)}
    settle()
    do("kick_left")
    clips["kick"] = {"loop": False, "frames": record(truth, 3.0)}
    settle(3.0)
    do("sit_toggle")
    clips["sit_down"] = {"loop": False, "frames": record(truth, 6.0)}
    clips["sitting"] = {"loop": True, "frames": record(truth, 1.0)}
    do("sit_toggle")
    clips["stand_up"] = {"loop": False, "frames": record(truth, 7.0)}
    settle(3.0)
    do("roulade")
    clips["roll"] = {"loop": False, "frames": record(truth, 6.5)}
    settle(1.5)
    settle(4.0)
    print("after the roll:", {k: v for k, v in robot_state(sock).items() if k in ("policy", "fallen", "held")})
    robot.call("robot.relax")
    clips["go_limp"] = {"loop": False, "frames": record(truth, 4.0)}
    robot.call("robot.enable", on=True)
    clips["get_up"] = {"loop": False, "frames": record(truth, 12.0)}
    print("at the end:", {k: v for k, v in robot_state(sock).items() if k in ("policy", "fallen", "held")})

    OUT.write_text(json.dumps({"hz": RATE_HZ, "clips": clips}, separators=(",", ":")))
    for name, clip in clips.items():
        z = [f[1] for f in clip["frames"]]
        print(f"  {name:11s} {len(clip['frames']) / RATE_HZ:4.1f} s   trunk height {min(z):.3f} to {max(z):.3f} m")
    print(f"{OUT.name}: {OUT.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
