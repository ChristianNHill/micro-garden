"""2D stub body: kinematic ducks on the garden plane behind the microduck contract (PLAN.md Gate 3).

One Unix socket per duck (duck-a.sock ...), like duck-sim, plus control.sock with stub-only
sim.step {n} and sim.state. Every body step sends each duck's sensory frame over UDP.

Run free at real time with the debug window:  uv run python -m body.stub2d.stub --view --wander
Without --wander the ducks stand still until a client sends robot.move.
"""
import argparse
import os
import socket
import string
import threading
import time

import numpy as np

from body import frames
from body.contract import ROBOT_PARAMS, serve
from world.fields import SIZE_M, DUCK_R, World, contacts, temperature_at

DT = 0.02
# ponytail: guessed limits standing in for robotd's clamps; replace with the sim's real ones at Gate 10
MAX_V, MAX_VY, MAX_VYAW = 0.3, 0.15, 2.0
ANTENNA = np.array([0.06, 0.05])  # forward, lateral offset of each odor sample, metres
CONTROL_PARAMS = {"sim.step": {"n": 1}, "sim.state": {}}


class Stub:
    def __init__(self, n: int, seed: int, sock_dir: str, food_xy=((3.0, 3.0), (1.0, 1.0))):
        rng = np.random.default_rng(seed)
        self.world = World(food_xy)
        self.pose = np.column_stack([rng.uniform(0.5, SIZE_M - 0.5, (n, 2)), rng.uniform(-np.pi, np.pi, n)])
        self.cmd = np.zeros((n, 3))
        self.head = np.zeros((n, 4))
        self.relaxed = np.zeros(n, bool)
        self.skill = [""] * n
        self.t = 0.0
        self.lock = threading.Lock()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.names = [f"duck-{c}" for c in string.ascii_lowercase[:n]]
        self.servers = [
            serve(os.path.join(sock_dir, f"{name}.sock"), ROBOT_PARAMS, self._robot_call(i), self.lock)
            for i, name in enumerate(self.names)
        ]
        self.servers.append(serve(os.path.join(sock_dir, "control.sock"), CONTROL_PARAMS, self._control_call, self.lock))

    def _robot_call(self, i: int):
        def call(method, p):
            if method == "robot.move":
                if not self.relaxed[i]:
                    self.cmd[i] = np.clip([float(p["vx"]), float(p["vy"]), float(p["vyaw"])],
                                          [-MAX_V, -MAX_VY, -MAX_VYAW], [MAX_V, MAX_VY, MAX_VYAW])
            elif method == "robot.head":
                self.head[i] = [float(p[k]) for k in ROBOT_PARAMS["robot.head"]]
            elif method == "robot.do":
                self.skill[i] = str(p["skill"])  # ponytail: 2D stub records the skill and does nothing
            elif method == "robot.stop":
                self.cmd[i] = 0
            elif method == "robot.relax":
                self.relaxed[i], self.cmd[i] = True, 0
            elif method == "robot.init":
                self.relaxed[i] = False
            return {}
        return call

    def _control_call(self, method, p):
        if method == "sim.step":
            for _ in range(int(p["n"])):
                self.step()
            return {"t": self.t}
        return self.state()

    def step(self) -> None:
        x, y, h = self.pose.T
        vx, vy, vyaw = self.cmd.T
        h += vyaw * DT
        x += (vx * np.cos(h) - vy * np.sin(h)) * DT
        y += (vx * np.sin(h) + vy * np.cos(h)) * DT
        np.clip(self.pose[:, :2], DUCK_R, SIZE_M - DUCK_R, out=self.pose[:, :2])
        self.pose[:, 2] = (h + np.pi) % (2 * np.pi) - np.pi
        self.world.step()
        self.t += DT
        self.send_frames()

    def send_frames(self) -> None:
        xy, h = self.pose[:, :2], self.pose[:, 2]
        fwd = np.column_stack([np.cos(h), np.sin(h)])
        left = np.column_stack([-np.sin(h), np.cos(h)])
        base = xy + ANTENNA[0] * fwd
        odor_l = self.world.odor_at(base + ANTENNA[1] * left)
        odor_r = self.world.odor_at(base - ANTENNA[1] * left)
        touch, dish = contacts(xy, self.world.food)
        temp = temperature_at(xy)
        for i in range(len(xy)):
            self.udp.sendto(frames.pack(
                t=self.t, duck=i, x=xy[i, 0], y=xy[i, 1], heading=h[i],
                odor_left=odor_l[i], odor_right=odor_r[i],
                sugar=float(dish[i] >= 0), touch=touch[i], temperature=temp[i],
            ), (frames.HOST, frames.FRAME_PORT + i))

    def state(self) -> dict:
        return {"t": self.t, "pose": self.pose.tolist(), "food": self.world.food.tolist()}

    def close(self) -> None:
        for s in self.servers:
            s.shutdown()
            s.server_close()
        self.udp.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ducks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sock-dir", default=os.path.expanduser("~/.cache/micro-garden"))
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--wander", action="store_true", help="random walk every 0.5 s, for watching the stub alone")
    args = ap.parse_args()
    os.makedirs(args.sock_dir, exist_ok=True)
    stub = Stub(args.ducks, args.seed, args.sock_dir)
    print(f"sockets in {args.sock_dir}: {', '.join(stub.names)}, control")
    view = None
    if args.view:
        from viewer.debug2d import Viewer
        view = Viewer()
    rng = np.random.default_rng(args.seed)
    next_t, ticks = time.monotonic(), 0
    while view is None or view.alive():
        with stub.lock:
            if args.wander and ticks % 25 == 0:
                stub.cmd = np.column_stack([rng.uniform(-0.1, 0.3, args.ducks), np.zeros(args.ducks),
                                            rng.uniform(-1.5, 1.5, args.ducks)])
            ticks += 1
            stub.step()
            if view:
                view.draw(stub)
        next_t += DT
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    main()
