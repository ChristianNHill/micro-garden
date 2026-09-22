"""MuJoCo microduck body: the garden of the 2D stub around real simulated robots (Gate 10).

The brain sees the same sockets and frames as with the stub. Behind them a real `robotd` runs the robot's
50 Hz loop and walking policy on a body in MuJoCo (pollen-robotics/microduck, `scripts/duck-sim`,
which runs natively on a Mac).

Two things differ from the stub: a duck's pose comes from the simulator's ground truth (the body server on
TCP 7801 + n), as an overhead tracker would give it in a real room, and the brain's requests go on to that
duck's robotd. The garden itself (smells, wind, pond, day and night, food) stays in Python.

Start the simulator first, then this, then the brain:

    (in the microduck checkout)  DUCK_SIM_VIEWER=0 scripts/duck-sim
    uv run python -m body.mujoco.adapter --view --brain
"""
import argparse
import json
import math
import os
import socket
import time

import numpy as np

from body.contract import ROBOT_PARAMS, Client
from world.fields import SIZE_M
from body.mujoco.camera import FRAME_PORT, SimCamera
from body.stub2d.stub import DEMO_GARDEN, DT, EMOTE_ACTS, Stub, take_the_wheel

SIM_STATE = os.path.expanduser("~/.cache/duck-sim")  # where duck-sim puts each duck's robotd socket
TRUTH_PORT = 7801  # the first duck's body server; +1 per duck
SPACING_M = 0.5  # duck-body sets its ducks down in a row along y, this far apart

# ponytail: the upstream walking policy does not track velocity (pollen-robotics/microduck_rl issue 46):
# below about 0.3 the duck stands still, and turning ignores the sign. These four commands move it
# repeatably: forward 0.12 m/s, back 0.14 m/s, left and right 0.05 to 0.3 rad/s. The brain's intent is
# snapped to the nearest. Delete `snap` once upstream ships a policy that tracks a velocity.
FORWARD, BACK, LEFT, RIGHT, STAND = (0.30, 0.0), (-0.30, 0.0), (0.0, 2.0), (0.30, -1.5), (0.0, 0.0)
WALK_AT = 0.02  # m/s the brain has to ask for before the duck means to go anywhere
# Walking in bouts, explicit code and not the fly brain: the robot can only march at one speed or pivot, so
# a duck sits by default and banks what its brain asks for as distance. With a walk's worth banked it gets
# up, marches it off, and sits again. Anything urgent gets it up at once and keeps it marching.
MARCH_MS = 0.12  # what FORWARD actually does
BOUT_M, BANK_M = 0.6, 1.0  # a walk is worth getting up for at this much; nothing more than BANK_M is remembered
URGENT_MS = 0.1  # asked for this much or more, a duck does not wait to bank it
# rad/s of smoothed turning intent before it pivots instead. Above the steering readout's built-in ~0.4 rad/s
# left bias; following the wind asks for 1 to 3.
TURN_AT = 0.8
TURN_MEMORY_S = 3.0  # the brain's turning is noisy moment to moment; a pivot takes seconds
SIT_AFTER_S = 3.0  # on its feet with nowhere to go for this long, a duck sits down
TOGGLE_S = 6.0  # sitting down or standing up takes about this long; the duck is told nothing meanwhile


# ponytail: a fence in software, not the fly brain. The simulator's floor has no walls and the smells stop at
# the edge, so a duck this close to the edge and heading out is pivoted toward the middle. The real fix is
# walls in the MuJoCo scene (a biped falls over them) or a wall the brain can sense.
FENCE_M = 0.3
# Two bipeds that bump fall over (falls happened with another duck 0.21 to 0.28 m away). So, explicit code:
# a duck will not march at one this close ahead, it pivots away, and the garden counts this distance as
# touching so contact still happens without falls.
REACH_M = 0.35
GET_UP_EVERY_S = 12.0  # a fallen duck that is not asleep is helped up this often
LIMP_S = 4.0  # how long a duck on its side lies limp before it is asked to stand
RISE_S = 12.0  # standing up takes about ten seconds; the duck is told nothing else meanwhile


ROBOT_SKILLS = {"ground_pick", "sit_toggle", "roulade", "kick_left", "kick_right"}  # what robotd will run


def robot_state(sock_path: str) -> dict:
    """The newest robot.state from a duck's robotd."""
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(3)
    s.connect(sock_path)
    f = s.makefile("rw")
    f.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "robot.subscribe", "params": {}}) + "\n")
    f.flush()
    while True:
        m = json.loads(f.readline())
        if m.get("method") == "robot.state":
            s.close()
            return m["params"]


def snap(vx: float, turning: float, marching: bool = True) -> tuple[float, float]:
    """The brain's intent as one of the commands the simulated duck obeys. Between bouts (`marching`
    false) a duck that means to go forward stands."""
    if vx < -WALK_AT:
        return BACK
    if vx <= WALK_AT or not marching:
        return STAND  # going nowhere: no pivoting on the spot either
    if abs(turning) > TURN_AT:
        return LEFT if turning > 0 else RIGHT
    return FORWARD


class Truth:
    """Where one simulated duck really is, from the simulator."""

    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rw")
        self._ask({"op": "hello", "protocol": 1, "joints": 15})

    def _ask(self, request: dict) -> dict:
        self.f.write(json.dumps(request) + "\n")
        self.f.flush()
        return json.loads(self.f.readline())

    def pose(self) -> tuple[float, float, float]:
        """(x, y, heading) in the simulator's frame. The full reading is kept in `last` for the viewer."""
        r = self.last = self._ask({"op": "read"})
        w, x, y, z = r["imu"]["quat"]
        return r["trunk"][0], r["trunk"][1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def articulation(self) -> dict:
        """Joint angles in robotd's order, trunk height, and tilt: the orientation with the heading taken out."""
        w, x, y, z = self.last["imu"]["quat"]
        half = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) / 2
        c, s = math.cos(half), -math.sin(half)  # inverse heading quaternion (c, 0, 0, s), times the whole
        tilt = (c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w)
        return {"joints": [round(v, 3) for v in self.last["positions"]], "z": round(self.last["trunk"][2], 4),
                "tilt": [round(v, 4) for v in tilt]}

    def close(self) -> None:
        self.f.close()
        self.sock.close()


class MujocoBody(Stub):
    can_carry_ducks = False  # a cursor cannot lift a physics-held robot

    def __init__(self, n: int, seed: int, sock_dir: str, sim_state: str = SIM_STATE, truth_port: int = TRUTH_PORT,
                 origin=None, cameras=(), **garden):
        """cameras: indices of ducks that see through their sim head camera (as given to DUCK_SIM_CAMERAS).
        The rest see the garden drawn as on the stub. Off by default: MuJoCo's world has no dishes, tree or pond."""
        names = [f"duck-{c}" for c in "abcdefghij"[:n]]
        self.robot_paths = [os.path.join(sim_state, f"{name}.sock") for name in names]
        self.robots = [Client(path) for path in self.robot_paths]
        self.kick_foot = np.zeros(n, bool)  # which foot kicked last
        self.truth = [Truth(truth_port + i) for i in range(n)]
        # where MuJoCo's (0, 0) sits in the garden; by default the row of ducks straddles the middle
        size = garden.get("size", SIZE_M)  # the world is not made until Stub's __init__
        self.origin = np.asarray(origin if origin is not None else (size / 2, size / 2 - SPACING_M * (n - 1) / 2), float)
        self.turning = np.zeros(n)  # the brain's turning intent, smoothed
        self.banked = np.zeros(n)  # metres the brain has asked for and the duck has not walked yet
        self.marching = np.zeros(n, bool)
        self.idle_for = np.zeros(n)  # seconds on its feet and not marching
        self.eating_until = np.zeros(n)  # garden time until which it counts as eating or drinking
        self.sat = np.zeros(n, bool)  # sitting because it wants nothing (asleep is `relaxed`)
        self.cameras = {i: SimCamera(FRAME_PORT + i) for i in cameras}
        self.got_up_at = np.full(n, -np.inf)
        self.limp = np.zeros(n, bool)  # let go to settle flat before getting up
        self.rising_until = np.zeros(n)  # garden time until which a duck is left alone to stand up
        self.turn = 0  # which duck get_up checks next
        super().__init__(n, seed, sock_dir, **garden)  # serves the brain's sockets, sends the first frames
        self.touch_m = REACH_M
        self._read_poses()

    def _read_poses(self) -> None:
        for i, t in enumerate(self.truth):
            x, y, h = t.pose()
            self.pose[i] = [x + self.origin[0], y + self.origin[1], h]

    def fence(self, i: int):
        """The pivot that turns duck i back in, if it is at the edge and heading out; else None."""
        x, y, h = self.pose[i]
        out = np.array([float(x > self.world.size - FENCE_M) - float(x < FENCE_M), float(y > self.world.size - FENCE_M) - float(y < FENCE_M)])
        for rx, ry, r in self.world.rocks:  # a rock turns a robot back like the fence
            if math.hypot(x - rx, y - ry) < r + FENCE_M:
                out = np.array([rx - x, ry - y])
        ahead = np.array([math.cos(h), math.sin(h)])
        if not out.any() or ahead @ out <= 0:
            return None
        to_mid = np.full(2, self.world.size / 2) - (x, y)
        return LEFT if ahead[0] * to_mid[1] - ahead[1] * to_mid[0] > 0 else RIGHT

    def keep_apart(self, i: int):
        """The pivot that turns duck i away from a duck at arm's length ahead of it; else None."""
        x, y, h = self.pose[i]
        rel = self.pose[:, :2] - (x, y)
        d = np.linalg.norm(rel, axis=1)
        d[i] = np.inf
        j = int(d.argmin())
        ahead = np.array([math.cos(h), math.sin(h)])
        if d[j] > REACH_M or ahead @ rel[j] <= 0:
            return None
        return RIGHT if ahead[0] * rel[j, 1] - ahead[1] * rel[j, 0] > 0 else LEFT  # it is on my left: turn right

    def get_up(self) -> None:
        """Fallen and not asleep: stand up.
        - Tipped over while sitting: still in the sit policy, where enable and init do nothing; sit_toggle
          stands it. (A seated duck leans enough that robotd calls it fallen, so sleepers are left alone.)
        - On its side: the rising policy cannot start there. Going limp lets it settle flat, and from flat
          `robot.enable` stands it in about ten seconds.
        While limp or rising a duck is told nothing (see `_move`): any command sends it back over."""
        # one duck per call: robot_state waits up to 20 ms, a whole tick if all five are asked at once
        self.turn = (self.turn + 1) % len(self.robot_paths)
        for i, path in [(self.turn, self.robot_paths[self.turn])]:
            if self.relaxed[i] or self.sat[i] or self.t - self.got_up_at[i] < (LIMP_S if self.limp[i] else GET_UP_EVERY_S):
                continue  # asleep, sitting on purpose (reads as fallen), or helped a moment ago
            if self.limp[i]:  # it has had its moment on the floor
                # must be a request with an id: robotd silently drops enable, relax, init and stop sent
                # as notifications
                self.robots[i].call("robot.enable", on=True)
                self.limp[i], self.got_up_at[i], self.rising_until[i] = False, self.t, self.t + RISE_S
                continue
            state = robot_state(path)
            if state["policy"] == "held":  # let go and never re-enabled: just enable it
                self.robots[i].call("robot.enable", on=True)
                self.got_up_at[i], self.rising_until[i] = self.t, self.t + RISE_S
            elif state["safety"]["fallen"]:
                if state["policy"] == "sit":
                    self.robots[i].notify("robot.do", skill="sit_toggle")
                    self.rising_until[i] = self.t + RISE_S
                else:
                    self.robots[i].call("robot.relax")
                    self.limp[i] = True
                self.got_up_at[i] = self.t

    # the brain's requests go on to the robot; their effect on the garden stays here

    def _move(self, i: int, p: dict) -> None:
        super()._move(i, p)  # kept for the viewer and for `relaxed`
        vx, _, vyaw = self.cmd[i]
        self.turning[i] += (vyaw - self.turning[i]) * min(DT / TURN_MEMORY_S, 1.0)
        if self.limp[i] or self.t < self.rising_until[i]:
            # not even "stand": any robot.move switches the robot from the rising policy to walking
            return
        if self.relaxed[i]:
            return
        urgent = vx >= URGENT_MS or vx < -WALK_AT
        self.banked[i] = min(self.banked[i] + max(vx, 0.0) * DT, BANK_M) if vx > WALK_AT else 0.0  # asking to stop forgets the walk
        if self.sat[i]:
            if urgent or self.banked[i] >= BOUT_M:  # stand, and nothing else until up
                if self._sitting(i):  # asked, not assumed: a toggle to a duck that never sat would sit it
                    self.robots[i].notify("robot.do", skill="sit_toggle")
                self.sat[i], self.marching[i], self.rising_until[i] = False, True, self.t + TOGGLE_S
            return
        if self.marching[i]:
            self.banked[i] -= MARCH_MS * DT
            self.marching[i] = urgent or self.banked[i] > 0
        else:
            self.marching[i] = urgent or self.banked[i] >= BOUT_M
        eating = self.t < self.eating_until[i]  # eating is not idle
        self.idle_for[i] = 0.0 if self.marching[i] or eating else self.idle_for[i] + DT
        if self.idle_for[i] > SIT_AFTER_S:
            if not self._sitting(i):
                self.robots[i].notify("robot.do", skill="sit_toggle")
            self.sat[i], self.idle_for[i], self.rising_until[i] = True, 0.0, self.t + TOGGLE_S
            return
        vx, vyaw = self.fence(i) or self.keep_apart(i) or snap(vx, self.turning[i], self.marching[i])
        self.robots[i].notify("robot.move", vx=vx, vy=0.0, vyaw=vyaw)

    def _head(self, i: int, p: dict) -> None:
        super()._head(i, p)
        self.robots[i].notify("robot.head", **p)

    def _stop(self, i: int, p: dict) -> None:
        super()._stop(i, p)
        self.robots[i].call("robot.stop")

    def _sitting(self, i: int) -> bool:
        return robot_state(self.robot_paths[i])["policy"] == "sit"

    def _relax(self, i: int, p: dict) -> None:
        """Asleep is sitting down via sit_toggle (trunk 0.116 to 0.059 m), not `robot.relax`, which drops
        the duck flat on its face."""
        super()._relax(i, p)
        if not self.sat[i] and not self._sitting(i):
            self.robots[i].notify("robot.do", skill="sit_toggle")
        self.sat[i] = False  # asleep is its own state

    def _init(self, i: int, p: dict) -> None:
        super()._init(i, p)
        if self._sitting(i):
            self.robots[i].notify("robot.do", skill="sit_toggle")
            self.rising_until[i] = self.t + RISE_S

    def knock_down(self, j: int) -> None:
        """A kick that lands: the duck goes limp and `get_up` stands it in 10 to 15 s. The kicker keeps
        its distance (`keep_apart`), so the blow is the garden's and the fall is the robot's."""
        super().knock_down(j)
        if not self.limp[j] and not self.relaxed[j]:
            self.robots[j].call("robot.relax")
            self.limp[j], self.got_up_at[j], self.sat[j] = True, self.t, False

    def _sound(self, i: int, p: dict) -> None:
        super()._sound(i, p)
        self.robots[i].notify("robot.sound", tag=p["tag"])

    def _do(self, i: int, p: dict) -> None:
        """The garden decides the effect; the robot acts it out with the skills it has (headbutt is a kick,
        zoomies a roulade)."""
        super()._do(i, p)
        if p["skill"] in ("ground_pick", "drink"):
            self.eating_until[i] = self.t + 2.0
        if self.limp[i] or self.t < self.rising_until[i]:
            return
        if p["skill"].startswith("emote_"):
            return  # acted already, by _emote, through Stub._do
        if self.sat[i]:
            return
        skill = p["skill"]
        if skill in ("headbutt", "kick"):  # both are a foot
            self.kick_foot[i] = not self.kick_foot[i]
            skill = "kick_left" if self.kick_foot[i] else "kick_right"
        skill = {"zoomies": "roulade", "drum": "ground_pick"}.get(skill, skill)  # a drum tap is a peck
        if skill in ROBOT_SKILLS:
            self.robots[i].notify("robot.do", skill=skill)

    def articulation(self) -> list[dict]:
        return [t.articulation() for t in self.truth]

    def down_left(self) -> np.ndarray:
        return np.zeros(len(self.names))  # the robot gets itself up; Godot shows its real joints

    def posture(self) -> list[str]:
        return ["down" if limp else "sat" if sat else "up" for limp, sat in zip(self.limp, self.sat)]

    def _emote(self, i: int, feeling: str) -> None:
        """As Stub's, but a robot that is down or rising is left alone, and a playful one on its feet rolls over."""
        if self.limp[i] or self.t < self.rising_until[i]:
            return
        if feeling == "playful" and not (self.sat[i] or self.marching[i] or self.acting[i] or self.relaxed[i]):
            self.emotes.append((self.t, i, feeling))
            self._sound(i, {"tag": EMOTE_ACTS[feeling][0]})  # playful always has a voice
            self.robots[i].notify("robot.do", skill="roulade")
            return
        super()._emote(i, feeling)

    def _act(self) -> None:
        for i, queue in enumerate(self.acting):  # a duck that went down mid-emote drops the rest
            if queue and (self.limp[i] or self.t < self.rising_until[i]):
                queue.clear()
        super()._act()

    def sight(self, xy, h, light) -> np.ndarray:
        lum = super().sight(xy, h, light)
        for i, cam in self.cameras.items():
            seen = cam.retina()
            if seen is not None:
                lum[i] = seen
        return lum

    def step(self) -> None:
        """One garden step. The ducks move themselves; this reads where they got to."""
        self._read_poses()
        every = 2.0 / len(self.robot_paths)  # each duck is looked at every two seconds
        if int(self.t / every) > int((self.t - DT) / every):
            self.get_up()
        self._act()
        self.world.step(self.t)
        self.t += DT
        if self.fruit_every_s and int(self.t / self.fruit_every_s) > int((self.t - DT) / self.fruit_every_s):
            self.world.drop_fruit(self.fruit_rng)
        self.send_frames()

    def close(self) -> None:
        for cam in self.cameras.values():
            cam.close()
        for r, t in zip(self.robots, self.truth):
            r.notify("robot.move", vx=0.0, vy=0.0, vyaw=0.0)
            r.close()
            t.close()
        super().close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ducks", type=int, default=1, help="how many duck-sim is running (DUCK_SIM_DUCKS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sock-dir", default=os.path.expanduser("~/.cache/micro-garden"))
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--godot", action="store_true", help="open the Godot garden (viewer/godot/) on this one; closing it ends the garden")
    ap.add_argument("--no-window", action="store_true", help="with --godot: publish only, and open Godot yourself")
    ap.add_argument("--brain", action="store_true", help="drive the ducks from here with the real brain")
    ap.add_argument("--labels", default="Bully,Napper,Carefree,Chatty,Scaredy")
    ap.add_argument("--cameras", default="", help="ducks that see through their sim camera, by letter: a or a,c")
    args = ap.parse_args()
    os.makedirs(args.sock_dir, exist_ok=True)
    cameras = [ord(c.strip()) - ord("a") for c in args.cameras.split(",") if c.strip()]
    from body import frames
    frame_port = frames.free_port_base(frames.FRAME_PORT, args.ducks)  # skip ports another garden holds
    body = MujocoBody(args.ducks, args.seed, args.sock_dir, cameras=cameras, frame_port=frame_port, **DEMO_GARDEN)
    view = None
    if args.view:
        from viewer.debug2d import Viewer
        view = Viewer(body.world)
    server = None
    if args.brain:
        from brain.data import load_connectome, named_sets
        from brain.personality import preset, stack
        from brain.server import BrainServer
        print("loading the connectome, which takes a moment ...")
        W, ann = load_connectome()
        rng = np.random.default_rng(args.seed)
        labels = (args.labels.split(",") * args.ducks)[:args.ducks]
        bodies = [(os.path.join(args.sock_dir, f"{name}.sock"), frame_port + i) for i, name in enumerate(body.names)]
        server = BrainServer(W, ann, named_sets(ann), bodies, args.seed, personality=stack([preset(x, rng) for x in labels]))
        print(f"driving {', '.join(labels)} in MuJoCo")
    world_out = None
    if args.godot:
        from viewer.snapshot import Snapshot
        world_out = Snapshot(window=not args.no_window)
    next_t = time.monotonic()
    try:
        while view is None or view.alive():
            with body.lock:
                body.step()
            if server is not None:
                server.step(lockstep=False)
            take_the_wheel(body, server, view)
            with body.lock:
                if view:
                    view.draw(body, server)
                if world_out:
                    world_out.step(body, server)
            next_t += DT  # the robots run on the wall clock, so the garden does too
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        body.close()


if __name__ == "__main__":
    assert snap(0.08, 0.0) == FORWARD and snap(0.0, 0.0) == STAND and snap(-0.3, 0.0) == BACK and snap(0.08, 0.0, False) == STAND
    assert snap(0.08, TURN_AT + 0.1) == LEFT and snap(0.08, -TURN_AT - 0.1) == RIGHT and snap(0.08, TURN_AT / 2) == FORWARD and snap(0.0, 2.0) == STAND
    main()
