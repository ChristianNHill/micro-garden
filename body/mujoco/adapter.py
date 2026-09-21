"""MuJoCo microduck body: the garden of the 2D stub around real simulated robots (PLAN.md Gate 10).

The brain does not know the difference. It sends the same `robot.*` calls to the same socket names and
reads the same sensory frames. Behind them, where the stub integrated a circle on a plane, a real
`robotd` runs the robot's own 50 Hz loop and walking policy on a body in MuJoCo
(pollen-robotics/microduck, `scripts/duck-sim`; see PLAN.md Gate 10 for how it runs on a Mac).

So this is the stub with two things swapped. A duck's pose comes from the simulator's ground truth
(the body server on TCP 7801 + n), the way an overhead tracker will give it in a real room. And what the
brain asks of the body is passed on to that duck's robotd. Everything the garden owns stays the garden's,
and stays in Python (ARCHITECTURE.md decision 1): smells and the wind they ride, the pond, day and night,
what is on the ground and who ate it.

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
from body.mujoco.camera import FRAME_PORT, SimCamera
from body.stub2d.stub import DEMO_GARDEN, DT, Stub, take_the_wheel

SIM_STATE = os.path.expanduser("~/.cache/duck-sim")  # where duck-sim puts each duck's robotd socket
TRUTH_PORT = 7801  # the first duck's body server; +1 per duck
SPACING_M = 0.5  # duck-body sets its ducks down in a row along y, this far apart

# ponytail: the walking policy upstream ships does not walk (pollen-robotics/microduck_rl issue 46, open
# since 2026-09-10): below a command of about 0.3 the duck stands still, and turning ignores the command's
# sign. Measured 2026-09-20, these four commands are the ones that do something, repeatably in direction:
# forward 0.12 m/s, back 0.14 m/s, left and right 0.05 to 0.3 rad/s. So what the brain asks for is snapped
# to the nearest of them. It is a duck that can only march or pivot, and slowly. Delete `snap` and send
# the brain's own numbers the day upstream ships a policy that tracks a velocity.
FORWARD, BACK, LEFT, RIGHT, STAND = (0.30, 0.0), (-0.30, 0.0), (0.0, 2.0), (0.30, -1.5), (0.0, 0.0)
WALK_AT = 0.02  # m/s the brain has to ask for before the duck means to go anywhere
# The robot marches at one speed, and steps on the spot for every pivot, so a duck told to march or pivot
# whenever its brain wanted any movement at all never stopped shuffling its feet, and Chris, watching, asked
# for ducks that sit when they are not going anywhere (2026-09-21). So a duck walks in bouts. It sits by
# default and banks what its brain asks for as distance; when it has banked a walk's worth it gets up and
# marches it off without a break, then sits down again. Anything urgent, food on the wind, fear, a fight,
# gets it up at once and keeps it marching. A duck ambling at 0.04 m/s sits for most of a minute between walks.
MARCH_MS = 0.12  # what FORWARD actually does
BOUT_M, BANK_M = 0.6, 1.0  # a walk is worth getting up for at this much; nothing more than BANK_M is remembered
URGENT_MS = 0.1  # asked for this much or more, a duck does not wait to bank it
# rad/s of smoothed turning intent before it pivots instead. Above the steering readout's built-in turn,
# which is about 0.4 rad/s to the left whatever the duck senses (PLAN.md Gate 9b screen): at 0.5 that alone
# had ducks pivoting a third of the time. Following the wind asks for 1 to 3.
TURN_AT = 0.8
TURN_MEMORY_S = 3.0  # the brain's turning is mostly noise from moment to moment; a pivot takes seconds
SIT_AFTER_S = 3.0  # on its feet with nowhere to go for this long, a duck sits down
TOGGLE_S = 6.0  # sitting down or standing up takes about this long, and the duck is told nothing meanwhile


# ponytail: a fence in software. The simulator's floor has no walls and the garden's smells stop at its
# edge, so two ducks of five walked out of the garden in three minutes and had nothing to lead them back
# (the 2D stub clamped positions and never said so). A duck this close to the edge and heading out is
# pivoted toward the middle instead. The real fixes are walls in the MuJoCo scene, which a biped falls over,
# or a wall the brain can sense; a room will have its own.
FENCE_M = 0.3
# Two bipeds that bump fall over: every fall in a first day of five ducks had another duck 0.21 to 0.28 m
# away and nothing else going on, and two that fell against each other could not get up. So a simulated
# duck keeps the others at arm's length: it will not march at one that is this close and ahead of it, it
# pivots away instead; and the garden counts that distance as touching, so that brushing past, keeping
# company and the Bully's kicks all still happen, without anybody being knocked down.
REACH_M = 0.35
GET_UP_EVERY_S = 12.0  # a duck that has fallen and is not asleep is helped up, this often until it is
LIMP_S = 4.0  # how long a duck on its side lies limp before it is asked to stand
RISE_S = 12.0  # standing up takes about ten seconds, and for that long the duck is told nothing else


ROBOT_SKILLS = {"ground_pick", "sit_toggle", "roulade", "kick_left", "kick_right"}  # what robotd will run
# How a robot acts out each feeling (brain/emotes.py): a voice tag, and its head through a few poses, each
# (seconds after the last, neck_pitch, head_pitch, head_yaw, head_roll) in radians, pitch positive down. The
# head works sitting as well as standing, so a duck at rest still shows what it feels. Every one ends level.
# ponytail: drawn by eye on the simulated duck; a real one's neck will want these re-posed.
LEVEL = (0.0, 0.0, 0.0, 0.0)
EMOTE_ACTS = {
    "happy": ("wheee", [(0.0, -0.1, -0.2, 0.0, 0.3), (0.3, -0.1, -0.2, 0.0, -0.3), (0.3, -0.1, -0.2, 0.0, 0.3),
                        (0.3, -0.1, -0.2, 0.0, -0.3)]),
    "playful": ("wheee", [(0.0, 0.0, -0.2, 0.4, 0.3), (0.3, 0.0, -0.2, -0.4, -0.3), (0.3, 0.0, -0.2, 0.4, 0.3)]),
    "scared": ("alarm", [(0.0, 0.3, 0.3, 0.0, 0.0), (0.4, 0.3, 0.3, 0.5, 0.0), (0.4, 0.3, 0.3, -0.5, 0.0), (0.6, 0.3, 0.3, 0.0, 0.0)]),
    "angry": ("alarm", [(0.0, -0.2, 0.0, 0.0, 0.0), (0.25, 0.3, 0.2, 0.0, 0.0), (0.25, -0.2, 0.0, 0.0, 0.0), (0.25, 0.3, 0.2, 0.0, 0.0)]),
    "sad": ("coo", [(0.0, 0.3, 0.4, 0.0, 0.0), (1.0, 0.3, 0.4, 0.2, 0.0), (1.0, 0.3, 0.4, -0.2, 0.0), (1.0, 0.3, 0.4, 0.0, 0.0)]),
    "lonely": ("inquire", [(0.0, -0.2, -0.2, 0.6, 0.0), (0.9, -0.2, -0.2, -0.6, 0.0), (0.9, 0.2, 0.3, 0.0, 0.0), (0.8, 0.2, 0.3, 0.0, 0.0)]),
    "bored": ("inquire", [(0.0, 0.0, 0.0, 0.6, 0.0), (1.0, 0.0, 0.0, -0.6, 0.0), (1.0, 0.0, 0.2, 0.0, 0.2), (0.8, 0.0, 0.2, 0.0, 0.2)]),
    "hungry": ("peck", [(0.0, 0.3, 0.4, 0.0, 0.0), (0.3, 0.0, 0.0, 0.0, 0.0), (0.3, 0.3, 0.4, 0.0, 0.0), (0.3, 0.0, 0.0, 0.0, 0.0)]),
    "thirsty": ("chirp", [(0.0, 0.3, 0.4, 0.0, 0.0), (0.5, -0.2, -0.4, 0.0, 0.0), (0.7, -0.2, -0.4, 0.0, 0.0)]),
    "sleepy": ("coo", [(0.0, 0.2, 0.4, 0.0, 0.1), (0.8, 0.0, 0.0, 0.0, 0.0), (0.4, 0.3, 0.4, 0.0, 0.1), (1.0, 0.3, 0.4, 0.0, 0.1)]),
    "curious": ("chirp", [(0.0, -0.1, 0.0, 0.3, 0.4), (1.0, -0.1, 0.0, -0.3, -0.4), (1.0, -0.1, 0.0, -0.3, -0.4)]),
    "proud": ("greet", [(0.0, -0.3, -0.3, 0.0, 0.0), (0.6, -0.3, -0.3, 0.5, 0.0), (0.6, -0.3, -0.3, -0.5, 0.0), (0.6, -0.3, -0.3, 0.0, 0.0)]),
}


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
    """The brain's intent as one of the commands the simulated duck actually obeys. `marching` is whether
    this is one of its walking bursts; between bursts a duck that means to go forward stands and waits."""
    if vx < -WALK_AT:
        return BACK
    if vx <= WALK_AT or not marching:
        return STAND  # going nowhere: it does not pivot on the spot for the sake of it either
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
        """(x, y, heading) in the simulator's own frame."""
        r = self._ask({"op": "read"})
        w, x, y, z = r["imu"]["quat"]
        return r["trunk"][0], r["trunk"][1], math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def close(self) -> None:
        self.f.close()
        self.sock.close()


class MujocoBody(Stub):
    def __init__(self, n: int, seed: int, sock_dir: str, sim_state: str = SIM_STATE, truth_port: int = TRUTH_PORT,
                 origin=None, cameras=(), **garden):
        """cameras: which ducks see through their simulated head camera (the ones duck-sim was given in
        DUCK_SIM_CAMERAS), by index. The rest see the garden drawn from above, as the stub's ducks do.
        The camera sees MuJoCo's world, which has the other ducks and the floor in it and none of the
        garden's dishes, tree or pond, so it is off unless asked for."""
        names = [f"duck-{c}" for c in "abcdefghij"[:n]]
        self.robot_paths = [os.path.join(sim_state, f"{name}.sock") for name in names]
        self.robots = [Client(path) for path in self.robot_paths]
        self.kicks = np.zeros(n, bool)  # which foot kicked last
        self.truth = [Truth(truth_port + i) for i in range(n)]
        # where MuJoCo's (0, 0) sits in the garden: by default, so that the row of ducks straddles the middle
        self.origin = np.asarray(origin if origin is not None
                                 else (self.world.size / 2, self.world.size / 2 - SPACING_M * (n - 1) / 2), float)
        self.turning = np.zeros(n)  # the brain's turning intent, smoothed
        self.banked = np.zeros(n)  # metres the brain has asked for and the duck has not walked yet
        self.marching = np.zeros(n, bool)
        self.idle_for = np.zeros(n)  # seconds on its feet and not marching
        self.eating_until = np.zeros(n)  # garden time until which it counts as eating or drinking
        self.acting = [[] for _ in range(n)]  # head poses still to come in an emote, as (garden time, pose)
        self.sat = np.zeros(n, bool)  # sitting because it wants nothing (asleep is `relaxed`)
        self.cameras = {i: SimCamera(FRAME_PORT + i) for i in cameras}
        self.got_up_at = np.full(n, -np.inf)
        self.limp = np.zeros(n, bool)  # let go so as to settle flat, on the way to getting up
        self.rising_until = np.zeros(n)  # garden time until which a duck is left alone to stand up
        self.turn = 0  # whose turn it is to be asked whether it has fallen
        super().__init__(n, seed, sock_dir, **garden)  # serves the brain's sockets and sends the first frames
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
        for rx, ry, r in self.world.rocks:  # a rock turns a robot back as the fence does; nothing can push one
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
        """Fallen and not asleep: stand up. How depends on how the duck went down, and both were found the
        hard way, on ducks that lay there for most of a day.
        - Tipped over while sitting: it is still in the sit policy, where enable and init do nothing at all.
          The sit_toggle stands it. (A seated duck settles into a forward lean that robotd already calls
          fallen, so a sleeper reads as fallen too; that one is left to sleep.)
        - Over on its side: the policy that rises from the floor cannot start from there. Going limp first
          lets the duck settle flat on its face, and from flat, `robot.enable` stands it up in ten seconds.
        While a duck is limp or rising it is told nothing at all (see `_move`). The brain goes on asking it
        to march and pivot every tick, and a duck half way up that is told to walk, or even to stand, goes
        straight back over: fallen ducks spent whole days failing to get up that way (timelines, 2026-09-20)."""
        # One duck a call, in turn: asking a robot how it is waits for its next state, up to 20 ms, which is
        # a whole tick if all five are asked at once.
        self.turn = (self.turn + 1) % len(self.robot_paths)
        for i, path in [(self.turn, self.robot_paths[self.turn])]:
            if self.relaxed[i] or self.sat[i] or self.t - self.got_up_at[i] < (LIMP_S if self.limp[i] else GET_UP_EVERY_S):
                continue  # asleep or sitting on purpose (a seated duck reads as fallen), or helped a moment ago
            if self.limp[i]:  # it has had its moment on the floor
                # a request, with an id: robotd answers enable, relax, init and stop, and one sent as a
                # notification is dropped without a word, which is how ducks went limp (a request) and then
                # never rose (a notification), all day, three days running
                self.robots[i].call("robot.enable", on=True)
                self.limp[i], self.got_up_at[i], self.rising_until[i] = False, self.t, self.t + RISE_S
                continue
            state = robot_state(path)
            if state["policy"] == "held":  # let go and never picked up again: it needs no moment, only the word
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

    # what the brain asks of the body goes on to the robot; what it means for the garden stays here

    def _move(self, i: int, p: dict) -> None:
        super()._move(i, p)  # kept for the viewer and for `relaxed`
        vx, _, vyaw = self.cmd[i]
        self.turning[i] += (vyaw - self.turning[i]) * min(DT / TURN_MEMORY_S, 1.0)
        if self.limp[i] or self.t < self.rising_until[i]:
            # Not even "stand": any robot.move hands the robot from the policy that is getting it up to the
            # one that walks. By hand, with nothing else talking to it, a duck flat on its face stood up in
            # six seconds; in the garden, told to stand fifty times a second, the same duck lay there all day.
            return
        if self.relaxed[i]:
            return
        urgent = vx >= URGENT_MS or vx < -WALK_AT
        self.banked[i] = min(self.banked[i] + max(vx, 0.0) * DT, BANK_M) if vx > WALK_AT else 0.0  # wanting to stop is forgetting the walk
        if self.sat[i]:
            if urgent or self.banked[i] >= BOUT_M:  # up it gets, and nothing else until it is up
                if self._sitting(i):  # asked, not assumed: a toggle sent to a duck that never sat would sit it
                    self.robots[i].notify("robot.do", skill="sit_toggle")
                self.sat[i], self.marching[i], self.rising_until[i] = False, True, self.t + TOGGLE_S
            return
        if self.marching[i]:
            self.banked[i] -= MARCH_MS * DT
            self.marching[i] = urgent or self.banked[i] > 0
        else:
            self.marching[i] = urgent or self.banked[i] >= BOUT_M
        eating = self.t < self.eating_until[i]  # a duck with its beak in a dish is not at a loose end
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
        """Asleep is sitting down. `robot.relax` cuts the torque and a duck with no torque lies flat on its
        face (trunk 0.04 m); it does get up again from that, but it is no way to treat a robot. The
        robot's own sit_toggle folds it upright onto the floor and back (measured: 0.116 to 0.059 m)."""
        super()._relax(i, p)
        if not self.sat[i] and not self._sitting(i):
            self.robots[i].notify("robot.do", skill="sit_toggle")
        self.sat[i] = False  # it is asleep now, which is its own state

    def _init(self, i: int, p: dict) -> None:
        super()._init(i, p)
        if self._sitting(i):
            self.robots[i].notify("robot.do", skill="sit_toggle")
            self.rising_until[i] = self.t + RISE_S

    def knock_down(self, j: int) -> None:
        """A kick that lands: the duck goes limp and drops, and `get_up` has it back on its feet in ten to
        fifteen seconds, the way it raises any duck that went over. The kicker keeps its distance
        (`keep_apart`), so the blow is the garden's and the fall is the robot's own."""
        super().knock_down(j)
        if not self.limp[j] and not self.relaxed[j]:
            self.robots[j].call("robot.relax")
            self.limp[j], self.got_up_at[j], self.sat[j] = True, self.t, False

    def _sound(self, i: int, p: dict) -> None:
        super()._sound(i, p)
        self.robots[i].notify("robot.sound", tag=p["tag"])

    def _do(self, i: int, p: dict) -> None:
        """What a duck does is the garden's business (the bite, the sip, the shove, the hat) and the
        robot's to act out, in the skills it has: a headbutt is a kick, zoomies are a roulade."""
        super()._do(i, p)
        if p["skill"] in ("ground_pick", "drink"):
            self.eating_until[i] = self.t + 2.0
        if self.limp[i] or self.t < self.rising_until[i]:
            return
        if p["skill"].startswith("emote_"):
            return self._emote(i, p["skill"][len("emote_"):])
        if self.sat[i]:
            return
        skill = p["skill"]
        if skill == "headbutt":
            self.kicks[i] = not self.kicks[i]
            skill = "kick_left" if self.kicks[i] else "kick_right"
        skill = {"zoomies": "roulade"}.get(skill, skill)
        if skill in ROBOT_SKILLS:
            self.robots[i].notify("robot.do", skill=skill)

    def posture(self) -> list[str]:
        return ["down" if limp else "sat" if sat else "up" for limp, sat in zip(self.limp, self.sat)]

    def _emote(self, i: int, feeling: str) -> None:
        """Act a feeling out: say it, and queue the head's poses for step() to play. A playful duck on its
        feet rolls over instead, which is the one emote the legs join in."""
        if self.acting[i] or self.relaxed[i]:
            return
        tag, poses = EMOTE_ACTS[feeling]
        self.robots[i].notify("robot.sound", tag=tag)
        if feeling == "playful" and not self.sat[i] and not self.marching[i]:
            self.robots[i].notify("robot.do", skill="roulade")
            return
        at = self.t
        for after, *pose in poses + [(0.5, *LEVEL)]:
            at += after
            self.acting[i].append((at, pose))

    def _act(self) -> None:
        """Send whichever queued head poses have come due. A duck that went down mid-emote drops the rest."""
        for i, queue in enumerate(self.acting):
            if queue and (self.limp[i] or self.t < self.rising_until[i]):
                queue.clear()
            while queue and queue[0][0] <= self.t:
                self.robots[i].notify("robot.head", **dict(zip(ROBOT_PARAMS["robot.head"], queue.pop(0)[1])))

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
    ap.add_argument("--godot", action="store_true", help="publish the world for the Godot garden (viewer/godot/)")
    ap.add_argument("--brain", action="store_true", help="drive the ducks from here with the real brain")
    ap.add_argument("--labels", default="Bully,Napper,Carefree,Chatty,Scaredy")
    ap.add_argument("--cameras", default="", help="ducks that see through their sim camera, by letter: a or a,c")
    args = ap.parse_args()
    os.makedirs(args.sock_dir, exist_ok=True)
    cameras = [ord(c.strip()) - ord("a") for c in args.cameras.split(",") if c.strip()]
    body = MujocoBody(args.ducks, args.seed, args.sock_dir, cameras=cameras, **DEMO_GARDEN)
    view = None
    if args.view:
        from viewer.debug2d import Viewer
        view = Viewer(body.world)
    server = None
    if args.brain:
        from body import frames
        from brain.data import load_connectome, named_sets
        from brain.personality import preset, stack
        from brain.server import BrainServer
        print("loading the connectome, which takes a moment ...")
        W, ann = load_connectome()
        rng = np.random.default_rng(args.seed)
        labels = (args.labels.split(",") * args.ducks)[:args.ducks]
        bodies = [(os.path.join(args.sock_dir, f"{name}.sock"), frames.FRAME_PORT + i) for i, name in enumerate(body.names)]
        server = BrainServer(W, ann, named_sets(ann), bodies, args.seed, personality=stack([preset(x, rng) for x in labels]))
        print(f"driving {', '.join(labels)} in MuJoCo")
    world_out = None
    if args.godot:
        from viewer.snapshot import Snapshot
        world_out = Snapshot()
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
            next_t += DT  # the robots run on the wall clock, so the garden has to as well
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        body.close()


if __name__ == "__main__":
    assert snap(0.08, 0.0) == FORWARD and snap(0.0, 0.0) == STAND and snap(-0.3, 0.0) == BACK and snap(0.08, 0.0, False) == STAND
    assert snap(0.08, TURN_AT + 0.1) == LEFT and snap(0.08, -TURN_AT - 0.1) == RIGHT and snap(0.08, TURN_AT / 2) == FORWARD and snap(0.0, 2.0) == STAND
    main()
