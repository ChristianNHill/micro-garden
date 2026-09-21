"""2D stub body: kinematic ducks on the garden plane behind the microduck contract (PLAN.md Gate 3).

One Unix socket per duck (duck-a.sock ...), like duck-sim, plus control.sock with stub-only
sim.step {n} and sim.state. Every body step sends each duck's sensory frame over UDP to
frame_port + duck. robot.do ground_pick on a dish eats it; robot.do headbutt pushes a touching duck
in front of the attacker back by PUSH_M and knocks it over for DOWN_S. robot.do drink at the pond's shore band takes a sip; past the
shore a duck swims at SWIM_SPEED. robot.sound is logged. With fruit_every_s set, the shade tree drops fruit on that
period; garden.shake_tree on control.sock (a player action) drops SHAKE_FRUIT at once. In the viewer,
click the tree; Tab takes the wheel of the selected duck (W A S D drive it, Tab gives it back); P pets it, C claps, F feeds by hand at the mouse, M starts or stops
music there, H puts a hat on the selected duck or takes it off. garden.pet {duck} is the player's hand on a duck's head: bristles, and a reward.
garden.scare claps, startling every duck. garden.hand {x, y, feed} puts the hand in the garden, where
the ducks can see it, and drops that many bites at it. garden.music {x, y, on} picks the music up and puts it down there, or takes it away,
and garden.hat {duck, on} puts a hat on one; a duck shakes a hat off by grooming.

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
from body.stub2d import retina
from world.fields import (DAY_S, DUCK_SMELL_M, SHORE_M, SIZE_M, TREE, DUCK_R, World, contacts, daylight, duck_odor_at,
                          music_at, temperature_at, wind_on)

# Light air that starts in the north and swings right round the compass every 0.7 of a day, so that no
# hour always has the same wind: at once a day the south wind, the one that brings the pond to the ducks,
# only ever blew at night while they slept, and the Napper never drank. From the north the
# smell of the dish and the fruit tree lies across the garden and a duck at the pond can follow it home;
# from the south the pond's damp air reaches the ducks that are eating. A steady wind can only do one.
# The tree drops a fruit every 10 s. At 20 the garden ran out: the dish is gone in the first minute, the
# tree stops dropping with four things on the ground, which is all night while the ducks sleep, and what
# fell in their waking hours came to about 100 bites in twenty minutes against the 108 five ducks need. They
# ate all of it and a third of them still starved, the Bully first, since its appetite asks 40% more
# (Gate 9b; Claude's call while Chris was out, 2026-09-19, for him to ratify).
# The garden people watch is bigger than the one the gates measure in (Chris, 2026-09-21: 4 m was cramped for
# five ducks), and laid out after the Chao gardens of Sonic Adventure 2: a big pond tucked into the back
# corner, where a viewer can put a waterfall and cliffs behind it, the fruit tree on the other side of the
# back, and the lawn in front left open, with the dish and the stink on it. The rocks are
# the foot of that waterfall, stepping up from the pond's edge into the corner: solid, so no duck walks through
# what a viewer draws there. They stand on the shore and not in the water: the first ones reached half a metre
# into the pond, so every duck swimming or asleep at that end looked pressed against the waterfall (2026-09-21).
DEMO_GARDEN = dict(size=6.0, tree=(1.5, 4.3, 0.9), food_xy=((3.0, 2.2),), bites=10, danger_xy=((4.9, 1.3),),
                   pond=(4.5, 4.4, 1.2), rocks=((5.5, 5.47, 0.42), (5.95, 5.9, 0.55)), fruit_every_s=10.0, wind=(0.0, -1.0), wind_turns_s=0.7 * DAY_S,
                   personal_m=0.24,  # the drawn duck's width: they stop at each other and do not merge
                   music=None)  # the music box is the player's to put down (M), and to pick up again

DT = 0.02
# ponytail: guessed limits standing in for robotd's clamps; replace with the sim's real ones at Gate 10
MAX_V, MAX_VY, MAX_VYAW = 0.3, 0.15, 2.0
ANTENNA = np.array([0.06, 0.05])  # forward, lateral offset of each odor sample, metres
CONTROL_PARAMS = {"sim.step": {"n": 1}, "sim.state": {}, "garden.shake_tree": {}, "garden.pet": {"duck": 0},
                  "garden.scare": {}, "garden.hand": {"x": 0.0, "y": 0.0, "feed": 0},
                  "garden.music": {"x": 0.0, "y": 0.0, "on": 1}, "garden.hat": {"duck": 0, "on": 1},
                  "garden.drop_hat": {"x": 0.0, "y": 0.0}, "garden.drop_ball": {"x": 0.0, "y": 0.0}, "garden.volume": {"level": 0.75}, "garden.give": {"duck": 0}, "garden.drum": {"x": 0.0, "y": 0.0, "on": 1}, "garden.grab": {"x": 0.0, "y": 0.0}, "garden.hand_at": {"x": 0.0, "y": 0.0},
                  "garden.release": {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}}
SHAKE_FRUIT = 2
PUSH_M = 0.15
DOWN_S = 10.0  # a kicked duck goes over, and this is about how long a microduck takes to get back on its feet
SWIM_SPEED = 0.5  # fraction of commanded speed while swimming
SOUND_TAGS = {"alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee"}  # microduck's voice bank
SIT_AFTER_S = 3.0  # a duck that has not moved for this long is drawn sitting (body/mujoco/adapter.py really sits)
DRUM_REACH_M = 0.32  # how near a drum a duck has to be to tap it
KICK_REACH_M, KICK_MS = 0.22, 1.6  # how near its feet a ball has to be for a duck to kick it, and how fast it leaves
BALL_SEEN_M = 1.5  # a ball this far off fills half what it would at a duck's feet
NEAR_M = 1.0  # another duck nearer than this is the one a duck is with
EARSHOT_M = 2.0  # how far an alarm, a whoop or a cry carries
WITNESS_M = 1.5  # how near a duck has to be to a shove, or to a hat being taken, to have seen it
GRAB_M = 0.3  # how near the hand has to be to a thing to pick it up
THROW_MS, THROW_MAX_MS, THROW_S = 0.8, 3.0, 0.35  # let go faster than the first and it is thrown; capped; and how long it flies
HAND_S = 4.0  # how long the player's hand stays in the garden once it has come
HAND_SEEN_M = 1.5  # a hand this far off is half as plain as one at a duck's beak
CRY_S = 6.0  # how long a cry lasts
AUDIENCE_M = 1.5  # how near a duck has to be to a song or a dance to be its audience
PERFORMANCES = ("sing", "dance", "singdance")
MAX_BALLS = 3
HAT_REACH_M = 0.25  # a hat on the ground nearer than this is one a duck could put on
BITE_S = 0.5  # ground_pick takes this long, so at most one bite per BITE_S
# "headbutt", "drink", "preen" and "zoomies" are our names for what a duck does; this body acts the first three
# out in the garden and ignores the last. "emote_<feeling>" (brain/emotes.py) it writes down for the viewer to show. body/mujoco/adapter.py maps them onto the robot's own skills.

# How a duck acts out each feeling (brain/emotes.py), in every body: a voice tag, and its head through a few poses, each
# (seconds after the last, neck_pitch, head_pitch, head_yaw, head_roll) in radians, pitch positive down. The
# head works sitting as well as standing, so a duck at rest still shows what it feels. Every one ends level.
# One definition, on the robot's own four head joints: this body records the poses, the MuJoCo body forwards
# them to the robot, and a viewer draws whatever the head was told. There were two, one here for the robot and
# a looser one in Godot for a cartoon duck, and the same feeling looked different in each (Chris, 2026-09-21).
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
    # the signatures (brain/emotes.py): a stamp of the head, a long yawn, dips in the water, a song, a shiver
    "stomp": ("alarm", [(0.0, -0.2, -0.2, 0.0, 0.0), (0.2, 0.35, 0.3, 0.0, 0.0), (0.2, -0.2, -0.2, 0.0, 0.0), (0.2, 0.35, 0.3, 0.0, 0.0)]),
    "yawn": ("coo", [(0.0, -0.25, -0.5, 0.0, 0.1), (1.2, -0.3, -0.6, 0.0, 0.15), (0.8, 0.2, 0.3, 0.0, 0.0)]),
    "splash": ("wheee", [(0.0, 0.4, 0.5, 0.0, 0.0), (0.2, -0.2, -0.3, 0.0, 0.0), (0.2, 0.4, 0.5, 0.0, 0.0), (0.2, -0.2, -0.3, 0.0, 0.0)]),
    "sing": ("chirp", [(0.0, -0.25, -0.4, 0.35, 0.0), (0.35, -0.25, -0.4, -0.35, 0.0), (0.35, -0.25, -0.4, 0.35, 0.0),
                       (0.35, -0.25, -0.4, -0.35, 0.0)]),
    "cower": ("inquire", [(0.0, 0.45, 0.5, 0.0, 0.0), (0.3, 0.45, 0.5, 0.15, 0.0), (0.15, 0.45, 0.5, -0.15, 0.0),
                          (0.15, 0.45, 0.5, 0.15, 0.0), (0.15, 0.45, 0.5, -0.15, 0.0), (0.8, 0.45, 0.5, 0.0, 0.0)]),
    # to music it likes: the head nods on the beat and swings side to side, four beats, and it says nothing
    "dance": (None, [(0.0, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0),
                     (0.35, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0)]),
    # and the two at once, which is what a duck mostly does with music it likes
    "singdance": ("chirp", [(0.0, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0),
                     (0.35, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0)]),
    # miserable, and saying so: head down and shaking
    "cry": ("coo", [(0.0, 0.4, 0.5, 0.0, 0.0), (0.4, 0.4, 0.5, 0.2, 0.0), (0.4, 0.4, 0.5, -0.2, 0.0), (0.4, 0.4, 0.5, 0.2, 0.0),
                    (0.4, 0.4, 0.5, -0.2, 0.0), (0.8, 0.4, 0.5, 0.0, 0.0)]),
}


class Stub:
    def __init__(self, n: int, seed: int, sock_dir: str, food_xy=((3.0, 3.0), (1.0, 1.0)), danger_xy=(),
                 pond=None, bites=1, fruit_every_s=None, frame_port: int = frames.FRAME_PORT, pose=None,
                 wind=None, wind_turns_s=None, music=None, size=SIZE_M, tree=TREE, rocks=(), balls=(), personal_m=0.0):
        rng = np.random.default_rng(seed)
        self.fruit_rng = np.random.default_rng(seed + 1)
        self.fruit_every_s = fruit_every_s
        self.world = World(food_xy, danger_xy, pond, bites, wind, wind_turns_s, music, size, tree, rocks)
        for x, y in balls:  # toys put down before anyone is watching, for a gate
            self.world.balls = np.vstack([self.world.balls, [x, y, 0.0, 0.0]])
        self.pose = np.column_stack([rng.uniform(0.5, size - 0.5, (n, 2)), rng.uniform(-np.pi, np.pi, n)])
        if pose is not None:
            self.pose = np.array(pose, float).reshape(n, 3)
        self.frame_port = frame_port
        self.eaten = []  # (t, duck)
        self.headbutts = []  # (t, attacker, victim)
        self.emotes = []  # (t, duck, feeling)
        self.acting = [[] for _ in range(n)]  # head poses still to come in an emote, as (garden time, pose)
        self.bumped = np.zeros(n, bool)
        self.petted = np.zeros(n, bool)
        self.scared = np.zeros(n, bool)
        self.hats = np.zeros(n, bool)
        self.still_for = np.zeros(n)  # seconds since it last moved, for posture()
        self.hat_style = np.full(n, -1)  # which hat each wears: a number a viewer makes a hat from; -1 for none
        self.hat_items = []  # hats lying in the garden, as [x, y, style], for a duck to put on if it likes
        self.donned = []  # (t, duck) each time one puts a hat on
        self.kicks = []  # (t, duck) each time one kicks a ball
        self.kicked = np.zeros(n, bool)
        self.drums = []  # (t, duck) each tap on the drum
        self.drummed = np.zeros(n, bool)
        self.saw_show = np.zeros(n, bool)
        # who did what to whom this step, for the ducks it happened near (-1 is nobody); sent and then cleared
        self.events = {name: np.full(n, -1.0) for name in frames.IDS if name != "near_id"}
        self.heard = {"heard_alarm": np.zeros(n), "heard_joy": np.zeros(n)}
        self.hand_fed = np.zeros(n, bool)
        self.crying_until = np.zeros(n)  # garden time until which a duck is crying, for the others to hear
        self.hand_until = 0.0
        self.held = None  # what the player's hand is carrying: ("ball" | "hat" | "food" | "music" | "duck", which one)
        self.thrown = np.zeros(n, bool)
        self.throws = []  # (t, duck) each time the player throws one
        self.given = []  # (t, duck) each time the player hands one a fruit  # a duck nearby has just begun to sing or dance
        self.velocity = np.zeros((n, 2))  # metres a second over the ground, for what a duck walks into
        self.hat_rng = np.random.default_rng(seed + 2)
        self.preened = []  # (t, duck) each time one is shaken off
        self.pets = []  # (t, duck)
        self.ate = np.zeros(n, bool)
        self.drank = np.zeros(n, bool)
        self.sounds = []  # (t, duck, tag)
        self.last_bite = np.full(n, -np.inf)
        self.cmd = np.zeros((n, 3))
        self.head = np.zeros((n, 4))
        self.relaxed = np.zeros(n, bool)
        self.seen = None
        self.touch_m = 2 * DUCK_R  # how near two ducks' centres are when they touch
        # How near two ducks' centres can come, or 0 for ducks that pass through each other, which is what the
        # gates were measured with. The garden people watch draws its ducks larger than life, and there they
        # walked through one another (Chris, 2026-09-21); with a personal space they stop at each other, and
        # touching is that distance and a little, so a shove, company and comfort all still reach.
        self.personal_m = float(personal_m)
        if self.personal_m > 0:
            self.touch_m = self.personal_m + 0.05
        self.down_until = np.zeros(n)  # garden time until which a kicked duck is on the floor
        self.t = 0.0
        self.lock = threading.Lock()
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.names = [f"duck-{c}" for c in string.ascii_lowercase[:n]]
        self.servers = [
            serve(os.path.join(sock_dir, f"{name}.sock"), ROBOT_PARAMS, self._robot_call(i), self.lock)
            for i, name in enumerate(self.names)
        ]
        self.servers.append(serve(os.path.join(sock_dir, "control.sock"), CONTROL_PARAMS, self._control_call, self.lock))
        self.send_frames()

    def _robot_call(self, i: int):
        handlers = {"robot.move": self._move, "robot.head": self._head, "robot.do": self._do,
                    "robot.sound": self._sound, "robot.stop": self._stop, "robot.relax": self._relax,
                    "robot.init": self._init}

        def call(method, p):
            handlers[method](i, p)
            return {}
        return call

    def _move(self, i: int, p: dict) -> None:
        if self.t < self.down_until[i]:
            self.cmd[i] = 0  # on the floor: whatever its brain wants, its legs are not under it
        elif not self.relaxed[i]:
            self.cmd[i] = np.clip([float(p["vx"]), float(p["vy"]), float(p["vyaw"])],
                                  [-MAX_V, -MAX_VY, -MAX_VYAW], [MAX_V, MAX_VY, MAX_VYAW])

    def _head(self, i: int, p: dict) -> None:
        self.head[i] = [float(p[k]) for k in ROBOT_PARAMS["robot.head"]]

    def _do(self, i: int, p: dict) -> None:
        """Other skill names are accepted and do nothing on the 2D stub."""
        if p["skill"] == "ground_pick":
            self._pick(i)
        elif p["skill"] == "headbutt":
            self._headbutt(i)
        elif p["skill"] == "drink":
            self._drink(i)
        elif p["skill"].startswith("emote_"):
            self._emote(i, p["skill"][len("emote_"):])
        elif p["skill"] == "kick":
            fwd = np.array([np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])])
            for ball in self.world.balls:
                rel = ball[:2] - self.pose[i, :2]
                if np.linalg.norm(rel) < KICK_REACH_M and rel @ fwd > 0:
                    ball[2:] += KICK_MS * fwd
                    self.kicks.append((self.t, i))
                    self.kicked[i] = True
                    break
        elif p["skill"] == "drum" and self.world.drum is not None:
            rel = np.asarray(self.world.drum) - self.pose[i, :2]
            if np.linalg.norm(rel) < DRUM_REACH_M:
                self.drums.append((self.t, i))
                self.drummed[i] = True
                self.sounds.append((self.t, i, "drum"))  # the drum's voice and not the duck's; a viewer plays it quietly
                near = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < AUDIENCE_M  # and it is a performance
                near[i] = False
                self.saw_show |= near
                self.events["show_by"][near] = i
        elif p["skill"] == "wear" and not self.hats[i]:
            near = [k for k, (x, y, _) in enumerate(self.hat_items) if np.hypot(x - self.pose[i, 0], y - self.pose[i, 1]) < HAT_REACH_M]
            if near:
                self.hats[i], self.hat_style[i] = True, self.hat_items.pop(near[0])[2]
                self.donned.append((self.t, i))
                saw = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < WITNESS_M
                saw[i] = False
                self.events["hat_taken_by"][saw] = i
        elif p["skill"] == "preen" and self.hats[i]:
            self.hats[i] = False  # shaken off, and it lands where the duck stands, for whoever wants it next
            self.hat_items.append([float(self.pose[i, 0]), float(self.pose[i, 1]), int(max(self.hat_style[i], 0))])
            self.hat_style[i] = -1
            self.preened.append((self.t, i))

    def _emote(self, i: int, feeling: str) -> None:
        """Act a feeling out: say it, and queue the head's poses for step() to play. One at a time, and not
        in its sleep."""
        if self.acting[i] or self.relaxed[i] or self.t < self.down_until[i]:
            return  # one at a time, not in its sleep, and not from flat on the floor
        tag, poses = EMOTE_ACTS[feeling]
        self.emotes.append((self.t, i, feeling))
        if feeling == "cry":
            self.crying_until[i] = self.t + CRY_S
        if feeling in PERFORMANCES:  # and it has an audience: what they make of it is theirs (brain/physiology.py)
            near = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < AUDIENCE_M
            near[i] = False
            self.saw_show |= near
            self.events["show_by"][near] = i
        if tag:
            self._sound(i, {"tag": tag})
        at = self.t
        for after, *pose in poses + [(0.5, *LEVEL)]:
            at += after
            self.acting[i].append((at, pose))

    def _act(self) -> None:
        """Turn the head to whichever queued poses have come due."""
        for i, queue in enumerate(self.acting):
            while queue and queue[0][0] <= self.t:
                self._head(i, dict(zip(ROBOT_PARAMS["robot.head"], queue.pop(0)[1])))

    def _sound(self, i: int, p: dict) -> None:
        if p["tag"] not in SOUND_TAGS:
            raise ValueError(f"unknown sound tag {p['tag']!r}; known: {sorted(SOUND_TAGS)}")
        self.sounds.append((self.t, i, p["tag"]))
        if p["tag"] in ("alarm", "wheee"):  # a fright and a whoop both carry, and both are catching
            near = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < EARSHOT_M
            near[i] = False
            self.heard["heard_alarm" if p["tag"] == "alarm" else "heard_joy"][near] = 1.0

    def _stop(self, i: int, p: dict) -> None:
        self.cmd[i] = 0

    def _relax(self, i: int, p: dict) -> None:
        self.relaxed[i], self.cmd[i] = True, 0

    def _init(self, i: int, p: dict) -> None:
        self.relaxed[i] = False

    def _pick(self, i: int) -> None:
        *_, dish = contacts(self.pose[:, :2], self.pose[:, 2], self.world.food)
        if dish[i] >= 0 and self.t - self.last_bite[i] >= BITE_S:
            self.events["ate_kind"][i] = self.world.kinds[dish[i]]
            hand = self.world.hand
            self.hand_fed[i] = hand is not None and np.hypot(hand[0] - self.pose[i, 0], hand[1] - self.pose[i, 1]) < 0.5
            self.world.eat(dish[i])
            self.eaten.append((self.t, i))
            self.last_bite[i], self.ate[i] = self.t, True

    def _drink(self, i: int) -> None:
        shore = abs(self.world.pond_distance(self.pose[i, :2])) <= SHORE_M
        if shore and self.t - self.last_bite[i] >= BITE_S:
            self.last_bite[i], self.drank[i] = self.t, True

    def articulation(self) -> list[dict]:
        """Per duck, what a jointed body knows of itself, for a viewer: nothing here, so a viewer poses the
        duck by rule. The MuJoCo body gives the robot's joints, height and lean."""
        return [{} for _ in self.names]

    def posture(self) -> list[str]:
        """Per duck, "up", "sat" or "down", for a viewer to draw. A duck that has stood still for SIT_AFTER_S
        has sat down, as the robots do. Here that is only how it is drawn: it is up and walking the step it
        wants to be, so nothing a gate measures moves."""
        return ["down" if self.t < until else "sat" if still > SIT_AFTER_S else "up"
                for until, still in zip(self.down_until, self.still_for)]

    def down_left(self) -> np.ndarray:
        """Seconds until each duck that is down is on its feet again, 0 for one that is up: a viewer fits the
        fall and the getting up inside it, so that a duck is standing by the time it walks."""
        return np.maximum(self.down_until - self.t, 0.0)

    def _swimming(self) -> np.ndarray:
        return self.world.pond_distance(self.pose[:, :2]) < -SHORE_M

    def _headbutt(self, i: int) -> None:
        fwd = np.array([np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])])
        rel = self.pose[:, :2] - self.pose[i, :2]
        hit = (np.linalg.norm(rel, axis=1) < self.touch_m) & (rel @ fwd > 0)
        hit[i] = False
        for j in np.flatnonzero(hit):
            self.pose[j, :2] = np.clip(self.pose[j, :2] + PUSH_M * fwd, DUCK_R, self.world.size - DUCK_R)
            self.bumped[j] = True
            self.events["bumped_by"][j] = i
            seen = np.linalg.norm(self.pose[:, :2] - self.pose[j, :2], axis=1) < WITNESS_M
            seen[[i, j]] = False
            self.events["saw_shove_by"][seen], self.events["saw_shove_of"][seen] = i, j
            self.headbutts.append((self.t, i, j))
            self.knock_down(j)

    def knock_down(self, j: int) -> None:
        """A kick that lands puts a duck on the floor (Chris, 2026-09-21). Here that is a duck that cannot
        move until it is up again; a body with legs falls over for real."""
        self.down_until[j] = self.t + DOWN_S
        self.cmd[j] = 0

    def shake_tree(self) -> int:
        """Callers hold self.lock (control calls do; the viewer takes it)."""
        return self.world.drop_fruit(self.fruit_rng, SHAKE_FRUIT)

    def _pet(self, p):
        i = int(p["duck"])
        self.petted[i] = True
        self.pets.append((self.t, i))

    def _place_music(self, p):
        self.world.music = (float(p["x"]), float(p["y"])) if int(p["on"]) else None

    def _hat(self, p):
        """The player's hand putting a hat on a duck, or taking it away."""
        i = int(p["duck"])
        self.hats[i] = bool(int(p["on"]))
        self.hat_style[i] = int(self.hat_rng.integers(1_000_000)) if self.hats[i] else -1

    def _volume(self, p):
        """How loud the music box plays, 0 to 1. Off is off for the ducks too."""
        self.world.music_volume = float(np.clip(p["level"], 0, 1))

    def _drum(self, p):
        self.world.drum = (float(p["x"]), float(p["y"])) if int(p["on"]) else None

    def _drop_ball(self, p):
        """A ball for the ducks, put down where the player says. A few is plenty."""
        self.world.balls = np.vstack([self.world.balls, [float(p["x"]), float(p["y"]), 0.0, 0.0]])[-MAX_BALLS:]

    def _drop_hat(self, p):
        """A hat left in the garden, no two alike. Whether anyone wears it is up to the ducks."""
        self.hat_items.append([float(p["x"]), float(p["y"]), int(self.hat_rng.integers(1_000_000))])

    def _scare(self, p):
        self.scared[:] = True  # a clap: everything in the garden hears it

    def _keep_apart(self) -> None:
        """Two ducks nearer than personal_m are each moved half the difference apart: solid to each other, as
        they are to a rock. A few passes settle a huddle."""
        if self.personal_m <= 0 or len(self.pose) < 2:
            return
        xy = self.pose[:, :2]
        for _ in range(3):
            rel = xy[:, None] - xy[None]  # [i, j]: i from j
            d = np.linalg.norm(rel, axis=-1)
            np.fill_diagonal(d, np.inf)
            close = d < self.personal_m
            if not close.any():
                return
            away = np.where(d[..., None] > 1e-6, rel / np.maximum(d, 1e-6)[..., None], [1.0, 0.0])
            xy += (away * (np.where(close, self.personal_m - d, 0.0) / 2)[..., None]).sum(axis=1)
        np.clip(xy, DUCK_R, self.world.size - DUCK_R, out=xy)

    def _things(self) -> list:
        """Everything the hand could pick up, as (kind, which, x, y). Things before ducks, so a fruit beside
        a duck is the fruit."""
        w = self.world
        things = [("ball", k, *b[:2]) for k, b in enumerate(w.balls)] + [("hat", k, h[0], h[1]) for k, h in enumerate(self.hat_items)]
        things += [("food", k, *xy) for k, xy in enumerate(w.food)] + ([("music", 0, *w.music)] if w.music else [])
        things += [("drum", 0, *w.drum)] if w.drum else []
        return things + [("duck", i, *xy) for i, xy in enumerate(self.pose[:, :2]) if self.can_carry_ducks]

    can_carry_ducks = True  # a simulated robot is not something a cursor can lift (body/mujoco/adapter.py)

    def _grab(self, p):
        """The hand closes on whatever is nearest it, if anything is near enough (Sonic Adventure's gardens)."""
        at = np.array([float(p["x"]), float(p["y"])])
        self._hand_at(p)
        near = [(np.hypot(x - at[0], y - at[1]) + (0.1 if kind == "duck" else 0.0), kind, which) for kind, which, x, y in self._things()]
        near = [n for n in near if n[0] < GRAB_M + (0.1 if n[1] == "duck" else 0.0)]
        self.held = min(near)[1:] if near else None

    def _hand_at(self, p):
        """The hand moves, and what it holds goes with it."""
        at = (float(np.clip(p["x"], 0.05, self.world.size - 0.05)), float(np.clip(p["y"], 0.05, self.world.size - 0.05)))
        self.world.hand, self.hand_until = at, self.t + HAND_S
        self._carry(at)

    def _carry(self, at, velocity=(0.0, 0.0)) -> None:
        if self.held is None:
            return
        kind, which = self.held
        w = self.world
        if kind == "ball" and which < len(w.balls):
            w.balls[which] = [*at, *velocity]
        elif kind == "hat" and which < len(self.hat_items):
            self.hat_items[which][:2] = at
        elif kind == "food" and which < len(w.food):
            w.food[which] = at
        elif kind == "music":
            w.music = tuple(at)
        elif kind == "drum":
            w.drum = tuple(at)
        elif kind == "duck":
            self.pose[which, :2] = at
            self.cmd[which] = 0

    def _release(self, p):
        """The hand opens. Let go gently, the thing is put down; let go on the move, it is thrown: a ball
        rolls off, anything else lands a little way on, and a duck lands on its side and thinks less of you."""
        if self.held is None:
            return
        v = np.array([float(p["vx"]), float(p["vy"])])
        speed = np.linalg.norm(v)
        v = v * min(1.0, THROW_MAX_MS / max(speed, 1e-9))
        at = np.array([float(p["x"]), float(p["y"])])
        kind, which = self.held
        thrown = speed > THROW_MS
        if kind == "ball":
            self._carry(tuple(at), tuple(v) if thrown else (0.0, 0.0))
        else:
            land = np.clip(at + (v * THROW_S if thrown else 0.0), 0.1, self.world.size - 0.1)
            self._carry(tuple(land))
            if kind == "duck" and thrown:
                self.thrown[which] = True
                self.throws.append((self.t, which))
                self.knock_down(which)
        self.held = None

    def _give(self, p):
        """The player holds a fruit out to one duck: the hand comes to it, with a few bites at its beak."""
        i = int(p["duck"])
        h = self.pose[i, 2]
        at = self.pose[i, :2] + 0.12 * np.array([np.cos(h), np.sin(h)])
        self._hand({"x": at[0], "y": at[1], "feed": 3})
        self.given.append((self.t, i))

    def _hand(self, p):
        self.world.hand = (float(p["x"]), float(p["y"]))
        self.hand_until = self.t + HAND_S
        if p.get("feed"):
            self.world.add_food(self.world.hand, int(p["feed"]))

    def _sim_step(self, p):
        for _ in range(int(p["n"])):
            self.step()
        if int(p["n"]) == 0:
            self.send_frames()  # lets a lockstep client read the starting state
        return {"t": self.t, "eaten": len(self.eaten)}

    def _control_call(self, method, p):
        handlers = {"garden.pet": self._pet, "garden.music": self._place_music, "garden.hat": self._hat,
                    "garden.drop_hat": self._drop_hat, "garden.drop_ball": self._drop_ball, "garden.drum": self._drum, "garden.volume": self._volume, "garden.give": self._give, "garden.grab": self._grab, "garden.hand_at": self._hand_at,
                    "garden.release": self._release,
                    "garden.scare": self._scare, "garden.hand": self._hand, "sim.step": self._sim_step,
                    "garden.shake_tree": lambda p: {"fell": self.shake_tree()}, "sim.state": lambda p: self.state()}
        return handlers[method](p) or {}

    def step(self) -> None:
        before = self.pose[:, :2].copy()
        x, y, h = self.pose.T
        vx, vy, vyaw = (self.cmd * np.where(self._swimming(), SWIM_SPEED, 1.0)[:, None]).T
        h += vyaw * DT
        x += (vx * np.cos(h) - vy * np.sin(h)) * DT
        y += (vx * np.sin(h) + vy * np.cos(h)) * DT
        self.still_for = np.where(np.hypot(vx, vy) > 0.01, 0.0, self.still_for + DT)
        np.clip(self.pose[:, :2], DUCK_R, self.world.size - DUCK_R, out=self.pose[:, :2])
        self._keep_apart()
        if self.held is not None and self.held[0] == "duck" and self.world.hand is not None:
            self.pose[self.held[1], :2] = self.world.hand  # its legs may go, and it goes nowhere
        self.world.push_out(self.pose[:, :2], DUCK_R)
        self.velocity = (self.pose[:, :2] - before) / DT
        self.world.roll_balls(DT, self.pose[:, :2], self.velocity)
        if self.world.hand is not None and self.t > self.hand_until:
            self.world.hand = None  # the hand goes away again; it used to stay where it was last put for good
            self.held = None  # and whatever it held is where it was left
        self._act()
        self.pose[:, 2] = (h + np.pi) % (2 * np.pi) - np.pi
        self.world.step(self.t)
        self.t += DT
        if self.fruit_every_s and int(self.t / self.fruit_every_s) > int((self.t - DT) / self.fruit_every_s):
            self.world.drop_fruit(self.fruit_rng)
        self.send_frames()

    def send_frames(self) -> None:
        xy, h = self.pose[:, :2], self.pose[:, 2]
        fwd = np.column_stack([np.cos(h), np.sin(h)])
        left = np.column_stack([-np.sin(h), np.cos(h)])
        base = xy + ANTENNA[0] * fwd
        w = self.world
        light = daylight(self.t)
        sense = {}
        for side, p in (("left", base + ANTENNA[1] * left), ("right", base - ANTENNA[1] * left)):
            sense[f"odor_{side}"] = w.odor_at(p)
            sense[f"danger_{side}"] = w.odor_at(p, w.danger_odor)
            sense[f"humidity_{side}"] = w.humidity_at(p)
            sense[f"temp_{side}"] = temperature_at(p, light, self.world.tree)
            sense[f"duck_{side}"] = np.array([duck_odor_at(p[i], xy, i) for i in range(len(xy))])
            each = np.exp(-np.linalg.norm(xy[None] - p[:, None], axis=-1) / DUCK_SMELL_M)  # [i, j]: duck j at i's antenna
            np.fill_diagonal(each, 0.0)
            sense[f"scent_{side}"] = np.pad(each, ((0, 0), (0, frames.MAX_DUCKS - len(xy))))[:, :frames.MAX_DUCKS]
            sense[f"music_{side}"] = music_at(p, w.music) * w.music_volume
        sense["touch_left"], sense["touch_right"], dish = contacts(xy, h, w.food, self.touch_m)
        sense["sugar"] = (dish >= 0).astype(float)
        edge = w.pond_distance(xy)
        sense["water"] = (np.abs(edge) <= SHORE_M).astype(float)
        sense["swimming"] = (edge < -SHORE_M).astype(float)
        sense["bumped"], sense["ate"], sense["drank"], sense["petted"], sense["scared"] = (
            x.astype(float) for x in (self.bumped, self.ate, self.drank, self.petted, self.scared))
        sense["lum"] = self.seen = self.sight(xy, h, light)  # kept for the viewer's possession view
        self.bumped[:] = self.ate[:] = self.drank[:] = self.petted[:] = self.scared[:] = False
        sense["light"] = np.full(len(xy), light)
        sense["hat"] = self.hats.astype(float)
        seen = np.zeros((2, len(xy)))  # the nearest ball, to the left eye and to the right
        near = np.zeros(len(xy), bool)
        for ball in w.balls:
            rel = ball[:2] - xy
            d = np.linalg.norm(rel, axis=1)
            ahead, to_left = (rel * fwd).sum(1), (rel * left).sum(1)
            size = np.where(ahead > -0.3 * d, 1 / (1 + d / BALL_SEEN_M), 0.0)  # nothing of it from behind
            seen[0] = np.maximum(seen[0], size * (to_left >= 0))
            seen[1] = np.maximum(seen[1], size * (to_left < 0))
            near |= (d < KICK_REACH_M) & (ahead > 0)
        sense["ball_left"], sense["ball_right"], sense["ball_near"] = seen[0], seen[1], near.astype(float)
        if w.drum is not None:  # the drum, as the ball is seen: which eye, how plain, and whether it is in reach
            rel = np.asarray(w.drum) - xy
            d_drum = np.linalg.norm(rel, axis=1)
            plain_drum = np.where((rel * fwd).sum(1) > -0.3 * d_drum, 1 / (1 + d_drum / BALL_SEEN_M), 0.0)
            on_left = (rel * left).sum(1) >= 0
            sense["drum_left"], sense["drum_right"] = plain_drum * on_left, plain_drum * ~on_left
            sense["drum_near"] = (d_drum < DRUM_REACH_M).astype(float)
        sense["drummed"] = self.drummed.astype(float)
        self.drummed[:] = False
        sense["kicked"], sense["show"] = self.kicked.astype(float), self.saw_show.astype(float)
        self.kicked[:] = self.saw_show[:] = False
        side = lambda rel: (rel * left).sum(1) >= 0  # is it to this duck's left
        rel = xy[None] - xy[:, None]  # [i, j]: duck j from duck i
        d = np.linalg.norm(rel, axis=-1) + np.eye(len(xy)) * 1e9
        nearest = d.argmin(1)
        with_one = d.min(1) < NEAR_M
        to_near = xy[nearest] - xy
        plain = np.where(with_one, 1 - d.min(1) / NEAR_M, 0.0)
        sense["near_id"] = np.where(with_one, nearest, -1).astype(float)
        sense["near_left"], sense["near_right"] = plain * side(to_near), plain * ~side(to_near)
        crying = self.crying_until > self.t
        cry = np.zeros((2, len(xy)))
        for j in np.flatnonzero(crying):  # the nearest cry in earshot, and which side it is on
            loud = np.where((d[:, j] < EARSHOT_M), 1 - d[:, j] / EARSHOT_M, 0.0)
            on_left = side(xy[j] - xy)
            cry[0], cry[1] = np.maximum(cry[0], loud * on_left), np.maximum(cry[1], loud * ~on_left)
        sense["cry_left"], sense["cry_right"] = cry
        # a duck that comes right up to a crying one has comforted it
        touch = d < self.touch_m
        for j in np.flatnonzero(crying & touch.any(0)):
            self.events["comforted_by"][j] = int(np.flatnonzero(touch[:, j])[0])
            self.crying_until[j] = 0.0
        hand = np.zeros((2, len(xy)))
        if w.hand is not None:
            to_hand = np.asarray(w.hand) - xy
            plain_hand = 1 / (1 + np.linalg.norm(to_hand, axis=1) / HAND_SEEN_M)
            hand = np.array([plain_hand * side(to_hand), plain_hand * ~side(to_hand)])
        sense["hand_left"], sense["hand_right"], sense["hand_fed"] = hand[0], hand[1], self.hand_fed.astype(float)
        sense["held"] = np.array([self.held == ("duck", i) for i in range(len(xy))], float)
        sense["thrown"] = self.thrown.astype(float)
        self.thrown[:] = False
        self.hand_fed[:] = False
        for name, values in {**self.events, **self.heard}.items():
            sense[name] = values.copy()
            values[:] = -1.0 if name in frames.IDS else 0.0
        lying = np.array([h[:2] for h in self.hat_items], float).reshape(-1, 2)
        sense["hat_near"] = (np.linalg.norm(xy[:, None] - lying[None], axis=-1) < HAT_REACH_M).any(axis=1).astype(float)
        sense["wind"], sense["wind_from"] = wind_on(h, w.wind)
        for i in range(len(xy)):
            self.udp.sendto(frames.pack(
                t=self.t, duck=i, x=xy[i, 0], y=xy[i, 1], heading=h[i], **{k: v[i] for k, v in sense.items()},
            ), (frames.HOST, self.frame_port + i))

    def sight(self, xy, h, light) -> np.ndarray:
        """(ducks, 2 eyes, 721) of what each duck sees: here, the garden drawn from above."""
        return retina.luminance(xy, h, self.world, light)

    def state(self) -> dict:
        w = self.world
        return {"t": self.t, "pose": self.pose.tolist(), "food": w.food.tolist(), "eaten": self.eaten,
                "music": w.music, "drum": w.drum, "balls": len(w.balls), "hats": len(self.hat_items)}

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
    ap.add_argument("--frame-port", type=int, default=frames.FRAME_PORT,
                    help="first UDP port to try for the ducks' senses; a free block at or after it is used")
    ap.add_argument("--godot", action="store_true", help="publish the world for the Godot garden (viewer/godot/)")
    ap.add_argument("--wander", action="store_true", help="random walk every 0.5 s, for watching the stub alone")
    ap.add_argument("--brain", action="store_true",
                    help="drive the ducks from here with the real brain, so the viewer has drives to show")
    ap.add_argument("--labels", default="Bully,Napper,Carefree,Chatty,Scaredy",
                    help="one personality per duck when --brain is on")
    ap.add_argument("--learns", action=argparse.BooleanOptionalAction, default=True,
                    help="ducks learn from what happens to them and it changes who they keep company with; "
                         "--no-learns for ducks that stay as they hatched")
    ap.add_argument("--save", default=os.path.expanduser("~/.cache/micro-garden/garden.npz"),
                    help="with --brain: the garden is loaded from here on launch, aged by however long you "
                         "were away, and saved here on exit")
    ap.add_argument("--fresh", action="store_true", help="ignore the save and hatch new ducks")
    ap.add_argument("--blind", action="store_true",
                    help="the Gate 5 blind test: deal --labels to the ducks in an order nobody is told, "
                         "ignore any save, and print who was who on exit")
    args = ap.parse_args()
    os.makedirs(args.sock_dir, exist_ok=True)
    args.frame_port = frames.free_port_base(args.frame_port, args.ducks)  # never the ports of a garden that is up
    stub = Stub(args.ducks, args.seed, args.sock_dir, frame_port=args.frame_port, **DEMO_GARDEN)
    print(f"sockets in {args.sock_dir}: {', '.join(stub.names)}, control")
    view = None
    if args.view:
        from viewer.debug2d import Viewer
        view = Viewer(stub.world)
    server = None
    if args.brain:
        from brain.data import load_connectome, named_sets
        from brain.personality import preset, stack
        from brain.server import BrainServer
        print("loading the connectome, which takes a moment ...")
        W, ann = load_connectome()
        rng_k = np.random.default_rng(args.seed)
        labels = (args.labels.split(",") * args.ducks)[:args.ducks]
        if args.blind:
            labels = list(np.random.default_rng().permutation(labels))  # unseeded on purpose
        bodies = [(os.path.join(args.sock_dir, f"{name}.sock"), args.frame_port + i)
                  for i, name in enumerate(stub.names)]
        server = BrainServer(W, ann, named_sets(ann), bodies, args.seed, learns=args.learns,
                             personality=stack([preset(x, rng_k) for x in labels]))
        print(f"driving {', '.join(sorted(labels) if args.blind else labels)}")
        if os.path.exists(args.save) and not args.fresh and not args.blind:
            from brain.save import load
            gap = load(args.save, server.body, server.plastic, stub)
            server.decoder.stink_affinity = server.body.k["stink_affinity"].copy()  # the saved ducks, not --labels
            print(f"welcome back: {gap / 60:.0f} minutes away")
    rng = np.random.default_rng(args.seed)

    def click(xy):
        if view.on_tree(xy):
            with stub.lock:
                stub.shake_tree()

    def key(name, xy):
        """The player's verbs, the same calls a client makes on control.sock."""
        duck = view.selected
        with stub.lock:
            if name == "p":
                stub._control_call("garden.pet", {"duck": duck})
            elif name == "c":
                stub._control_call("garden.scare", {})
            elif name == "f":
                stub._control_call("garden.hand", {"x": xy[0], "y": xy[1], "feed": 3})
            elif name == "m":
                stub._control_call("garden.music", {"x": xy[0], "y": xy[1], "on": int(stub.world.music is None)})
            elif name == "h":
                stub._control_call("garden.hat", {"duck": duck, "on": int(not stub.hats[duck])})

    world_out = None
    if args.godot:
        from viewer.snapshot import Snapshot
        world_out = Snapshot(args.blind)
    try:
        run_loop(args, stub, view, server, rng, click, key, world_out)
    except KeyboardInterrupt:
        pass
    finally:
        if args.blind and server is not None:
            print("who was who: " + ", ".join(f"{n} = {l}" for n, l in zip(stub.names, labels)))
        elif server is not None:
            from brain.save import save
            save(args.save, server.body, server.plastic, stub)
            print(f"saved to {args.save}")


WHEEL_VX, WHEEL_VYAW = 0.2, 1.2  # what W/S and A/D ask for while the player has the wheel


def take_the_wheel(stub, server, view) -> None:
    """Possession (ARCHITECTURE.md 2.6): Tab hands the selected duck's legs to the player and back. The
    brain goes on seeing, smelling and learning; only its motor output is muted."""
    if view is None or server is None:
        return
    server.possessed[:] = False
    if view.possessing:
        i = view.selected
        server.possessed[i] = True
        fwd, turn = view.wasd()
        with stub.lock:
            stub._move(i, {"vx": WHEEL_VX * fwd, "vy": 0.0, "vyaw": WHEEL_VYAW * turn})


def run_loop(args, stub, view, server, rng, click, key, world_out=None) -> None:
    next_t, ticks = time.monotonic(), 0
    while view is None or view.alive(on_click=click, on_key=key):
        with stub.lock:
            if args.wander and ticks % 25 == 0:
                stub.cmd = np.column_stack([rng.uniform(-0.1, 0.3, args.ducks), np.zeros(args.ducks),
                                            rng.uniform(-1.5, 1.5, args.ducks)])
            ticks += 1
            stub.step()
        if server is not None:
            server.step(lockstep=False)  # outside the lock: it talks to the stub over its sockets
        take_the_wheel(stub, server, view)
        with stub.lock:
            if view:
                view.draw(stub, server)
            if world_out:
                world_out.step(stub, server)
        next_t += DT
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    main()
