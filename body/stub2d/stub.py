"""2D stub body: kinematic ducks on the garden plane behind the microduck contract (Gate 3).

One Unix socket per duck (duck-a.sock ...), like duck-sim, plus control.sock with stub-only
sim.step {n} and sim.state. Every body step sends each duck's sensory frame over UDP to
frame_port + duck. robot.do ground_pick on a dish eats it; robot.do headbutt pushes a touching duck in
front back by PUSH_M and knocks it over for DOWN_S; robot.do drink at the shore band takes a sip; past the
shore a duck swims at SWIM_SPEED. robot.sound is logged. With fruit_every_s set, the tree drops fruit on
that period.

Player actions on control.sock: garden.shake_tree drops SHAKE_FRUIT at once; garden.pet {duck} pets a duck;
garden.scare claps, startling every duck; garden.hand {x, y, feed} puts the hand in the garden, visible,
and drops that many bites at it; garden.music {x, y, on} places or removes the music; garden.hat {duck, on}
puts a hat on a duck or takes it off (a duck shakes a hat off by grooming).

In the viewer: click the tree to shake it; Tab takes and gives back the wheel of the selected duck (W A S D
drive it); P pets, C claps, F feeds by hand at the mouse, M toggles music there, H toggles a hat.

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

# The wind turns right round every 0.7 of a day, so each direction comes at every hour: a north wind carries
# food smells to a duck at the pond, a south wind carries the pond's damp air to ducks that are eating.
# A fruit every 10 s feeds five ducks; at 20 s some starved (Gate 9b). Laid out after the Chao gardens of
# Sonic Adventure 2. The rocks are the solid foot of a waterfall a viewer draws there, on the shore.
DEMO_GARDEN = dict(size=6.0, tree=(1.5, 4.3, 0.9), food_xy=((3.0, 2.2),), bites=10, danger_xy=((4.9, 1.3),),
                   pond=(4.5, 4.4, 1.2), rocks=((5.5, 5.47, 0.42), (5.95, 5.9, 0.55)), fruit_every_s=10.0, wind=(0.0, -1.0), wind_turns_s=0.7 * DAY_S,
                   personal_m=0.24,  # the drawn duck's width, so they do not merge
                   music=None)  # the player puts the music box down (M)

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
# The player's shake has its own limit, above the tree's MAX_FOOD, which the tree reaches by itself.
SHAKE_MOST = 12
PUSH_M = 0.15
DOWN_S = 10.0  # about how long a microduck takes to get back on its feet
BLOCKED_VYAW = 1.5  # rad/s at which a duck stopped by the fence or a rock turns back in
SWIM_SPEED = 0.5  # fraction of commanded speed while swimming
SOUND_TAGS = {"alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee"}  # microduck's voice bank
SIT_AFTER_S = 3.0  # still this long and a duck is drawn sitting (body/mujoco/adapter.py really sits)
DRUM_REACH_M = 0.32  # how near a drum a duck has to be to tap it
KICK_REACH_M, KICK_MS = 0.22, 1.6  # ball reach for a kick, and its speed after
COOL_SEEN_M = 3.0  # pond or shade this far past its edge is half as plain
BALL_SEEN_M = 1.5  # a ball this far off is half as plain as at the feet
NEAR_M = 1.0  # the nearest duck within this is the one a duck is with
EARSHOT_M = 2.0  # how far an alarm, a whoop or a cry carries
WITNESS_M = 1.5  # how near a duck must be to a shove or a hat taken to see it
GRAB_M = 0.3  # how near the hand must be to pick a thing up
THROW_MS, THROW_MAX_MS, THROW_S = 0.8, 3.0, 0.35  # release speed that throws, the cap, and flight time
HAND_S = 4.0  # how long the player's hand stays in the garden
HAND_SEEN_M = 1.5  # a hand this far off is half as plain as one at the beak
CRY_S = 6.0  # how long a cry lasts
AUDIENCE_M = 1.5  # how near a duck must be to a song or dance to be its audience
PERFORMANCES = ("sing", "dance", "singdance")
MAX_BALLS = 3
HAT_REACH_M = 0.25  # a duck can put on a hat on the ground within this
BITE_S = 0.5  # ground_pick takes this long, so at most one bite per BITE_S
# "headbutt", "drink", "preen" and "zoomies" are our skill names; this body acts out the first three and ignores
# zoomies. "emote_<feeling>" (brain/emotes.py) is recorded for the viewer. body/mujoco/adapter.py maps them to robot skills.

# How each feeling (brain/emotes.py) is acted out in every body, as explicit code: a voice tag and head poses, each
# (seconds after the last, neck_pitch, head_pitch, head_yaw, head_roll) in radians, pitch positive down. Works
# sitting or standing; every one ends level. The MuJoCo body forwards the poses and viewers draw the head as told.
# ponytail: posed by eye on the simulated duck; a real one's neck will want these re-posed.
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
    # the signature emotes (brain/emotes.py)
    "stomp": ("alarm", [(0.0, -0.2, -0.2, 0.0, 0.0), (0.2, 0.35, 0.3, 0.0, 0.0), (0.2, -0.2, -0.2, 0.0, 0.0), (0.2, 0.35, 0.3, 0.0, 0.0)]),
    "yawn": ("coo", [(0.0, -0.25, -0.5, 0.0, 0.1), (1.2, -0.3, -0.6, 0.0, 0.15), (0.8, 0.2, 0.3, 0.0, 0.0)]),
    "splash": ("wheee", [(0.0, 0.4, 0.5, 0.0, 0.0), (0.2, -0.2, -0.3, 0.0, 0.0), (0.2, 0.4, 0.5, 0.0, 0.0), (0.2, -0.2, -0.3, 0.0, 0.0)]),
    "sing": ("chirp", [(0.0, -0.25, -0.4, 0.35, 0.0), (0.35, -0.25, -0.4, -0.35, 0.0), (0.35, -0.25, -0.4, 0.35, 0.0),
                       (0.35, -0.25, -0.4, -0.35, 0.0)]),
    "cower": ("inquire", [(0.0, 0.45, 0.5, 0.0, 0.0), (0.3, 0.45, 0.5, 0.15, 0.0), (0.15, 0.45, 0.5, -0.15, 0.0),
                          (0.15, 0.45, 0.5, 0.15, 0.0), (0.15, 0.45, 0.5, -0.15, 0.0), (0.8, 0.45, 0.5, 0.0, 0.0)]),
    # to liked music: nod on the beat and swing side to side, silent
    "dance": (None, [(0.0, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0),
                     (0.35, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0)]),
    # both at once
    "singdance": ("chirp", [(0.0, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0),
                     (0.35, 0.25, 0.2, 0.4, 0.15), (0.35, -0.1, -0.2, 0.0, 0.0), (0.35, 0.25, 0.2, -0.4, -0.15), (0.35, -0.1, -0.2, 0.0, 0.0)]),
    # head down and shaking
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
        for x, y in balls:  # toys placed at start, for a gate
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
        self.hat_style = np.full(n, -1)  # a seed a viewer makes the hat from; -1 for none
        self.hat_items = []  # hats lying in the garden, as [x, y, style]
        self.donned = []  # (t, duck) each time one puts a hat on
        self.kicks = []  # (t, duck) each time one kicks a ball
        self.kicked = np.zeros(n, bool)
        self.drums = []  # (t, duck) each tap on the drum
        self.drummed = np.zeros(n, bool)
        self.saw_show = np.zeros(n, bool)
        # who did what to whom this step, for nearby ducks (-1 is nobody); cleared after sending
        self.events = {name: np.full(n, -1.0) for name in frames.IDS if name != "near_id"}
        self.heard = {"heard_alarm": np.zeros(n), "heard_joy": np.zeros(n)}
        self.hand_fed = np.zeros(n, bool)
        self.crying_until = np.zeros(n)  # garden time until which a duck is crying
        self.hand_until = 0.0
        self.held = None  # what the player's hand carries: (kind, index)
        self.thrown = np.zeros(n, bool)
        self.throws = []  # (t, duck) each time the player throws one
        self.given = []  # (t, duck) each time the player hands one a fruit
        self.velocity = np.zeros((n, 2))  # m/s over the ground, for what a duck walks into
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
        # How near two ducks' centres can come; 0 lets them pass through each other, as in the gates.
        # Touching is a little beyond it, so shoves and company still reach.
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
            self.cmd[i] = 0  # on the floor: its legs ignore the brain
        elif not self.relaxed[i]:
            self.cmd[i] = np.clip([float(p["vx"]), float(p["vy"]), float(p["vyaw"])],
                                  [-MAX_V, -MAX_VY, -MAX_VYAW], [MAX_V, MAX_VY, MAX_VYAW])

    def _head(self, i: int, p: dict) -> None:
        self.head[i] = [float(p[k]) for k in ROBOT_PARAMS["robot.head"]]

    def _do(self, i: int, p: dict) -> None:
        """Other skill names are accepted and do nothing on the 2D stub."""
        skill = p["skill"]
        if skill.startswith("emote_"):
            self._emote(i, skill[len("emote_"):])
            return
        do = {"ground_pick": self._pick, "headbutt": self._headbutt, "drink": self._drink, "kick": self._kick,
              "drum": self._tap_drum, "wear": self._wear, "preen": self._shed}.get(skill)
        if do is not None:
            do(i)

    def _audience(self, i: int) -> None:
        """Duck i is performing; the ducks near it see it (brain/social.py decides the response)."""
        near = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < AUDIENCE_M
        near[i] = False
        self.saw_show |= near
        self.events["show_by"][near] = i

    def _kick(self, i: int) -> None:
        fwd = np.array([np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])])
        for ball in self.world.balls:
            rel = ball[:2] - self.pose[i, :2]
            if np.linalg.norm(rel) < KICK_REACH_M and rel @ fwd > 0:
                ball[2:] += KICK_MS * fwd
                self.kicks.append((self.t, i))
                self.kicked[i] = True
                return

    def _tap_drum(self, i: int) -> None:
        if self.world.drum is None or np.linalg.norm(np.asarray(self.world.drum) - self.pose[i, :2]) >= DRUM_REACH_M:
            return
        self.drums.append((self.t, i))
        self.drummed[i] = True
        self.sounds.append((self.t, i, "drum"))  # the drum's sound, not the duck's
        self._audience(i)  # a performance

    def _wear(self, i: int) -> None:
        near = [k for k, (x, y, _) in enumerate(self.hat_items) if np.hypot(x - self.pose[i, 0], y - self.pose[i, 1]) < HAT_REACH_M]
        if self.hats[i] or not near:
            return
        self.hats[i], self.hat_style[i] = True, self.hat_items.pop(near[0])[2]
        self.donned.append((self.t, i))
        saw = np.linalg.norm(self.pose[:, :2] - self.pose[i, :2], axis=1) < WITNESS_M
        saw[i] = False
        self.events["hat_taken_by"][saw] = i

    def _shed(self, i: int) -> None:
        if not self.hats[i]:
            return
        self.hats[i] = False  # shaken off; it lands where the duck stands
        self.hat_items.append([float(self.pose[i, 0]), float(self.pose[i, 1]), int(max(self.hat_style[i], 0))])
        self.hat_style[i] = -1
        self.preened.append((self.t, i))

    def _emote(self, i: int, feeling: str) -> None:
        """Act a feeling out: say it, and queue the head poses for step() to play."""
        if self.acting[i] or self.relaxed[i] or self.t < self.down_until[i]:
            return  # one at a time, not in its sleep, and not from flat on the floor
        tag, poses = EMOTE_ACTS[feeling]
        self.emotes.append((self.t, i, feeling))
        if feeling == "cry":
            self.crying_until[i] = self.t + CRY_S
        if feeling in PERFORMANCES:  # and it has an audience
            self._audience(i)
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
        if p["tag"] in ("alarm", "wheee"):  # both carry, and both are catching
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
        """Per duck, joint state for a viewer: empty here; the MuJoCo body gives joints, height and lean."""
        return [{} for _ in self.names]

    def posture(self) -> list[str]:
        """Per duck, "up", "sat" or "down", for a viewer. Here "sat" is only drawn: the duck can walk at
        once, so no gate measurement changes."""
        return ["down" if self.t < until else "sat" if still > SIT_AFTER_S else "up"
                for until, still in zip(self.down_until, self.still_for)]

    def down_left(self) -> np.ndarray:
        """Seconds until each downed duck is up again, 0 if up; a viewer fits the fall and rise inside it."""
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
        """A kick that lands puts a duck on the floor: here it cannot move for DOWN_S."""
        self.down_until[j] = self.t + DOWN_S
        self.cmd[j] = 0

    def shake_tree(self) -> int:
        """Callers hold self.lock (control calls do; the viewer takes it)."""
        return self.world.drop_fruit(self.fruit_rng, SHAKE_FRUIT, SHAKE_MOST)

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
        """A ball put down where the player says, at most MAX_BALLS."""
        self.world.balls = np.vstack([self.world.balls, [float(p["x"]), float(p["y"]), 0.0, 0.0]])[-MAX_BALLS:]

    def _drop_hat(self, p):
        """A hat left in the garden, no two alike."""
        self.hat_items.append([float(p["x"]), float(p["y"]), int(self.hat_rng.integers(1_000_000))])

    def _scare(self, p):
        self.scared[:] = True  # a clap: everything in the garden hears it

    def _keep_apart(self) -> None:
        """Two ducks nearer than personal_m each move half the difference apart; a few passes settle a huddle."""
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
        """Everything the hand could pick up, as (kind, which, x, y). Things before ducks."""
        w = self.world
        things = [("ball", k, *b[:2]) for k, b in enumerate(w.balls)] + [("hat", k, h[0], h[1]) for k, h in enumerate(self.hat_items)]
        things += [("food", k, *xy) for k, xy in enumerate(w.food)] + ([("music", 0, *w.music)] if w.music else [])
        things += [("drum", 0, *w.drum)] if w.drum else []
        return things + [("duck", i, *xy) for i, xy in enumerate(self.pose[:, :2]) if self.can_carry_ducks]

    can_carry_ducks = True  # false for a simulated robot (body/mujoco/adapter.py)

    def _grab(self, p):
        """The hand closes on the nearest thing within GRAB_M, if any."""
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
        """The hand opens. Gently, the thing is put down; on the move, it is thrown: a ball rolls off, anything
        else lands a little way on, and a thrown duck is knocked down."""
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
        free = self.pose[:, :2].copy()
        np.clip(self.pose[:, :2], DUCK_R, self.world.size - DUCK_R, out=self.pose[:, :2])
        blocked = np.linalg.norm(self.pose[:, :2] - free, axis=1) > 1e-6
        self._keep_apart()
        if self.held is not None and self.held[0] == "duck" and self.world.hand is not None:
            self.pose[self.held[1], :2] = self.world.hand  # held: it goes nowhere
        free = self.pose[:, :2].copy()
        self.world.push_out(self.pose[:, :2], DUCK_R)
        blocked |= np.linalg.norm(self.pose[:, :2] - free, axis=1) > 1e-6
        # explicit code: a duck stopped by the fence or a rock turns toward the middle, like the robot's `fence`
        if blocked.any():
            ahead = np.column_stack([np.cos(h), np.sin(h)])
            to_mid = self.world.size / 2 - self.pose[:, :2]
            h += blocked * np.where(ahead[:, 0] * to_mid[:, 1] - ahead[:, 1] * to_mid[:, 0] >= 0, 1.0, -1.0) * BLOCKED_VYAW * DT
        self.velocity = (self.pose[:, :2] - before) / DT
        self.world.roll_balls(DT, self.pose[:, :2], self.velocity)
        if self.world.hand is not None and self.t > self.hand_until:
            self.world.hand = None  # the hand goes away
            self.held = None  # and drops what it held
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
        if w.drum is not None:  # seen like the ball: which eye, how plain, in reach
            rel = np.asarray(w.drum) - xy
            d_drum = np.linalg.norm(rel, axis=1)
            plain_drum = np.where((rel * fwd).sum(1) > -0.3 * d_drum, 1 / (1 + d_drum / BALL_SEEN_M), 0.0)
            on_left = (rel * left).sum(1) >= 0
            sense["drum_left"], sense["drum_right"] = plain_drum * on_left, plain_drum * ~on_left
            sense["drum_near"] = (d_drum < DRUM_REACH_M).astype(float)
        # Where a hot duck can cool off, by sight: which eye, and plainer the nearer. Explicit code, since
        # temperature and damp air give no direction from out in the sun.
        for name, place in (("pond", w.pond), ("shade", w.tree)):
            if place is not None:
                rel = np.asarray(place[:2]) - xy
                plain = 1 / (1 + np.maximum(np.linalg.norm(rel, axis=1) - place[2], 0) / COOL_SEEN_M)
                on_left = (rel * left).sum(1) >= 0
                sense[f"{name}_left"], sense[f"{name}_right"] = plain * on_left, plain * ~on_left
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
        for j in np.flatnonzero(crying):  # the nearest cry in earshot, and its side
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
    ap.add_argument("--godot", action="store_true", help="open the Godot garden (viewer/godot/) on this one; closing it ends the garden")
    ap.add_argument("--no-window", action="store_true", help="with --godot: publish only, and open Godot yourself")
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
    args.frame_port = frames.free_port_base(args.frame_port, args.ducks)  # skip ports another garden holds
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
        world_out = Snapshot(args.blind, window=not args.no_window)
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
    """Possession: Tab hands the selected duck's legs to the player and back. Only
    the brain's motor output is muted."""
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
