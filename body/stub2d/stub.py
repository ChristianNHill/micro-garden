"""2D stub body: kinematic ducks on the garden plane behind the microduck contract (PLAN.md Gate 3).

One Unix socket per duck (duck-a.sock ...), like duck-sim, plus control.sock with stub-only
sim.step {n} and sim.state. Every body step sends each duck's sensory frame over UDP to
frame_port + duck. robot.do ground_pick on a dish eats it; robot.do headbutt pushes a touching duck
in front of the attacker back by PUSH_M. robot.do drink at the pond's shore band takes a sip; past the
shore a duck swims at SWIM_SPEED. robot.sound is logged. With fruit_every_s set, the shade tree drops fruit on that
period; garden.shake_tree on control.sock (a player action) drops SHAKE_FRUIT at once. In the viewer,
click the tree; P pets the selected duck, C claps, F feeds by hand at the mouse, M starts or stops
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
from world.fields import (DAY_S, SHORE_M, SIZE_M, DUCK_R, World, contacts, daylight, duck_odor_at,
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
DEMO_GARDEN = dict(food_xy=((3.0, 3.0),), bites=10, danger_xy=((2.3, 1.7),), pond=(3.1, 0.9, 0.35), fruit_every_s=10.0,
                   wind=(0.0, -1.0), wind_turns_s=0.7 * DAY_S,
                   music=(0.9, 0.9))  # the corner nothing else is in: food north, pond south-east, stink in the middle

DT = 0.02
# ponytail: guessed limits standing in for robotd's clamps; replace with the sim's real ones at Gate 10
MAX_V, MAX_VY, MAX_VYAW = 0.3, 0.15, 2.0
ANTENNA = np.array([0.06, 0.05])  # forward, lateral offset of each odor sample, metres
CONTROL_PARAMS = {"sim.step": {"n": 1}, "sim.state": {}, "garden.shake_tree": {}, "garden.pet": {"duck": 0},
                  "garden.scare": {}, "garden.hand": {"x": 0.0, "y": 0.0, "feed": 0},
                  "garden.music": {"x": 0.0, "y": 0.0, "on": 1}, "garden.hat": {"duck": 0, "on": 1}}
SHAKE_FRUIT = 2
PUSH_M = 0.15
SWIM_SPEED = 0.5  # fraction of commanded speed while swimming
SOUND_TAGS = {"alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee"}  # microduck's voice bank
BITE_S = 0.5  # ground_pick takes this long, so at most one bite per BITE_S
# ponytail: "headbutt" is a stub-only skill name; map it to microduck's real kick skill at Gate 12


class Stub:
    def __init__(self, n: int, seed: int, sock_dir: str, food_xy=((3.0, 3.0), (1.0, 1.0)), danger_xy=(),
                 pond=None, bites=1, fruit_every_s=None, frame_port: int = frames.FRAME_PORT, pose=None,
                 wind=None, wind_turns_s=None, music=None):
        rng = np.random.default_rng(seed)
        self.fruit_rng = np.random.default_rng(seed + 1)
        self.fruit_every_s = fruit_every_s
        self.world = World(food_xy, danger_xy, pond, bites, wind, wind_turns_s, music)
        self.pose = np.column_stack([rng.uniform(0.5, SIZE_M - 0.5, (n, 2)), rng.uniform(-np.pi, np.pi, n)])
        if pose is not None:
            self.pose = np.array(pose, float).reshape(n, 3)
        self.frame_port = frame_port
        self.eaten = []  # (t, duck)
        self.headbutts = []  # (t, attacker, victim)
        self.bumped = np.zeros(n, bool)
        self.petted = np.zeros(n, bool)
        self.scared = np.zeros(n, bool)
        self.hats = np.zeros(n, bool)
        self.preened = []  # (t, duck) each time one is shaken off
        self.pets = []  # (t, duck)
        self.ate = np.zeros(n, bool)
        self.drank = np.zeros(n, bool)
        self.sounds = []  # (t, duck, tag)
        self.last_bite = np.full(n, -np.inf)
        self.cmd = np.zeros((n, 3))
        self.head = np.zeros((n, 4))
        self.relaxed = np.zeros(n, bool)
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
        if not self.relaxed[i]:
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
        elif p["skill"] == "preen" and self.hats[i]:
            self.hats[i] = False  # shaken off
            self.preened.append((self.t, i))

    def _sound(self, i: int, p: dict) -> None:
        if p["tag"] not in SOUND_TAGS:
            raise ValueError(f"unknown sound tag {p['tag']!r}; known: {sorted(SOUND_TAGS)}")
        self.sounds.append((self.t, i, p["tag"]))

    def _stop(self, i: int, p: dict) -> None:
        self.cmd[i] = 0

    def _relax(self, i: int, p: dict) -> None:
        self.relaxed[i], self.cmd[i] = True, 0

    def _init(self, i: int, p: dict) -> None:
        self.relaxed[i] = False

    def _pick(self, i: int) -> None:
        *_, dish = contacts(self.pose[:, :2], self.pose[:, 2], self.world.food)
        if dish[i] >= 0 and self.t - self.last_bite[i] >= BITE_S:
            self.world.eat(dish[i])
            self.eaten.append((self.t, i))
            self.last_bite[i], self.ate[i] = self.t, True

    def _drink(self, i: int) -> None:
        shore = abs(self.world.pond_distance(self.pose[i, :2])) <= SHORE_M
        if shore and self.t - self.last_bite[i] >= BITE_S:
            self.last_bite[i], self.drank[i] = self.t, True

    def _swimming(self) -> np.ndarray:
        return self.world.pond_distance(self.pose[:, :2]) < -SHORE_M

    def _headbutt(self, i: int) -> None:
        fwd = np.array([np.cos(self.pose[i, 2]), np.sin(self.pose[i, 2])])
        rel = self.pose[:, :2] - self.pose[i, :2]
        hit = (np.linalg.norm(rel, axis=1) < 2 * DUCK_R) & (rel @ fwd > 0)
        hit[i] = False
        for j in np.flatnonzero(hit):
            self.pose[j, :2] = np.clip(self.pose[j, :2] + PUSH_M * fwd, DUCK_R, SIZE_M - DUCK_R)
            self.bumped[j] = True
            self.headbutts.append((self.t, i, j))

    def shake_tree(self) -> int:
        """Callers hold self.lock (control calls do; the viewer takes it)."""
        return self.world.drop_fruit(self.fruit_rng, SHAKE_FRUIT)

    def _control_call(self, method, p):
        if method == "garden.pet":
            i = int(p["duck"])
            self.petted[i] = True
            self.pets.append((self.t, i))
            return {}
        if method == "garden.music":
            self.world.music = (float(p["x"]), float(p["y"])) if int(p["on"]) else None
            return {}
        if method == "garden.hat":
            self.hats[int(p["duck"])] = bool(int(p["on"]))
            return {}
        if method == "garden.scare":
            self.scared[:] = True  # a clap: everything in the garden hears it
            return {}
        if method == "garden.hand":
            self.world.hand = (float(p["x"]), float(p["y"]))
            if p.get("feed"):
                self.world.food = np.vstack([self.world.food, self.world.hand])
                self.world.bites = np.append(self.world.bites, int(p["feed"]))
            return {}
        if method == "garden.shake_tree":
            return {"fell": self.shake_tree()}
        if method == "sim.step":
            for _ in range(int(p["n"])):
                self.step()
            if int(p["n"]) == 0:
                self.send_frames()  # lets a lockstep client read the starting state
            return {"t": self.t, "eaten": len(self.eaten)}
        return self.state()

    def step(self) -> None:
        x, y, h = self.pose.T
        vx, vy, vyaw = (self.cmd * np.where(self._swimming(), SWIM_SPEED, 1.0)[:, None]).T
        h += vyaw * DT
        x += (vx * np.cos(h) - vy * np.sin(h)) * DT
        y += (vx * np.sin(h) + vy * np.cos(h)) * DT
        np.clip(self.pose[:, :2], DUCK_R, SIZE_M - DUCK_R, out=self.pose[:, :2])
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
            sense[f"temp_{side}"] = temperature_at(p, light)
            sense[f"duck_{side}"] = np.array([duck_odor_at(p[i], xy, i) for i in range(len(xy))])
            sense[f"music_{side}"] = music_at(p, w.music)
        sense["touch_left"], sense["touch_right"], dish = contacts(xy, h, w.food)
        sense["sugar"] = (dish >= 0).astype(float)
        edge = w.pond_distance(xy)
        sense["water"] = (np.abs(edge) <= SHORE_M).astype(float)
        sense["swimming"] = (edge < -SHORE_M).astype(float)
        sense["bumped"], sense["ate"], sense["drank"], sense["petted"], sense["scared"] = (
            x.astype(float) for x in (self.bumped, self.ate, self.drank, self.petted, self.scared))
        sense["lum"] = retina.luminance(xy, h, w, light)
        self.bumped[:] = self.ate[:] = self.drank[:] = self.petted[:] = self.scared[:] = False
        sense["light"] = np.full(len(xy), light)
        sense["hat"] = self.hats.astype(float)
        sense["wind"], sense["wind_from"] = wind_on(h, w.wind)
        for i in range(len(xy)):
            self.udp.sendto(frames.pack(
                t=self.t, duck=i, x=xy[i, 0], y=xy[i, 1], heading=h[i], **{k: v[i] for k, v in sense.items()},
            ), (frames.HOST, self.frame_port + i))

    def state(self) -> dict:
        return {"t": self.t, "pose": self.pose.tolist(), "food": self.world.food.tolist(), "eaten": self.eaten}

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
    stub = Stub(args.ducks, args.seed, args.sock_dir, **DEMO_GARDEN)
    print(f"sockets in {args.sock_dir}: {', '.join(stub.names)}, control")
    view = None
    if args.view:
        from viewer.debug2d import Viewer
        view = Viewer()
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
        bodies = [(os.path.join(args.sock_dir, f"{name}.sock"), frames.FRAME_PORT + i)
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

    try:
        run_loop(args, stub, view, server, rng, click, key)
    except KeyboardInterrupt:
        pass
    finally:
        if args.blind and server is not None:
            print("who was who: " + ", ".join(f"{n} = {l}" for n, l in zip(stub.names, labels)))
        elif server is not None:
            from brain.save import save
            save(args.save, server.body, server.plastic, stub)
            print(f"saved to {args.save}")


def run_loop(args, stub, view, server, rng, click, key) -> None:
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
        with stub.lock:
            if view:
                view.draw(stub, server)
        next_t += DT
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    main()
