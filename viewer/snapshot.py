"""The world snapshot: what the garden looks like this step, published for a viewer in another process.

One JSON datagram a step goes to 127.0.0.1:SNAPSHOT_PORT for the Godot garden (viewer/godot/) to draw. The
player's actions come back on ACTION_PORT as the garden's control calls (`garden.*` only). UDP on loopback
costs nothing with no viewer, and a late viewer picks up at the next step. Both bodies publish the
same, since the MuJoCo body is a Stub.

`garden.wheel` is the viewer's own call: the player riding a duck mutes that duck's motor output, as Tab
does in the 2D window, and the wheel is let go if the viewer stops sending it (quit or crashed).

Knobs go along so a viewer can shape each duck; they and the labels are withheld in a blind run, since a
silhouette would give the personality away.
"""
import atexit
import base64
import json
import os
import shutil
import socket
import subprocess

import numpy as np

from body.stub2d.retina import HEX_AZ, HEX_EL
from body.stub2d.stub import WHEEL_VX, WHEEL_VYAW
from brain.brainview import POINTS
from brain import social
from brain.emotes import phrase
from brain.personality import label_of
from brain.physiology import pressing
from world.fields import DAY_S, daylight

# Below the sensory frames (7700 and up). MICRO_GARDEN_PORT moves both, for a second garden (Godot's --port).
SNAPSHOT_PORT = int(os.environ.get("MICRO_GARDEN_PORT", 7650))
ACTION_PORT = SNAPSHOT_PORT + 1
MOODS = ("fear", "anger", "joy", "sorrow")
MOOD_AT = 0.25  # under this a duck is content, and the viewer's mood shape is hidden anyway
SHAPE_KNOBS = ("appetite", "aggressiveness", "timidity", "vanity", "chattiness", "energy", "sleepiness")
EMOTE_S = 3.0  # how long an emote stays in the snapshot
EATING_S = 1.0
KICKING_S = 0.4  # how long a kick shows
HEARD_S = 1.0  # quacks this recent are sent
WHEEL_S = 0.5  # a wheel not sent for this long is let go
TOASTS = 6
DN_NAMES = ("forward", "back", "steer L", "steer R", "giant fiber", "feed")  # decoder.rates, in its order
_b64 = lambda a: base64.b64encode(np.asarray(a).tobytes()).decode()
# where each of an eye's 721 columns looks, as signed bytes across the eye's field
HEX = _b64(np.round(np.concatenate([HEX_AZ, HEX_EL]) / np.abs(HEX_AZ).max() * 127).astype(np.int8))
TOAST_FORMATS = (("eaten", "{who} ate"), ("headbutts", "{who} shoved {other}"), ("pets", "{who} was petted"),
                 ("emotes", "{who} {other}"), ("kicks", "{who} kicked the ball"), ("drums", "{who} played the {other}"), ("given", "{who} was handed a fruit"), ("throws", "{who} was thrown"), ("donned", "{who} put a hat on"), ("preened", "{who} shook its hat off"), ("fails", "{who} {other}"))


FRUIT_NAMES = ("oranges", "apples", "bananas", "pears", "cherries", "grapes", "strawberries", "lemons", "plums", "peaches")


def among(body, i: int, names: list[str]) -> dict:
    """What duck i is to the others and to the player, in words, for its card."""
    friend, grudge = social.friends(body, i, names)
    trust = float(body.hand_trust[i])
    return {"friend": friend, "grudge": grudge, "favourite": FRUIT_NAMES[int(body.favourite[i])],
            "hand": "trusts you" if trust > 0.3 else "is wary of you" if trust < -0.2 else "is still making up its mind about you",
            "skills": [round(float(v[i]), 2) for v in (body.swim_skill, body.walk_skill, body.dance_skill, body.eat_skill, body.fight_skill, body.fashion_skill, body.music_skill)]}


def readout(body, i: int) -> list:
    """Duck i's needs, moods and wants as [section, name, 0 to 1] rows, for a viewer's bars.
    A want is a like scaled down as hunger or thirst presses."""
    hot, cold = (float(v[i]) for v in body.discomfort())
    k = lambda name: float(body.k[name][i])
    free = 1.0 - float(pressing(np.maximum(body.hunger, body.thirst))[i])
    rows = [("needs", "hungry", body.hunger[i]), ("needs", "thirsty", body.thirst[i]), ("needs", "sleepy", body.sleep_pressure[i]),
            ("needs", "tired", body.fatigue[i]), ("needs", "bored", body.boredom[i]), ("needs", "too hot", hot), ("needs", "too cold", cold),
            ("moods", "joy", body.joy[i]), ("moods", "fear", body.fear[i]), ("moods", "anger", body.anger[i]), ("moods", "sorrow", body.sorrow[i]),
            ("wants", "a swim", k("water_love") * (1 + hot) / 2 * free), ("wants", "company", k("sociability") * free),
            ("wants", "music", k("music_affinity") * free), ("wants", "a hat", k("vanity")), ("wants", "to play", k("playfulness") * free),
            ("wants", "the stink", k("stink_affinity"))]  # what it does when it meets one, not a need
    return [[section, name, round(float(np.clip(v, 0, 1)), 2)] for section, name, v in rows]


GODOT = os.environ.get("GODOT") or shutil.which("godot") or "/Applications/Godot.app/Contents/MacOS/Godot"
PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "godot")


class Snapshot:
    def __init__(self, blind: bool = False, window: bool = False):
        """window=True opens the Godot window too; closing it ends (and saves) the garden."""
        self.window = None
        if window and os.path.exists(GODOT):
            self.window = subprocess.Popen([GODOT, "--path", PROJECT, "--", f"--port={SNAPSHOT_PORT}"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(self.window.terminate)
        elif window:
            print(f"no Godot at {GODOT}: install Godot 4 or set GODOT, or open viewer/godot yourself")
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.actions = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.actions.bind(("127.0.0.1", ACTION_PORT))
        self.actions.setblocking(False)
        self.blind = blind
        self.seen = {key: 0 for key, _ in TOAST_FORMATS}
        self.toasts = []
        self.refused = set()  # viewer calls that failed, each reported once
        self.riding = -1  # the duck whose motor output is muted
        self.watched = -1  # the viewer's selected duck, whose readout and brain are sent
        self.wheel = (-1, 0.0, 0.0, -np.inf)  # duck, forward, turn, garden time last sent

    def close(self) -> None:
        self.out.close()
        self.actions.close()

    def _gather_toasts(self, stub) -> None:
        short = lambda who: stub.duck_names[who] if isinstance(who, (int, np.integer)) else who
        for key, fmt in TOAST_FORMATS:
            events = getattr(stub, key)
            for e in events[self.seen[key]:]:
                if key == "emotes" and e[2] in ("dance", "singdance"):
                    continue  # dances are visible, and would flood the news
                other = phrase(e[2]) if key == "emotes" else short(e[2]) if len(e) > 2 else ""
                line = fmt.format(who=short(e[1]), other=other)
                if line not in self.toasts[-3:]:  # no repeats of recent news
                    self.toasts.append(line)
            self.seen[key] = len(events)
        self.toasts = self.toasts[-TOASTS:]

    def build(self, stub, body=None) -> dict:
        """`body` is the brain's Physiology, or None when the garden runs without a brain."""
        self._gather_toasts(stub)
        n = len(stub.names)
        last = lambda events: {e[1]: e for e in events[-4 * n:]}  # each duck's latest
        recent = dict(emotes=last(stub.emotes), bites=last(stub.eaten), kicks=last(stub.kicks), taps=last(stub.drums),
                      posture=stub.posture(), joints=stub.articulation(), down_left=stub.down_left(), swimming=stub._swimming())
        ducks = [self._duck(stub, body, i, **recent) for i in range(n)]
        w = stub.world
        return {"t": round(stub.t, 2), "size": w.size, "light": round(float(daylight(stub.t)), 3),
                "day": round(stub.t % DAY_S / DAY_S, 4), "ducks": ducks,
                "food": [[round(float(x), 3), round(float(y), 3), int(k)] for (x, y), k in zip(w.food, w.kinds)],  # x, y, fruit kind
                "danger": [[float(x), float(y)] for x, y in w.danger],
                "pond": None if w.pond is None else [float(v) for v in w.pond], "tree": list(w.tree), "rocks": w.rocks.round(3).tolist(),
                "wind": None if w.wind is None else [round(float(v), 3) for v in w.wind],  # m/s
                "hats": [[round(x, 3), round(y, 3), k] for x, y, k in stub.hat_items],
                "balls": [[round(float(x), 3), round(float(y), 3), style] for (x, y), style in zip(w.balls[:, :2], stub.ball_styles)],
                "drum": None if w.drum is None else list(w.drum),
                "instruments": [[round(x, 3), round(y, 3), int(k)] for x, y, k in w.instruments],
                "held": list(stub.held) if stub.held else None,  # [kind, index]
                "music": None if w.music is None else list(w.music), "music_volume": round(float(w.music_volume), 2),
                "hand": None if w.hand is None else [float(v) for v in w.hand], "toasts": self.toasts,
                "sounds": [[round(t, 2), int(i), tag] for t, i, tag in stub.sounds[-2 * n:] if stub.t - t < HEARD_S]}

    def _duck(self, stub, body, i, emotes, bites, kicks, taps, posture, joints, down_left, swimming) -> dict:
        """One duck's part of the snapshot: the body's state, plus the brain's when there is one."""
        emote = emotes.get(i)
        showing = emote is not None and stub.t - emote[0] < EMOTE_S
        duck = {
            "name": stub.duck_names[i],
            "x": round(float(stub.pose[i, 0]), 3), "y": round(float(stub.pose[i, 1]), 3),
            "h": round(float(stub.pose[i, 2]), 3),
            "sat": posture[i] == "sat", "down": posture[i] == "down", "down_left": round(float(down_left[i]), 2),
            "swimming": bool(swimming[i]), "hat": bool(stub.hats[i]), "hat_style": int(max(stub.hat_style[i], 0)),
            "head": [round(float(v), 3) for v in stub.head[i]],  # neck_pitch, head_pitch, head_yaw, head_roll
            "eating": i in bites and stub.t - bites[i][0] < EATING_S,
            "kicking": i in kicks and stub.t - kicks[i][0] < KICKING_S,
            "drumming": i in taps and stub.t - taps[i][0] < 0.5,
            "emote": emote[2] if showing else "", "emote_t": round(emote[0], 2) if showing else -1.0,
            "crying": bool(stub.crying_until[i] > stub.t),
            "label": "", "asleep": False, "mood": "", "strength": 0.0, "hunger": 0.0, "thirst": 0.0, "sleepy": 0.0,
            "knobs": {}, "readout": [], "among": {},
            **joints[i],
        }
        if body is None:
            return duck
        mood, strength = max(((m, float(getattr(body, m)[i])) for m in MOODS), key=lambda x: x[1])
        if strength < MOOD_AT:  # otherwise a duck feeling nothing at all reads as the first mood on the list
            mood = "content"
        level = lambda name: round(float(getattr(body, name)[i]), 2)
        duck.update(asleep=bool(body.asleep[i]), mood=mood, strength=round(strength, 2), tune=round(float(body.music_skill[i]), 2),
                    hunger=level("hunger"), thirst=level("thirst"), sleepy=level("sleep_pressure"))
        if not self.blind:
            duck.update(label=label_of(body.k, i), knobs={k: round(float(body.k[k][i]), 2) for k in SHAPE_KNOBS})
        if i == self.watched:  # selected duck only: ~400 bytes
            duck.update(readout=readout(body, i), among=among(body, i, stub.duck_names))
        return duck

    def ride(self, stub, server, duck: int) -> dict:
        """The ride view's overlay: both hex retinas as bytes (left eye first) and the six descending readouts in Hz."""
        rates = server.decoder.rates
        return {"duck": duck, "hex": HEX, "lum": _b64(np.round(np.clip(stub.seen[duck], 0, 1) * 255).astype(np.uint8)),
                "dn": [[name, round(float(hz), 1)] for name, hz in zip(DN_NAMES, rates[duck])]}

    def step(self, stub, server=None) -> None:
        """Publish this step and do whatever the player asked. Call with the stub's lock held. `server` is
        the BrainServer, or None when the garden runs without a brain."""
        if self.window is not None and self.window.poll() is not None:
            raise KeyboardInterrupt  # window closed: end and save as on Ctrl-C
        world = self.build(stub, server.body if server else None)
        if server is not None and server.decoder is not None and self.wheel[0] >= 0 and stub.seen is not None:
            world["ride"] = self.ride(stub, server, self.wheel[0])
        if server is not None and server.brainview is not None and server.brainview.duck >= 0:
            world["brain"] = {"duck": server.brainview.duck, "points": str(POINTS), "spikes": server.brainview.take()}
        self.out.sendto(json.dumps(world).encode(), ("127.0.0.1", SNAPSHOT_PORT))
        self._take_actions(stub, server)
        if server is not None:
            self._hold_wheel(stub, server)

    def _take_actions(self, stub, server) -> None:
        """Everything the viewer has sent since the last step."""
        while True:
            try:
                action = json.loads(self.actions.recv(4096))
            except BlockingIOError:
                return
            method, p = str(action.get("method", "")), action.get("params", {})
            if method == "garden.watch":  # the selected duck, or -1
                self.watched = int(p["duck"]) if 0 <= int(p["duck"]) < len(stub.names) else -1
                if server is not None and server.brainview is not None:
                    server.brainview.duck = self.watched
            elif method == "garden.wheel":
                self.wheel = (int(p["duck"]), float(np.clip(p["fwd"], -1, 1)), float(np.clip(p["turn"], -1, 1)), stub.t)
            elif method.startswith("garden."):  # player calls only
                try:
                    stub._control_call(method, p)
                except (KeyError, ValueError, TypeError, IndexError) as e:
                    # a mismatched viewer version must not stop the garden: report once and go on
                    if method not in self.refused:
                        self.refused.add(method)
                        print(f"ignoring {method} from the viewer: {type(e).__name__} {e}")

    def _hold_wheel(self, stub, server) -> None:
        """Give the ridden duck's legs to the player while the viewer keeps asking, and back when it stops."""
        duck, fwd, turn, asked = self.wheel
        held = 0 <= duck < len(stub.names) and stub.t - asked < WHEEL_S
        if self.riding >= 0 and (not held or duck != self.riding):
            server.possessed[self.riding] = False
        self.riding = duck if held else -1
        if held:
            server.possessed[duck] = True
            stub._move(duck, {"vx": WHEEL_VX * fwd, "vy": 0.0, "vyaw": WHEEL_VYAW * turn})
        else:
            self.wheel = (-1, 0.0, 0.0, -np.inf)

if __name__ == "__main__":
    import tempfile
    import time
    from body.stub2d.stub import DEMO_GARDEN, Stub
    with tempfile.TemporaryDirectory() as d:
        stub, snap = Stub(3, 0, d, **DEMO_GARDEN), Snapshot()
        stub.emotes.append((0.0, 1, "happy"))
        stub.step()
        rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rx.bind(("127.0.0.1", SNAPSHOT_PORT))
        rx.settimeout(1.0)
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(json.dumps({"method": "garden.hand", "params": {"x": 2.0, "y": 2.0, "feed": 3}}).encode(), ("127.0.0.1", ACTION_PORT))
        tx.sendto(json.dumps({"method": "sim.step", "params": {"n": 100}}).encode(), ("127.0.0.1", ACTION_PORT))
        dishes, t = len(stub.world.food), stub.t
        time.sleep(0.05)  # loopback delivery is not instant
        snap.step(stub)
        got = json.loads(rx.recv(65535))
        world_fields = {"t", "size", "light", "day", "ducks", "food", "danger", "pond", "tree", "rocks", "wind", "hats", "balls", "drum", "held", "music", "music_volume", "hand",
                        "toasts", "sounds"}
        assert world_fields <= set(got), f"the snapshot lost {world_fields - set(got)}: Godot reads every one of these"
        duck_fields = {"name", "label", "x", "y", "h", "asleep", "sat", "down", "swimming", "hat", "hat_style", "head", "eating", "kicking",
                       "mood", "strength", "emote", "emote_t", "knobs", "readout", "among", "crying"}
        assert duck_fields <= set(got["ducks"][0]), f"a duck lost {duck_fields - set(got['ducks'][0])}"
        assert len(got["ducks"]) == 3 and got["ducks"][1]["emote"] == "happy" and got["ducks"][0]["emote"] == ""
        assert got["toasts"] == [f"{stub.duck_names[1]} looks happy"], got["toasts"]
        assert len(stub.world.food) == dishes + 1, "a player's feed reaches the garden"
        assert stub.t == t, "only garden.* calls are taken from the network"
        tx.sendto(json.dumps({"method": "garden.no_such_thing", "params": {}}).encode(), ("127.0.0.1", ACTION_PORT))
        tx.sendto(json.dumps({"method": "garden.pet", "params": {"duck": 99}}).encode(), ("127.0.0.1", ACTION_PORT))
        time.sleep(0.05)
        snap.step(stub)  # neither stops the garden
        from types import SimpleNamespace
        server = SimpleNamespace(possessed=np.zeros(3, bool), body=None, decoder=None, brainview=None)
        tx.sendto(json.dumps({"method": "garden.wheel", "params": {"duck": 1, "fwd": 1, "turn": 0}}).encode(), ("127.0.0.1", ACTION_PORT))
        snap.step(stub, server)
        assert server.possessed[1] and stub.cmd[1, 0] == WHEEL_VX, "the rider has duck b's legs"
        stub.t += 2 * WHEEL_S
        snap.step(stub, server)
        assert not server.possessed.any(), "a wheel nobody holds is let go"
        tx.sendto(json.dumps({"method": "garden.wheel", "params": {"duck": 1, "fwd": 1, "turn": 0}}).encode(), ("127.0.0.1", ACTION_PORT))
        time.sleep(0.05); snap.step(stub, server)
        tx.sendto(json.dumps({"method": "garden.wheel", "params": {"duck": -1, "fwd": 0, "turn": 0}}).encode(), ("127.0.0.1", ACTION_PORT))
        time.sleep(0.05); snap.step(stub, server)
        assert not server.possessed.any(), "getting off gives the duck its legs back at once"
        print(f"ok  {len(json.dumps(got))} bytes a snapshot; the player fed the garden; sim.step was refused")
        for s in (rx, tx):
            s.close()
        snap.close()
