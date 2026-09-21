"""The world snapshot: what the garden looks like this step, published for a viewer in another process.

One JSON datagram a step to 127.0.0.1:SNAPSHOT_PORT, which the Godot garden (viewer/godot/) draws, and the
player's actions come back on ACTION_PORT as the garden's own control calls (`garden.*` only). UDP on the
loopback, like the sensory frames: a viewer that is not there costs nothing, and one that starts late picks
up at the next step. Any body publishes the same thing, since the MuJoCo body is a Stub.

`garden.wheel` is the one call that is the viewer's own: the player riding a duck from Godot, which mutes that
duck's motor output in the brain server exactly as Tab does in the 2D window, and lets go by itself if the
viewer stops asking (it quit, or crashed, with a duck still in hand).

Knobs go along so a viewer can draw who a duck is in its outline; labels too, unless the run is blind, when
both are withheld, as telling a Bully by its silhouette is no blind test.
"""
import base64
import json
import os
import socket

import numpy as np

from body.stub2d.retina import HEX_AZ, HEX_EL
from body.stub2d.stub import WHEEL_VX, WHEEL_VYAW
from brain.brainview import POINTS
from brain import social
from brain.emotes import phrase
from brain.personality import label_of
from brain.physiology import pressing
from world.fields import DAY_S, daylight

# Under the sensory frames, which take 7700 and up. MICRO_GARDEN_PORT moves the pair, for a second garden
# beside one that is being watched (Godot's --port).
SNAPSHOT_PORT = int(os.environ.get("MICRO_GARDEN_PORT", 7650))
ACTION_PORT = SNAPSHOT_PORT + 1
MOODS = ("fear", "anger", "joy", "sorrow")
SHAPE_KNOBS = ("appetite", "aggressiveness", "timidity", "vanity", "chattiness", "energy", "sleepiness")
EMOTE_S = 3.0  # how long an emote stays in the snapshot
EATING_S = 1.0
KICKING_S = 0.4  # how long a kick shows
HEARD_S = 1.0  # quacks this recent go along, for a viewer with a voice
WHEEL_S = 0.5  # a wheel nobody has touched for this long is let go
TOASTS = 6
DN_NAMES = ("forward", "back", "steer L", "steer R", "giant fiber", "feed")  # decoder.rates, in its order
_b64 = lambda a: base64.b64encode(np.asarray(a).tobytes()).decode()
# where each of an eye's 721 columns looks, as signed bytes across the eye's field: sent with the view
HEX = _b64(np.round(np.concatenate([HEX_AZ, HEX_EL]) / np.abs(HEX_AZ).max() * 127).astype(np.int8))
TOAST_FORMATS = (("eaten", "{who} ate"), ("headbutts", "{who} shoved {other}"), ("pets", "{who} was petted"),
                 ("emotes", "{who} {other}"), ("kicks", "{who} kicked the ball"), ("drums", "{who} plays the drum"), ("given", "{who} was handed a fruit"), ("throws", "{who} was thrown"), ("donned", "{who} put a hat on"), ("preened", "{who} shook its hat off"))


FRUIT_NAMES = ("oranges", "apples", "bananas")


def among(body, i: int, names: list[str]) -> dict:
    """What duck i is to the others and to the player, in words, for its card."""
    friend, grudge = social.friends(body, i, names)
    trust = float(body.hand_trust[i])
    return {"friend": friend, "grudge": grudge, "favourite": FRUIT_NAMES[int(body.favourite[i])],
            "hand": "trusts your hand" if trust > 0.3 else "is wary of your hand" if trust < -0.2 else "does not know your hand yet",
            "skills": [round(float(v[i]), 2) for v in (body.swim_skill, body.run_skill, body.dance_skill)]}


def readout(body, i: int) -> list:
    """Duck i's needs, moods and wants as [section, name, 0 to 1] rows, for a viewer's bars: what the body
    keeps, as it keeps it. A want is a like the duck is free to act on: it gives way as a need presses."""
    hot, cold = (float(v[i]) for v in body.discomfort())
    k = lambda name: float(body.k[name][i])
    free = 1.0 - float(pressing(np.maximum(body.hunger, body.thirst))[i])
    rows = [("needs", "hunger", body.hunger[i]), ("needs", "thirst", body.thirst[i]), ("needs", "sleep", body.sleep_pressure[i]),
            ("needs", "tired", body.fatigue[i]), ("needs", "bored", body.boredom[i]), ("needs", "too hot", hot), ("needs", "too cold", cold),
            ("moods", "joy", body.joy[i]), ("moods", "fear", body.fear[i]), ("moods", "anger", body.anger[i]), ("moods", "sorrow", body.sorrow[i]),
            ("wants", "a swim", k("water_love") * (1 + hot) / 2 * free), ("wants", "company", k("sociability") * free),
            ("wants", "music", k("music_affinity") * free), ("wants", "a hat", k("vanity")), ("wants", "to play", k("playfulness") * free)]
    return [[section, name, round(float(np.clip(v, 0, 1)), 2)] for section, name, v in rows]


class Snapshot:
    def __init__(self, blind: bool = False):
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.actions = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.actions.bind(("127.0.0.1", ACTION_PORT))
        self.actions.setblocking(False)
        self.blind = blind
        self.seen = {key: 0 for key, _ in TOAST_FORMATS}
        self.toasts = []
        self.refused = set()  # calls from a viewer that this garden could not take, each reported once
        self.riding = -1  # the duck whose motor output this has muted, to give it back
        self.watched = -1  # the duck the viewer has selected, whose readout and brain go along
        self.wheel = (-1, 0.0, 0.0, -np.inf)  # duck, forward, turn, and the garden time it was last asked for

    def close(self) -> None:
        self.out.close()
        self.actions.close()

    def _gather_toasts(self, stub) -> None:
        short = lambda who: stub.names[who].replace("duck-", "") if isinstance(who, (int, np.integer)) else who
        for key, fmt in TOAST_FORMATS:
            events = getattr(stub, key)
            for e in events[self.seen[key]:]:
                if key == "emotes" and e[2] in ("dance", "singdance"):
                    continue  # a dance is seen, and five ducks at it would be all the news there is
                other = phrase(e[2]) if key == "emotes" else short(e[2]) if len(e) > 2 else ""
                line = fmt.format(who=short(e[1]), other=other)
                if line not in self.toasts[-3:]:  # two ducks at a dish are two pieces of news, not twenty
                    self.toasts.append(line)
            self.seen[key] = len(events)
        self.toasts = self.toasts[-TOASTS:]

    def build(self, stub, body=None) -> dict:
        """`body` is the brain's Physiology, or None when the garden runs without a brain."""
        self._gather_toasts(stub)
        n = len(stub.names)
        last = lambda events: {e[1]: e for e in events[-4 * n:]}  # each duck's latest, from the recent few
        emotes, bites, kicks, taps = last(stub.emotes), last(stub.eaten), last(stub.kicks), last(stub.drums)
        w = stub.world
        posture, joints, down_left = stub.posture(), stub.articulation(), stub.down_left()
        swimming = stub._swimming()
        ducks = []
        for i in range(n):
            mood, strength = max(((m, float(getattr(body, m)[i])) for m in MOODS), key=lambda x: x[1]) if body else ("", 0.0)
            emote = emotes.get(i)
            showing = emote is not None and stub.t - emote[0] < EMOTE_S
            level = lambda name: round(float(getattr(body, name)[i]), 2) if body else 0.0
            ducks.append({
                "name": stub.names[i].replace("duck-", ""), "label": label_of(body.k, i) if body and not self.blind else "",
                "x": round(float(stub.pose[i, 0]), 3), "y": round(float(stub.pose[i, 1]), 3),
                "h": round(float(stub.pose[i, 2]), 3),
                "asleep": bool(body.asleep[i]) if body else False, "sat": posture[i] == "sat",
                "down": posture[i] == "down", "down_left": round(float(down_left[i]), 2), "swimming": bool(swimming[i]), "hat": bool(stub.hats[i]), "hat_style": int(max(stub.hat_style[i], 0)),
                "head": [round(float(v), 3) for v in stub.head[i]],  # neck_pitch, head_pitch, head_yaw, head_roll, as told
                "eating": i in bites and stub.t - bites[i][0] < EATING_S,
                "kicking": i in kicks and stub.t - kicks[i][0] < KICKING_S,
                "drumming": i in taps and stub.t - taps[i][0] < 0.5,
                "mood": mood, "strength": round(strength, 2),
                "emote": emote[2] if showing else "", "emote_t": round(emote[0], 2) if showing else -1.0,
                "hunger": level("hunger"), "thirst": level("thirst"), "sleepy": level("sleep_pressure"),
                "knobs": {k: round(float(body.k[k][i]), 2) for k in SHAPE_KNOBS} if body and not self.blind else {},
                "readout": readout(body, i) if body and i == self.watched else [],
                "among": among(body, i, [n.replace("duck-", "") for n in stub.names]) if body and i == self.watched and hasattr(body, "bond") else {},
                "crying": bool(stub.crying_until[i] > stub.t),  # the selected duck's only: it is 400 bytes
                **joints[i],
            })
        return {"t": round(stub.t, 2), "size": w.size, "light": round(float(daylight(stub.t)), 3),
                "day": round(stub.t % DAY_S / DAY_S, 4), "ducks": ducks,
                "food": [[round(float(x), 3), round(float(y), 3), int(k)] for (x, y), k in zip(w.food, w.kinds)],  # and which fruit
                "danger": [[float(x), float(y)] for x, y in w.danger],
                "pond": None if w.pond is None else [float(v) for v in w.pond], "tree": list(w.tree), "rocks": w.rocks.round(3).tolist(),
                "wind": None if w.wind is None else [round(float(v), 3) for v in w.wind],  # where the air is going, m/s
                "hats": [[round(x, 3), round(y, 3), k] for x, y, k in stub.hat_items],
                "balls": [[round(float(x), 3), round(float(y), 3)] for x, y in w.balls[:, :2]],
                "drum": None if w.drum is None else list(w.drum),
                "held": list(stub.held) if stub.held else None,  # what the hand is carrying: [kind, which]
                "music": None if w.music is None else list(w.music), "music_volume": round(float(w.music_volume), 2),
                "hand": None if w.hand is None else [float(v) for v in w.hand], "toasts": self.toasts,
                "sounds": [[round(t, 2), int(i), tag] for t, i, tag in stub.sounds[-2 * n:] if stub.t - t < HEARD_S]}

    def ride(self, stub, server, duck: int) -> dict:
        """What the ridden duck sees and what its brain is asking of its legs, for the ride view's overlay:
        both hex retinas as bytes (left eye first) and the six descending readouts in Hz."""
        rates = server.decoder.rates
        return {"duck": duck, "hex": HEX, "lum": _b64(np.round(np.clip(stub.seen[duck], 0, 1) * 255).astype(np.uint8)),
                "dn": [] if rates is None else [[name, round(float(hz), 1)] for name, hz in zip(DN_NAMES, rates[duck])]}

    def step(self, stub, server=None) -> None:
        """Publish this step and do whatever the player asked. Call with the stub's lock held. `server` is
        the BrainServer, or None when the garden runs without a brain."""
        world = self.build(stub, server.body if server else None)
        if server is not None and getattr(server, "decoder", None) and self.wheel[0] >= 0 and stub.seen is not None:
            world["ride"] = self.ride(stub, server, self.wheel[0])
        if server is not None and getattr(server, "brainview", None) is not None and server.brainview.duck >= 0:
            world["brain"] = {"duck": server.brainview.duck, "points": str(POINTS), "spikes": server.brainview.take()}
        self.out.sendto(json.dumps(world).encode(), ("127.0.0.1", SNAPSHOT_PORT))
        while True:
            try:
                action = json.loads(self.actions.recv(4096))
            except BlockingIOError:
                break
            method, p = str(action.get("method", "")), action.get("params", {})
            if method == "garden.watch":  # which duck the viewer has selected, or -1 for none
                self.watched = int(p["duck"]) if 0 <= int(p["duck"]) < len(stub.names) else -1
                if server is not None and hasattr(server, "brainview"):
                    server.brainview.duck = int(p["duck"]) if 0 <= int(p["duck"]) < len(stub.names) else -1
            elif method == "garden.wheel":
                self.wheel = (int(p["duck"]), float(np.clip(p["fwd"], -1, 1)), float(np.clip(p["turn"], -1, 1)), stub.t)
            elif method.startswith("garden."):  # the player's calls, and nothing else
                try:
                    stub._control_call(method, p)
                except (KeyError, ValueError, TypeError, IndexError) as e:
                    # A viewer newer or older than this garden asks for things it does not have, or sends them
                    # wrong. That is the viewer's mistake and must not stop the garden: say so once, and go on.
                    if method not in self.refused:
                        self.refused.add(method)
                        print(f"ignoring {method} from the viewer: {type(e).__name__} {e}")
        duck, fwd, turn, asked = self.wheel
        if server is not None:
            held = 0 <= duck < len(stub.names) and stub.t - asked < WHEEL_S
            if self.riding >= 0 and (not held or duck != self.riding):
                server.possessed[self.riding] = False  # its legs are its brain's again: getting off used to leave it muted for good
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
        time.sleep(0.05)  # the loopback delivers when it likes: read at once, the two calls were sometimes not there yet
        snap.step(stub)
        got = json.loads(rx.recv(65535))
        world_fields = {"t", "size", "light", "day", "ducks", "food", "danger", "pond", "tree", "rocks", "wind", "hats", "balls", "drum", "held", "music", "music_volume", "hand",
                        "toasts", "sounds"}
        assert world_fields <= set(got), f"the snapshot lost {world_fields - set(got)}: Godot reads every one of these"
        duck_fields = {"name", "label", "x", "y", "h", "asleep", "sat", "down", "swimming", "hat", "hat_style", "head", "eating", "kicking",
                       "mood", "strength", "emote", "emote_t", "knobs", "readout", "among", "crying"}
        assert duck_fields <= set(got["ducks"][0]), f"a duck lost {duck_fields - set(got['ducks'][0])}"
        assert len(got["ducks"]) == 3 and got["ducks"][1]["emote"] == "happy" and got["ducks"][0]["emote"] == ""
        assert got["toasts"] == ["b looks happy"], got["toasts"]
        assert len(stub.world.food) == dishes + 1, "a player's feed reaches the garden"
        assert stub.t == t, "only garden.* calls are taken from the network"
        tx.sendto(json.dumps({"method": "garden.no_such_thing", "params": {}}).encode(), ("127.0.0.1", ACTION_PORT))
        tx.sendto(json.dumps({"method": "garden.pet", "params": {"duck": 99}}).encode(), ("127.0.0.1", ACTION_PORT))
        time.sleep(0.05)
        snap.step(stub)  # neither stops the garden
        from types import SimpleNamespace
        server = SimpleNamespace(possessed=np.zeros(3, bool), body=None)
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
