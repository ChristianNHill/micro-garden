"""Emotes: explicit code, not the fly brain, that lets a duck show what it is feeling now and then.

Emotes do not steer the duck; the brain does. Every few seconds a duck may act out its strongest mood
or need, and its personality sets how readily. It is decided on a slow clock, never per tick, since a
chance rolled fifty times a second is a certainty.

The body decides what an emote looks like: the 2D stub writes it on screen, a robot acts it out
(body/mujoco/adapter.py).
"""
import numpy as np

EVERY_S = 5.0  # how often a duck considers emoting
CHANCE = 0.35  # of doing so each time, for the most expressive duck whose feeling is at full strength
FELT_AT = 0.4  # below this a feeling is not worth showing

EMOTES = ("happy", "playful", "scared", "angry", "sad", "lonely", "bored", "hungry", "thirsty", "sleepy",
          "curious", "proud", "stomp", "yawn", "splash", "sing", "cower", "dance", "singdance", "cry", "laugh", "comfort")
# stomp, yawn, splash, sing and cower are signature tricks, scaled by a knob (aggressiveness, sleepiness,
# water_love, chattiness, timidity), not switched by preset label. SIGNATURE is where on the dial a knob's
# trick starts to show.
SIGNATURE = 0.6


DOES = {"laugh": "laughs", "comfort": "comforts a friend", "cry": "cries", "dance": "dances", "singdance": "sings and dances", "stomp": "stomps", "yawn": "yawns", "splash": "splashes", "sing": "sings", "cower": "cowers"}


AMUSING = ("sing", "dance", "singdance", "playful")  # these relieve boredom


def phrase(emote: str) -> str:
    """How an emote reads after a duck's name: "looks happy", "stomps"."""
    return DOES.get(emote, f"looks {emote}")


def feelings(body, i: int) -> dict[str, float]:
    """How strongly duck i feels each thing worth showing, 0 to 1, from its moods, needs and knobs."""
    k = lambda name: float(body.k[name][i])
    v = lambda name: float(getattr(body, name)[i])
    joy = v("joy")
    trait = lambda name: float(np.clip((k(name) - SIGNATURE) / (1 - SIGNATURE), 0, 1))  # 0 at SIGNATURE, 1 at the top
    return {
        "happy": joy * (1 - 0.5 * k("playfulness")),
        "playful": joy * k("playfulness") + 0.5 * v("boredom") * k("playfulness") * k("energy"),
        "scared": v("fear") * (0.5 + k("timidity")),
        "angry": v("anger"),
        "sad": v("sorrow") * (1 - k("sociability")),
        "lonely": v("sorrow") * k("sociability"),
        "bored": v("boredom") * (0.5 + k("boredom_rate")) * (1 - k("playfulness") * k("energy")),
        "hungry": max(v("hunger") - 0.5, 0) * 2 * (0.5 + k("appetite")),
        "thirsty": max(v("thirst") - 0.5, 0) * 2,
        "sleepy": max(v("sleep_pressure") - 0.5, 0) * 2,
        "curious": 0.6 * k("curiosity") * (1 - max(v("hunger"), v("thirst"))),
        "proud": 0.7 * k("vanity") * (1 - max(v("hunger"), v("thirst"))),
        "stomp": trait("aggressiveness") * max(v("anger"), 0.6 * max(v("hunger") - 0.4, 0) / 0.6),
        "yawn": trait("sleepiness") * max(v("sleep_pressure"), 0.45),
        "splash": trait("water_love") * float(body.swimming[i]),
        "sing": trait("chattiness") * (1 - max(v("hunger"), v("thirst"))) * (1 - v("fear")),
        "cower": trait("timidity") * v("fear") * 1.5,
        # heard by others: a miserable or starving duck cries, and a kind one comes
        "cry": max(v("sorrow"), max(v("hunger") - 0.85, 0) / 0.15) * 0.9,
        "dance": 0.0, "singdance": 0.0,  # performances, started by brain/server.py
        "laugh": 0.0, "comfort": 0.0,  # at a fall and beside a cry, started by what happens (brain/social.py, the body)
    }


def pick(body, i: int, rng: np.random.Generator) -> str | None:
    """The emote duck i shows now, if any. Called every EVERY_S, not every tick."""
    if body.asleep[i]:
        return None
    felt = {name: s for name, s in feelings(body, i).items() if s >= FELT_AT}
    if not felt:
        return None
    expressive = 0.4 + 0.6 * float(body.k["chattiness"][i])
    names, strengths = zip(*felt.items())
    strengths = np.clip(np.array(strengths), 0, 1)
    if rng.random() > CHANCE * expressive * strengths.max():
        return None
    return str(rng.choice(names, p=strengths / strengths.sum()))  # mostly the strongest, not always


if __name__ == "__main__":
    from body.stub2d.stub import EMOTE_ACTS
    assert set(EMOTES) == set(EMOTE_ACTS), "every feeling a duck can show has an act, and every act a feeling"
    from brain.personality import preset, stack
    from brain.physiology import Physiology
    rng = np.random.default_rng(0)
    b = Physiology(3, stack([preset(x) for x in ("Chatty", "Quiet", "Zoomer")]), hunger=0.2, thirst=0.2)
    assert all(pick(b, i, rng) in (None, "curious", "proud", "sing") for i in range(3) for _ in range(50)), "a content duck shows little"
    b.joy[:] = 1.0
    shown = [[pick(b, i, rng) for _ in range(1500)] for i in range(3)]
    count = [sum(x is not None for x in s) for s in shown]
    assert count[0] > 1.5 * count[1], f"a chatty duck emotes more than a quiet one: {count}"
    assert shown[2].count("playful") > shown[2].count("happy"), "a playful duck turns joy into play"
    b.joy[:] = 0; b.fear[:] = 0.9
    assert all(x in (None, "scared", "curious", "proud") for x in (pick(b, 0, rng) for _ in range(100)))
    b.asleep[:] = True
    assert pick(b, 0, rng) is None
    assert phrase("sing") == "sings" and phrase("happy") == "looks happy" and set(DOES) < set(EMOTES)
    five = Physiology(5, stack([preset(x) for x in ("Bully", "Napper", "Carefree", "Chatty", "Scaredy")]), hunger=0.2, thirst=0.2)
    five.anger[:] = 0.8; five.sleep_pressure[:] = 0.7; five.swimming[:] = True; five.fear[:] = 0.5
    tricks = [max(("stomp", "yawn", "splash", "sing", "cower"), key=lambda e: feelings(five, i)[e]) for i in range(5)]
    assert tricks[:3] == ["stomp", "yawn", "splash"] and tricks[4] == "cower", tricks
    five.fear[:] = 0
    assert max(("stomp", "yawn", "splash", "sing", "cower"), key=lambda e: feelings(five, 3)[e]) == "sing"
    assert feelings(five, 1)["stomp"] == 0 and feelings(five, 0)["cower"] == 0, "a trick belongs to its trait"
    print(f"ok  full of joy, in 1500 tries: Chatty {count[0]} emotes, Quiet {count[1]}, Zoomer {count[2]} "
          f"({shown[2].count('playful')} of them playful); each of the demo five has its own trick")
