"""Emotes: a duck shows what it is feeling, now and then, the way a Chao or a Sim does (Chris, 2026-09-21).

Nothing here decides what a duck does; that is the brain's. This is a duck wearing its state on its sleeve:
every few seconds it may act out whichever of its moods and needs is strongest, and its personality sets how
readily. A chatty duck emotes often, a playful one turns joy into a roll, a vain one shows off, a timid
one cowers at less. It is decided on a slow clock and never per tick, since anything rolled fifty times a
second is a certainty (the hats at Gate 8b).

What an emote looks like is the body's business: the 2D stub writes it on the screen, and a robot acts it
out with its head, its voice and the skills it has (body/mujoco/adapter.py).
"""
import numpy as np

EVERY_S = 5.0  # how often a duck considers emoting
CHANCE = 0.35  # of doing so each time, for the most expressive duck whose feeling is at full strength
FELT_AT = 0.4  # below this a feeling is not worth showing

EMOTES = ("happy", "playful", "scared", "angry", "sad", "lonely", "bored", "hungry", "thirsty", "sleepy",
          "curious", "proud")


def feelings(body, i: int) -> dict[str, float]:
    """How strongly duck i feels each thing worth showing, 0 to 1, from its moods, needs and knobs."""
    k = lambda name: float(body.k[name][i])
    v = lambda name: float(getattr(body, name)[i])
    joy = v("joy")
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
        "curious": 0.6 * k("curiosity") * (1 - max(v("hunger"), v("thirst"))),  # a content, nosy duck looks about
        "proud": 0.7 * k("vanity") * (1 - max(v("hunger"), v("thirst"))),
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
    from brain.personality import preset, stack
    from brain.physiology import Physiology
    rng = np.random.default_rng(0)
    b = Physiology(3, stack([preset(x) for x in ("Chatty", "Quiet", "Zoomer")]), hunger=0.2, thirst=0.2)
    assert all(pick(b, i, rng) in (None, "curious", "proud") for i in range(3) for _ in range(50)), "a content duck shows little"
    b.joy[:] = 1.0
    shown = [[pick(b, i, rng) for _ in range(400)] for i in range(3)]
    count = [sum(x is not None for x in s) for s in shown]
    assert count[0] > 1.5 * count[1], f"a chatty duck emotes more than a quiet one: {count}"
    assert shown[2].count("playful") > shown[2].count("happy"), "a playful duck turns joy into play"
    b.joy[:] = 0; b.fear[:] = 0.9
    assert all(x in (None, "scared", "curious", "proud") for x in (pick(b, 0, rng) for _ in range(100)))
    b.asleep[:] = True
    assert pick(b, 0, rng) is None
    print(f"ok  full of joy, in 400 tries: Chatty {count[0]} emotes, Quiet {count[1]}, Zoomer {count[2]} "
          f"({shown[2].count('playful')} of them playful)")
