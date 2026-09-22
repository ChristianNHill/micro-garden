"""What the ducks are to each other, and to the player's hand.

Explicit code, none of it the fly's: who likes whom, who has been treated well, who learned what from
watching. These values change slowly and reach the legs through explicit turns in the decoder (towards a
friend, away from a grudge, towards a kind hand, towards a cry). The body reports what happened and to
whom (body/frames.py); this decides what a duck makes of it, by its knobs.

Everything lives on the Physiology so a save keeps it (brain/save.py keeps every array there):
  bond[i, j]   what duck i thinks of duck j, -1 a grudge to 1 a friend; not mutual
  hand_trust   what it thinks of the player's hand, -1 to 1
  favourite    the fruit it likes best
  dance_skill, swim_skill, run_skill   0 to 1, grown by doing and, for dancing, by watching
"""
import numpy as np

from world.fields import FRUITS

BOND_FADES_S = 1800.0  # decay time of a bond nobody renews
TOGETHER_S = 120.0  # this long at ease beside one duck adds a full step of bond
SCENT_HALF = 0.05  # a duck about 1.5 m off counts half
SCENT_CONTRAST = 8.0  # two antennae 10 cm apart differ by a few percent; this amplifies it
KINDLY = 0.5  # kindness past this minds what happens to others


def init(body, seed: int = 0) -> None:
    n = len(body.hunger)
    rng = np.random.default_rng(seed + 11)
    body.bond = np.zeros((n, n))
    body.hand_trust = np.zeros(n)
    body.favourite = rng.integers(0, FRUITS, n).astype(float)
    body.dance_skill = np.full(n, 0.2)
    body.swim_skill, body.run_skill = np.zeros(n), np.zeros(n)


def trait(body, name: str, past: float = 0.5) -> np.ndarray:
    """How far a knob is past `past`, 0 to 1: a duck in the middle of a dial has none of the trait."""
    return np.clip((body.k[name] - past) / (1 - past), 0, 1)


def update(body, f: dict, dt: float, speed: np.ndarray) -> None:
    """One body step of all of it. `f` is the frames' fields as arrays, one value a duck."""
    n = len(body.hunger)
    me = np.arange(n)
    fan, put_off = trait(body, "sociability"), np.clip(1 - 2 * body.k["sociability"], 0, 1)
    cross, kind, vain = trait(body, "aggressiveness", 0.6), trait(body, "kindness", KINDLY), trait(body, "vanity")
    who = lambda name: (f[name].astype(int), f[name] >= 0)

    by, hit = who("bumped_by")  # shoved: a grudge, whoever it is
    body.bond[me[hit], by[hit]] -= 0.3
    by, saw = who("saw_shove_by")  # saw a shove: a kind duck blames the shover and is sorry,
    body.bond[me[saw], by[saw]] -= 0.15 * kind[saw]  # a timid one is frightened (the decoder does the flight)
    victim, _ = who("saw_shove_of")
    body.bond[me[saw], victim[saw]] += 0.1 * kind[saw]
    body.sorrow = np.clip(body.sorrow + 0.1 * kind * saw, 0, 1)
    body.fear = np.clip(body.fear + 0.25 * body.k["timidity"] * saw, 0, 1)
    by, show = who("show_by")  # a song or a dance changes the bond with the performer
    body.bond[me[show], by[show]] += (0.08 * fan * (1 - cross) - 0.05 * put_off - 0.08 * cross)[show]
    body.dance_skill = np.clip(body.dance_skill + 0.04 * fan * (1 - cross) * show, 0, 1)  # dancing is learned by watching
    by, took = who("hat_taken_by")  # a hat taken under a vain duck's beak
    body.bond[me[took], by[took]] -= 0.2 * vain[took]
    body.anger = np.clip(body.anger + 0.3 * vain * took, 0, 1)
    by, held = who("comforted_by")  # another duck came to it while it cried
    body.bond[me[held], by[held]] += 0.25
    body.bond[by[held], me[held]] += 0.1
    body.sorrow = np.clip(body.sorrow - 0.4 * held, 0, 1)
    np.add.at(body.joy, by[held], 0.1)

    near, with_one = who("near_id")  # time at ease beside one duck, more for sleeping beside it
    at_ease = np.maximum(body.hunger, body.thirst) < 0.6
    both_asleep = body.asleep & body.asleep[np.where(with_one, near, 0)]
    grows = dt / TOGETHER_S * (fan * at_ease + 2.0 * both_asleep) * with_one
    body.bond[me[with_one], near[with_one]] += grows[with_one]
    body.bond *= np.exp(-dt / BOND_FADES_S)
    np.clip(body.bond, -1, 1, out=body.bond)
    np.fill_diagonal(body.bond, 0.0)

    # moods are catching: an alarm nearby frightens, a whoop nearby cheers the sociable
    body.fear = np.clip(body.fear + 0.25 * (0.3 + body.k["timidity"]) * (f["heard_alarm"] > 0), 0, 1)
    body.joy = np.clip(body.joy + 0.1 * fan * (f["heard_joy"] > 0), 0, 1)

    # the hand: petting and feeding earn trust, a clap loses a little
    body.hand_trust = np.clip(body.hand_trust + 0.08 * (f["petted"] > 0) + 0.15 * (f["hand_fed"] > 0) - 0.05 * (f["scared"] > 0), -1, 1)
    body.joy = np.clip(body.joy + 0.25 * (f["ate_kind"] == body.favourite), 0, 1)  # its favourite fruit
    # Picked up: a trusting duck enjoys it and trusts more; others are frightened, the timid most.
    # Thrown: any duck is frightened and trusts the hand much less; an aggressive one gets angry.
    held, thrown, trusting = f["held"] > 0, f["thrown"] > 0, body.hand_trust > 0.2
    body.joy = np.clip(body.joy + dt * 0.3 * held * trusting, 0, 1)
    body.fear = np.clip(body.fear + dt * 0.5 * (0.3 + body.k["timidity"]) * held * ~trusting + 0.5 * thrown, 0, 1)
    body.anger = np.clip(body.anger + 0.4 * cross * thrown, 0, 1)
    body.hand_trust = np.clip(body.hand_trust + dt * 0.02 * held * trusting - 0.35 * thrown, -1, 1)

    body.swim_skill = np.clip(body.swim_skill + dt / 300.0 * (f["swimming"] > 0), 0, 1)
    body.run_skill = np.clip(body.run_skill + dt / 600.0 * (np.abs(speed) > 0.1), 0, 1)


def performed(body, i: int) -> None:
    body.dance_skill[i] = min(body.dance_skill[i] + 0.02, 1.0)


def steering(body, f: dict, scent=None) -> dict:
    """What the decoder needs, one value per duck (see brain/decoder.py). `scent` is each duck's smell of
    every other as (n, n) left and right (BrainServer._scents); None reads the frames' columns, which is
    the same when every duck is in one garden."""
    n = len(body.hunger)
    near = f["near_id"].astype(int)
    with_one = f["near_id"] >= 0
    kind = trait(body, "kindness", KINDLY)
    crying = np.maximum(f["cry_left"], f["cry_right"])
    hand = np.maximum(f["hand_left"], f["hand_right"])
    # Friends and grudges from afar: each duck has its own smell, and the left/right difference as a share
    # of the total gives its side even when faint; its weight falls off with faintness.
    left, right = scent if scent is not None else (f["scent_left"][:, :n], f["scent_right"][:, :n])
    total = left + right
    side = (left - right) / np.maximum(total, 1e-9)  # +1 all on the left, -1 all on the right
    plain = total / (total + SCENT_HALF)
    bond_turn = np.clip((body.bond * side * plain).sum(axis=1) * SCENT_CONTRAST, -1, 1)
    return {
        "bond_turn": bond_turn,
        "bond_near": np.where(with_one, body.bond[np.arange(n), np.where(with_one, near, 0)], 0.0),
        "near_left": f["near_left"], "near_right": f["near_right"],
        "hand_trust": body.hand_trust, "hand_left": f["hand_left"], "hand_right": f["hand_right"],
        "comfort": kind, "cry_left": f["cry_left"], "cry_right": f["cry_right"],
        # a kind duck that is not starving leaves the food to one that is crying for it
        "sharing": (kind > 0.3) & (crying > 0.3) & (body.hunger < 0.6),
        # reasons to walk: a cry to go to, a trusted hand to come to
        "social_want": np.maximum(kind * crying, np.clip(body.hand_trust, 0, 1) * hand),
        "sleepy_together": trait(body, "sociability") * np.clip((body.sleep_pressure - 0.6) / 0.4, 0, 1),
        "swim_skill": body.swim_skill, "run_skill": body.run_skill,
    }


def friends(body, i: int, names: list[str]) -> tuple[str, str]:
    """Duck i's best friend and worst grudge, by name, or "" where it has neither to speak of."""
    row = body.bond[i]
    best, worst = int(row.argmax()), int(row.argmin())
    return (names[best] if row[best] > 0.15 else "", names[worst] if row[worst] < -0.15 else "")


if __name__ == "__main__":
    from body import frames
    from brain.personality import preset, stack
    from brain.physiology import Physiology
    labels = ["Bully", "Gentle", "Scaredy", "Chatty"]
    b = Physiology(4, stack([preset(x) for x in labels]), hunger=0.2, thirst=0.2)
    init(b)
    one = frames.unpack(frames.pack())
    blank = lambda: {name: np.array([one[name]] * 4, float) for name in one.dtype.names if name != "lum"}
    f = blank()
    f["bumped_by"][2] = 0  # the Bully shoves the Scaredy, and the Gentle and the Chatty see it
    f["saw_shove_by"][[1, 3]], f["saw_shove_of"][[1, 3]] = 0, 2
    update(b, f, 0.02, np.zeros(4))
    assert b.bond[2, 0] < -0.25 and b.bond[1, 0] < -0.1 and b.bond[1, 2] > 0.05, b.bond.round(2)
    assert abs(b.bond[3, 0]) < 0.05, "a duck of ordinary kindness keeps out of it"
    f = blank(); f["petted"][1] = 1; f["hand_fed"][1] = 1; f["scared"][:] = 1
    update(b, f, 0.02, np.zeros(4))
    assert b.hand_trust[1] > 0.15 and b.hand_trust[0] < 0
    before = b.hand_trust.copy()
    f = blank(); f["thrown"][1] = 1; f["held"][0] = 1
    update(b, f, 0.02, np.zeros(4))
    assert b.hand_trust[1] < before[1] - 0.3 and b.fear[1] > 0.4 and b.fear[0] > 0, "thrown, a duck trusts the hand less; held by a hand it does not trust, it is afraid"
    f = blank(); f["show_by"][[0, 1]] = 3  # the Chatty sings to the Bully and the Gentle
    update(b, f, 0.02, np.zeros(4))
    assert b.bond[1, 3] > 0 and b.bond[0, 3] < 0 and b.dance_skill[1] > 0.2
    f = blank(); f["near_id"][1], f["near_left"][1] = 3, 0.8
    s = steering(b, f)
    assert s["bond_near"][1] > 0 and s["bond_near"][0] == 0
    f = blank(); f["scent_left"][2, 0], f["scent_right"][2, 0] = 0.021, 0.019  # the Bully, two metres off to the Scaredy's left
    assert steering(b, f)["bond_turn"][2] < -0.05 and steering(b, f)["bond_turn"][3] == 0, "a grudge turns a duck away from far off"
    assert friends(b, 2, labels) == ("", "Bully") and friends(b, 3, labels) == ("", ""), "one shove is a grudge for the duck it landed on"
    # one server, two gardens of two ducks
    from types import SimpleNamespace
    from brain.server import BrainServer
    two = SimpleNamespace(first=np.array([0, 0, 2, 2]), n=4)
    f2 = np.array([frames.blank() for _ in range(4)], frames.FRAME)
    f2["bumped_by"][3], f2["ate_kind"][1] = 0, 0  # the second garden's first duck shoves its other one
    f2["scent_left"][2, 1], f2["scent_left"][0, 3] = 0.5, 0.75
    g = BrainServer._in_all_gardens(two, f2)
    L, _ = BrainServer._scents(two, g)
    assert g["bumped_by"][3] == 2 and g["ate_kind"][1] == 0 and L[2, 3] == 0.5 and L[0, 3] == 0, "ids and smells stay in their garden"
    print("ok  a shove makes a grudge and kind witnesses take sides; a song makes a friend of the Gentle and an "
          f"enemy of the Bully; the hand is trusted by the duck it fed ({b.hand_trust[1]:+.2f}) and not by the rest")
