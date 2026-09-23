"""What the ducks are to each other, and to the player's hand.

Explicit code, none of it the fly's: who likes whom, who has been treated well, who learned what from
watching. These values change slowly and reach the legs through explicit turns in the decoder (towards a
friend, away from a grudge, towards a kind hand, towards a cry). The body reports what happened and to
whom (body/frames.py); this decides what a duck makes of it, by its knobs.

Everything lives on the Physiology so a save keeps it (brain/save.py keeps every array there):
  bond[i, j]   what duck i thinks of duck j, -1 a grudge to 1 a friend; not mutual
  hand_trust   what it thinks of the player's hand, -1 to 1
  favourite    the fruit it likes best
  dance_skill, swim_skill, walk_skill, eat_skill, fight_skill, fashion_skill, music_skill   0 to 1, grown here by doing, and for dancing by watching
"""
import numpy as np

from brain.physiology import FOOD_NEAR_HALF, pressing
from world.fields import FRUITS

BOND_FADES_S = 1800.0  # decay time of a bond nobody renews
TOGETHER_S = 120.0  # this long at ease beside one duck adds a full step of bond
SCENT_HALF = 0.05  # a duck about 1.5 m off counts half
SCENT_CONTRAST = 8.0  # two antennae 10 cm apart differ by a few percent; this amplifies it
KINDLY = 0.5  # kindness past this minds what happens to others

# Getting good at something is an S-curve, not a line: a beginner fumbles for a while, then it clicks, and
# the last few percent come slowest. SEED is how fast a duck that has never done the thing picks it up.
PRACTICE_S = 1800.0  # doing it this long takes a duck about halfway
PRACTICE_SEED = 0.15
WALKING_MS = 0.03  # a duck moving faster than this on land is walking, and learning to walk
BITE_PRACTICE = 0.01  # a bite's worth of learning to eat; a few hundred bites to get good
WATCHED_DANCE, DANCED = 0.04, 0.02  # dancing is learned both ways, and faster by watching a better dancer
RUST_S = 14400.0  # a skill left alone fades this fast: four hours from full to nothing, against half an hour of practice
KEPT = 0.3  # what a duck has learned up to here it keeps; only the skill above it fades
STARVING = 0.8  # eating only gets worse while a duck is this hungry and still not eating

# Mishaps: a duck that is bad at something now and then gets it wrong. The chance falls with the square of
# the skill, so a beginner fails often, a duck halfway there a quarter as often, and a master never; a
# careless duck fails more. Each chance is for a duck with none of the skill and middling carelessness.
# The faster a duck goes the likelier it trips, with the square of its speed: at a run (RUN_MS) it trips every
# TRIP_S, at an amble about one time in seven as often.
TRIP_S, RUN_MS = 20.0, 0.3
FLOUNDER_S = 40.0  # seconds of swimming between flounders
FUMBLE_P = 0.08  # that a bite goes wrong and the fruit is kicked away
DANCE_FALL_P = 0.3  # that a dance ends on the floor
WHIFF_P = 0.35  # that a shove misses and the duck ends up on the floor itself
SHOVE_PRACTICE = 0.03  # a landed shove's worth of learning to fight
HARD_P = 0.8  # the share of a master fighter's shoves that are hard ones, knocking further and for longer
HAT_SLIP_S = 90.0  # seconds of wearing a hat between its falling off, for a duck with no fashion sense
DAPPER = 0.02  # joy a second that a vain duck at full fashion takes from wearing a hat
NOTE_PRACTICE = 0.02  # a note's worth of learning to play
SOUR_P = 0.3  # that a note comes out sour, for a duck that has never played
WELL_PLAYED = 0.15  # joy a fan takes from a show by a master musician, on top of the show itself
LAUGHED_AT = 0.3  # how much being laughed at hurts, as sorrow or as anger

# Personality makes things likelier or rarer and never rules them out. The kindest duck still laughs at a
# fall now and then, the gentlest still shoves at a crowded dish, and the unkindest still goes to a crying
# duck sometimes. Each is a chance, rolled when it could happen, from the least likely duck to the most.
LAUGH_P = (0.05, 0.8)  # that a duck laughs at a fall it sees
SQUABBLE_S = 20.0  # seconds of jostling at wanted food, or of being angry, between shoves at full aggression
SQUABBLE_FLOOR = 0.1  # the gentlest duck shoves this share as often
CARE_FLOOR = 0.15  # how strongly the least kind duck is drawn to a cry, against 1 for the kindest


def init(body, seed: int = 0) -> None:
    n = len(body.hunger)
    rng = np.random.default_rng(seed + 11)
    body.bond = np.zeros((n, n))
    body.hand_trust = np.zeros(n)
    body.favourite = rng.integers(0, FRUITS, n).astype(float)
    # the skills themselves are set up with the rest of the body (brain/physiology.py)


def practised(skill, doing, step, idle_dt):
    """Up the S-curve while a duck is at it, and slowly back down over the time it is not.

    `step` is how far practice carries it in the thick of the climb; `idle_dt` the seconds of not doing it
    that count against it, which for eating is only the time it went hungry with food to be had. It never
    fades below KEPT, or below where it is if it has not got that far.
    """
    faded = np.maximum(skill - idle_dt / RUST_S, np.minimum(skill, KEPT))
    return np.clip(np.where(doing, skill + step * (PRACTICE_SEED + skill) * (1 - skill), faded), 0, 1)


def fails(body, skill, chance, rng, who=slice(None)):
    """Whether a go at something goes wrong, per duck: `chance` for a duck with none of the skill."""
    careless = 0.5 + body.k["carelessness"][who]
    return rng.random(np.shape(skill)) < chance * (1 - skill) ** 2 * careless


def mishaps(body, f: dict, dt: float, speed: np.ndarray, rng) -> list[str | None]:
    """What goes wrong for each duck this step, as a skill for the body to act out, or None. A duck that
    falls in front of the others is a little embarrassed, the timid most."""
    awake = ~body.asleep
    pace = np.clip(np.abs(speed) / RUN_MS, 0, 1.5) ** 2
    tripped = awake & (f["swimming"] == 0) & fails(body, body.walk_skill, dt / TRIP_S * pace, rng)
    under = awake & (f["swimming"] > 0) & fails(body, body.swim_skill, dt / FLOUNDER_S, rng)
    fumbled = (f["ate"] > 0) & fails(body, body.eat_skill, FUMBLE_P, rng)
    slipped = awake & (f["hat"] > 0) & fails(body, body.fashion_skill, dt / HAT_SLIP_S, rng)
    body.sorrow = np.clip(body.sorrow + 0.15 * body.k["timidity"] * (tripped | under), 0, 1)
    return ["trip" if r else "flounder" if u else "fumble" if e else "lose_hat" if h else None
            for r, u, e, h in zip(tripped, under, fumbled, slipped)]


def squabbles(body, f: dict, dt: float, rng) -> np.ndarray:
    """Explicit code: whether each duck shoves the duck it is pressed against this step. A reason to push
    (food it wants, or anger) times its aggressiveness, with a floor so the gentlest duck still shoves now
    and then. The brain's own attacks (aIPg, brain/decoder.py) come on top of this."""
    touching = (f["touch_left"] + f["touch_right"]) > 0
    food = (f["odor_left"] + f["odor_right"]) / 2
    cause = np.maximum(pressing(body.hunger) * food / (food + FOOD_NEAR_HALF), body.anger)
    temper = SQUABBLE_FLOOR + (1 - SQUABBLE_FLOOR) * body.k["aggressiveness"]
    return touching & ~body.asleep & (rng.random(len(cause)) < cause * temper * dt / SQUABBLE_S)


def strike(body, i: int, rng) -> str:
    """How duck i's shove goes, by its fighting: a miss (it falls itself), a hard shove, or an ordinary one."""
    if fails(body, body.fight_skill[i], WHIFF_P, rng, i):
        return "whiff"
    return "headbutt_hard" if rng.random() < HARD_P * body.fight_skill[i] else "headbutt"


def plays(body, i: int, rng) -> str:
    """How duck i's note comes out: sour now and then, and less often the more it has played."""
    return "sour" if fails(body, body.music_skill[i], SOUR_P, rng, i) else "drum"


def falls_dancing(body, i: int, rng) -> bool:
    """Whether duck i's dance ends on the floor."""
    return bool(fails(body, body.dance_skill[i], DANCE_FALL_P, rng, i))


def trait(body, name: str, past: float = 0.5) -> np.ndarray:
    """How far a knob is past `past`, 0 to 1: a duck in the middle of a dial has none of the trait."""
    return np.clip((body.k[name] - past) / (1 - past), 0, 1)


def update(body, f: dict, dt: float, speed: np.ndarray, rng) -> np.ndarray:
    """One body step of all of it. `f` is the frames' fields as arrays, one value a duck. Returns which ducks
    laugh at a fall this step, for the server to act out."""
    n = len(body.hunger)
    me = np.arange(n)
    fan, put_off = trait(body, "sociability"), np.clip(1 - 2 * body.k["sociability"], 0, 1)
    cross, kind, vain = trait(body, "aggressiveness", 0.6), trait(body, "kindness", KINDLY), trait(body, "vanity")
    who = lambda name: (f[name].astype(int), f[name] >= 0)

    by, hit = who("bumped_by")  # shoved: a grudge, whoever it is, and practice for the one that landed it
    body.bond[me[hit], by[hit]] -= 0.3
    landed = np.zeros(n, bool)
    landed[by[hit]] = True
    by, saw = who("saw_shove_by")  # saw a shove: a kind duck blames the shover and is sorry,
    body.bond[me[saw], by[saw]] -= 0.15 * kind[saw]  # a timid one is frightened (the decoder does the flight)
    victim, _ = who("saw_shove_of")
    body.bond[me[saw], victim[saw]] += 0.1 * kind[saw]
    body.sorrow = np.clip(body.sorrow + 0.1 * kind * saw, 0, 1)
    body.fear = np.clip(body.fear + 0.25 * body.k["timidity"] * saw, 0, 1)
    # A fall seen: the unkinder and the more playful or aggressive a duck, the likelier it laughs, less at a
    # friend and more at a grudge. The one laughed at is hurt or angry, and holds it against the laugher. If
    # the fall makes the duck cry, the others come to comfort it (comforted_by, below).
    by, fell = who("saw_fall_by")
    toward = body.bond[me, np.where(fell, by, 0)]
    unkind = (1 - body.k["kindness"]) * np.maximum(body.k["playfulness"], body.k["aggressiveness"])
    unkind = np.clip(unkind * (1 - np.clip(toward, 0, 1)) * (1 + np.clip(-toward, 0, 1)), 0, 1)
    laughs = fell & (rng.random(n) < LAUGH_P[0] + (LAUGH_P[1] - LAUGH_P[0]) * unkind)
    body.joy = np.clip(body.joy + 0.15 * laughs, 0, 1)
    for laugher, victim in zip(me[laughs], by[laughs]):  # a bold duck is likelier to take it as anger
        if rng.random() < 0.2 + 0.6 * body.k["aggressiveness"][victim]:
            body.anger[victim] = min(body.anger[victim] + LAUGHED_AT, 1)
        else:
            body.sorrow[victim] = min(body.sorrow[victim] + LAUGHED_AT * (0.5 + body.k["timidity"][victim]), 1)
        body.bond[victim, laugher] -= 0.15
    by, show = who("show_by")  # a song or a dance changes the bond with the performer
    body.bond[me[show], by[show]] += (0.08 * fan * (1 - cross) - 0.05 * put_off - 0.08 * cross)[show]
    body.dance_skill = practised(body.dance_skill, show, WATCHED_DANCE * fan * (1 - cross), dt)  # learned by watching
    body.joy = np.clip(body.joy + (WELL_PLAYED * fan * body.music_skill[np.where(show, by, 0)] * show), 0, 1)  # played well
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

    # what a duck is good at, from what it has been doing lately: kept up by doing it, lost by not
    ate = f["ate"] > 0
    body.swim_skill = practised(body.swim_skill, f["swimming"] > 0, dt / PRACTICE_S, dt)
    walking = (np.abs(speed) > WALKING_MS) & (f["swimming"] == 0)
    body.walk_skill = practised(body.walk_skill, walking, dt / PRACTICE_S, dt)
    body.eat_skill = practised(body.eat_skill, ate, BITE_PRACTICE, dt * ((body.hunger >= STARVING) & ~ate))
    body.fight_skill = practised(body.fight_skill, landed, SHOVE_PRACTICE, dt)
    body.music_skill = practised(body.music_skill, f["drummed"] > 0, NOTE_PRACTICE, dt)
    hatted = f["hat"] > 0
    body.fashion_skill = practised(body.fashion_skill, hatted, dt / PRACTICE_S, dt)
    body.joy = np.clip(body.joy + dt * DAPPER * body.k["vanity"] * body.fashion_skill * hatted, 0, 1)  # looking good
    return laughs


def performed(body, i: int) -> None:
    body.dance_skill[i] = practised(body.dance_skill[i], True, DANCED, 0.0)


def steering(body, f: dict, scent=None) -> dict:
    """What the decoder needs, one value per duck (see brain/decoder.py). `scent` is each duck's smell of
    every other as (n, n) left and right (BrainServer._scents); None reads the frames' columns, which is
    the same when every duck is in one garden."""
    n = len(body.hunger)
    near = f["near_id"].astype(int)
    with_one = f["near_id"] >= 0
    kind = trait(body, "kindness", KINDLY)
    carer = CARE_FLOOR + (1 - CARE_FLOOR) * kind  # every duck is drawn to a cry a little, the kind ones most
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
        "comfort": carer, "cry_left": f["cry_left"], "cry_right": f["cry_right"],
        # a kind duck that is not starving leaves the food to one that is crying for it
        "sharing": (kind > 0.3) & (crying > 0.3) & (body.hunger < 0.6),
        # reasons to walk: a cry to go to, a trusted hand to come to
        "social_want": np.maximum(carer * crying, np.clip(body.hand_trust, 0, 1) * hand),
        "sleepy_together": trait(body, "sociability") * np.clip((body.sleep_pressure - 0.6) / 0.4, 0, 1),
        "swim_skill": body.swim_skill, "walk_skill": body.walk_skill,
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
    dice = np.random.default_rng(1)
    b = Physiology(4, stack([preset(x) for x in labels]), hunger=0.2, thirst=0.2)
    init(b)
    one = frames.unpack(frames.pack())
    blank = lambda: {name: np.array([one[name]] * 4, float) for name in one.dtype.names if name != "lum"}
    f = blank()
    f["bumped_by"][2] = 0  # the Bully shoves the Scaredy, and the Gentle and the Chatty see it
    f["saw_shove_by"][[1, 3]], f["saw_shove_of"][[1, 3]] = 0, 2
    update(b, f, 0.02, np.zeros(4), dice)
    assert b.bond[2, 0] < -0.25 and b.bond[1, 0] < -0.1 and b.bond[1, 2] > 0.05, b.bond.round(2)
    assert abs(b.bond[3, 0]) < 0.05, "a duck of ordinary kindness keeps out of it"
    f = blank(); f["petted"][1] = 1; f["hand_fed"][1] = 1; f["scared"][:] = 1
    update(b, f, 0.02, np.zeros(4), dice)
    assert b.hand_trust[1] > 0.15 and b.hand_trust[0] < 0
    before = b.hand_trust.copy()
    f = blank(); f["thrown"][1] = 1; f["held"][0] = 1
    update(b, f, 0.02, np.zeros(4), dice)
    assert b.hand_trust[1] < before[1] - 0.3 and b.fear[1] > 0.4 and b.fear[0] > 0, "thrown, a duck trusts the hand less; held by a hand it does not trust, it is afraid"
    f = blank(); f["show_by"][[0, 1]] = 3  # the Chatty sings to the Bully and the Gentle
    update(b, f, 0.02, np.zeros(4), dice)
    assert b.bond[1, 3] > 0 and b.bond[0, 3] < 0 and b.dance_skill[1] > 0, "the sociable one picks up a little dancing"
    f = blank(); f["ate"][0] = 1
    update(b, f, 0.02, np.zeros(4), dice)
    assert b.eat_skill[0] > 0 and b.eat_skill[1] == 0, "a duck learns to eat by eating"

    def bites(start, span):  # how much practice it takes to climb that far from there
        x, n = start, 0
        while x < start + span:
            x, n = practised(x, True, BITE_PRACTICE, 0.0), n + 1
        return n
    assert bites(0.4, 0.15) < bites(0.0, 0.15) and bites(0.4, 0.15) < bites(0.8, 0.15), \
        "practice tells slowest at both ends: fumbling at first, and the last few percent"
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

    dry = Physiology(4, stack([preset(x) for x in labels]), hunger=0.2, thirst=0.2)  # its own garden: update ages everything
    init(dry)
    dry.swim_skill[:] = 0.5
    for _ in range(600):  # ten minutes out of the water
        update(dry, blank(), 1.0, np.zeros(4), dice)
    assert 0.47 > dry.swim_skill[0] > 0.44, f"a duck out of the water gets worse at swimming: {dry.swim_skill[0]:.3f}"
    assert practised(0.2, False, 0.0, 1e6) == 0.2 and practised(0.9, False, 0.0, 1e6) == KEPT, \
        "a skill never fades below 30%, and one under it does not fade at all"

    seen = Physiology(4, stack([preset(x) for x in labels]), hunger=0.2, thirst=0.2)  # a fall, in its own garden
    init(seen)
    f = blank(); f["saw_fall_by"][[0, 1, 3]] = 2  # the Scaredy falls in front of the Bully, the Gentle and the Chatty
    was = seen.sorrow[2] + seen.anger[2]
    laughed = sum(update(seen, f, 0.02, np.zeros(4), dice) for _ in range(400))  # four hundred falls
    assert laughed[0] > 3 * laughed[1] > 0, f"the Bully laughs most, and even the Gentle sometimes: {laughed}"
    assert seen.sorrow[2] + seen.anger[2] > was and seen.bond[2, 0] < 0, "being laughed at hurts, and is held against the laugher"
    pressed = blank(); pressed["touch_left"][:] = 1; pressed["odor_left"][:] = pressed["odor_right"][:] = 1.0
    seen.hunger[:], seen.anger[:], seen.asleep[:] = 0.9, 0.0, False
    shoves = sum(squabbles(seen, pressed, 1.0, dice) for _ in range(3600))  # an hour pressed together at the dish
    assert shoves[0] > 3 * shoves[1] > 0, f"the Bully shoves most at a crowded dish, and even the Gentle sometimes: {shoves}"
    rng = np.random.default_rng(0)
    seen.walk_skill[:] = [0.0, 0.5, 1.0, 0.0]
    running = np.full(4, 0.3)
    trips = np.zeros(4)
    quiet = blank()
    for _ in range(1800):  # half an hour of running, a second at a time: about 90 trips at no skill, 22 at half
        trips += [m == "trip" for m in mishaps(seen, quiet, 1.0, running, rng)]
    assert trips[0] > 2 * trips[1] > 0 and trips[2] == 0, f"a beginner trips most, a master never: {trips}"
    print("ok  a shove makes a grudge and kind witnesses take sides; a song makes a friend of the Gentle and an "
          f"enemy of the Bully; the hand is trusted by the duck it fed ({b.hand_trust[1]:+.2f}) and not by the rest")
