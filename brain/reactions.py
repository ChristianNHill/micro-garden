"""Chains of reactions, as explicit code: after Bailey and Katchabaw's Actor/Reactor system (RESEARCH.md §2).

Something a duck does or suffers is a stimulus: a kind, the duck that did it, the duck it was done to, and a
heat. Every duck near enough asks first whether it cares (whether sleep, a pressing need or the dark has it
busy) and then rolls one response from the table for that kind. Its personality, its ties to the two ducks,
and whether it is outmatched weight each response, and letting it go always has weight too, so nothing is
certain and nothing is ruled out.

A response is itself a stimulus, one link further down the chain and cooler for it (CHAIN_KEEP), so reactions
chain: the Bully shoves a duck, the duck's kind friend goes to comfort it, a bolder friend goes to shove the
Bully back, and the Bully's own friend may join in. Every duck already caught up in something about the same
duck cools the next one's interest (CROWD), so most chains stop after a link or two, and a whole-garden brawl
or a group comfort is rarer than once a day.

This sits on top of brain/social.py, which keeps the reactions it already had (laughing at a fall, a kind duck
drawn to a cry, blaming a shover, squabbling at a dish). A response that means going somewhere becomes an
intent: a duck to walk to, what to do on arrival, and how long to keep trying. It steers the duck through the
decoder like any other want (`intent_turn`, `intent`), and the fly brain's legs carry it there.
"""
from dataclasses import dataclass

import numpy as np

from brain.physiology import pressing

CHAIN_KEEP = 0.5  # of a stimulus's heat, what a response to it passes on
CROWD = 0.3  # each duck already caught up about the same duck leaves the next this much of its interest
IGNORE = 1.5  # the weight of letting it go, against the responses' own weights (about 0 to 1.5 each)
NOTICE = 0.5  # how plainly a duck must smell another to notice what happens to it: about 1.5 m
SCENT_HALF = 0.05  # as brain/social.py: a duck about 1.5 m off smells half as plain
SCENT_CONTRAST = 8.0
INTENT_S = 20.0  # how long a duck keeps trying to get there, long enough to get up first if it was knocked down
ARRIVE = {"comfort": 0.6, "shove": 0.7}  # near_* summed (1 - metres apart) at which it has got there
MOOD_STEP = 0.25  # what an angry duck nearby does to a witness's mood, and a comfort to an onlooker's joy


@dataclass
class Stimulus:
    kind: str  # "shove", "fall", "laugh", "comfort", "angry" or "sad"
    actor: int  # the duck that did it, or had it happen (a fall, a sad face)
    target: int = -1  # the duck it was done to, -1 for none
    heat: float = 1.0  # 1 at the start of a chain


# The one a comforter goes to: the duck that was hurt, which is the target of a shove or a laugh and the actor
# of a fall or a sad face.
HURT = {"shove": "target", "laugh": "target", "fall": "actor", "sad": "actor"}


def weights(kind: str, body, w: int, actor: int, target: int) -> dict[str, float]:
    """Each response duck w could make to a stimulus, before heat, busyness and the crowd."""
    k = lambda name: float(body.k[name][w])
    tie = lambda x: float(body.bond[w, x]) if x >= 0 else 0.0
    friend, grudge = (lambda x: max(tie(x), 0.0)), (lambda x: max(-tie(x), 0.0))
    # not outmatched: 1 against a duck no more aggressive than w, less against a fiercer one
    dares = lambda x: 1.0 - max(float(body.k["aggressiveness"][x]) - k("aggressiveness"), 0.0)
    if kind == "shove" and w == target:
        return {"retaliate": k("aggressiveness") * dares(actor) * (0.3 + grudge(actor)), "flee": k("timidity") * 0.6}
    if kind == "shove":
        return {"comfort": k("kindness") * (0.3 + friend(target)),
                "defend": k("aggressiveness") * (0.2 + friend(target)) * (1 + grudge(actor)) * dares(actor),
                "join": k("aggressiveness") * (1 - k("kindness")) * (0.1 + friend(actor)) * (0.2 + grudge(target)),
                "flee": k("timidity") * (0.3 + grudge(actor))}
    if kind == "fall":
        return {"comfort": k("kindness") * (0.2 + friend(actor))}
    if kind == "laugh":
        return {"laugh": (1 - k("kindness")) * k("playfulness") * (0.2 + grudge(target)),
                "defend": k("aggressiveness") * friend(target) * dares(actor),
                "comfort": k("kindness") * (0.2 + friend(target))}
    if kind == "comfort":
        return {"cheer": 0.5 * k("sociability") * (0.5 + friend(actor))}
    if kind == "angry":  # the Angry Individual: a match gets angry back, a kind one sad, the rest afraid
        mine, theirs = k("aggressiveness"), float(body.k["aggressiveness"][actor])
        return {"anger": mine * (1.0 if mine >= theirs - 0.1 else 0.2), "sadden": k("kindness") * (1 - mine),
                "fear": (1 - k("kindness")) * (1 - mine) * (0.5 + k("timidity"))}
    if kind == "sad":  # a friend in need: a friend comes, and even a cheerful stranger lifts it a little
        return {"comfort": k("kindness") * (0.2 + friend(actor)) * (0.5 + k("sociability"))}
    return {}


class Reactions:
    def __init__(self, n: int, rng):
        self.rng = rng
        self.n = n
        self.intent: list[tuple | None] = [None] * n  # (act, target, until, heat)
        self.heat_of: dict[tuple, float] = {}  # (actor, target) of a shove sent as a response, and its heat
        self.later: list[Stimulus] = []  # stimuli the responses made this step, for the next

    def busy(self, body, w: int, light: float) -> float:
        """How free duck w is to take notice, 0 to 1."""
        if body.asleep[w] or self.intent[w] is not None:
            return 0.0
        need = float(pressing(max(body.hunger[w], body.thirst[w])))
        return (1 - 0.7 * need) * (1 - 0.5 * float(body.fatigue[w])) * (0.4 + 0.6 * light)

    def caught_up(self, duck: int) -> int:
        return sum(1 for it in self.intent if it is not None and it[1] == duck)

    def react(self, body, stimuli: list[Stimulus], near: np.ndarray, light: np.ndarray, t: float) -> list[list[str]]:
        """Every duck's response to this step's stimuli: moods change here, intents are set, and the returned
        per-duck lists are what to act out now (an emote). `near[w, x]` is how plainly w smells x, 0 to 1."""
        acts = [[] for _ in range(self.n)]
        pending, self.later = self.later, []  # what responses make now waits for the next step
        for s in stimuli + pending:
            hurt = getattr(s, HURT[s.kind]) if s.kind in HURT else -1
            for w in range(self.n):
                if w == s.actor:
                    continue
                noticed = w == s.target or near[w, s.actor] > NOTICE or (s.target >= 0 and near[w, s.target] > NOTICE)
                free = self.busy(body, w, float(light[w])) if noticed else 0.0
                if free == 0.0:
                    continue
                crowd = CROWD ** self.caught_up(hurt if hurt >= 0 else s.actor)
                ws = {r: v * s.heat * free * crowd for r, v in weights(s.kind, body, w, s.actor, s.target).items() if v > 0}
                roll = self.rng.random() * (IGNORE + sum(ws.values()))
                for response, v in ws.items():
                    if roll < v:
                        self._respond(body, response, w, s, hurt, t, acts)
                        break
                    roll -= v
        return acts

    def _respond(self, body, response: str, w: int, s: Stimulus, hurt: int, t: float, acts) -> None:
        heat = s.heat * CHAIN_KEEP
        if response in ("retaliate", "defend"):
            self.intent[w] = ("shove", s.actor, t + INTENT_S, heat)
        elif response == "join":
            self.intent[w] = ("shove", s.target, t + INTENT_S, heat)
        elif response == "comfort" and hurt >= 0:
            self.intent[w] = ("comfort", hurt, t + INTENT_S, heat)
        elif response == "flee":
            body.fear[w] = min(body.fear[w] + MOOD_STEP, 1.0)  # the decoder runs it from the others
        elif response == "laugh":
            acts[w].append("emote_laugh")
            body.joy[w] = min(body.joy[w] + 0.15, 1.0)
            body.sorrow[s.target] = min(body.sorrow[s.target] + 0.15, 1.0)
            body.bond[s.target, w] -= 0.1
            self.later.append(Stimulus("laugh", w, s.target, heat))
        elif response == "cheer":
            body.joy[w] = min(body.joy[w] + MOOD_STEP, 1.0)
            body.bond[w, s.actor] += 0.05
        elif response in ("anger", "sadden", "fear"):
            mood = getattr(body, {"anger": "anger", "sadden": "sorrow", "fear": "fear"}[response])
            mood[w] = min(mood[w] + MOOD_STEP, 1.0)

    def steer(self, body, f: dict, left: np.ndarray, right: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
        """Walk each duck with an intent towards its duck, and act on arrival. Returns the turn towards it
        (-1 to 1, positive left), how much it wants to walk there (0 or 1), and what to do now."""
        turn, want, acts = np.zeros(self.n), np.zeros(self.n), [[] for _ in range(self.n)]
        beside = f["near_left"] + f["near_right"]
        for w, it in enumerate(self.intent):
            if it is None:
                continue
            act, x, until, heat = it
            if t > until or body.asleep[w] or body.asleep[x] or pressing(max(body.hunger[w], body.thirst[w])) > 0.9:
                self.intent[w] = None  # gave up, or a need got the better of it
                continue
            if int(f["near_id"][w]) == x and beside[w] >= ARRIVE[act]:
                self.intent[w] = None
                if act == "comfort":
                    acts[w].append("emote_comfort")
                    body.sorrow[x] = max(body.sorrow[x] - 0.3, 0.0)
                    body.fear[x] = max(body.fear[x] - 0.2, 0.0)
                    body.bond[x, w] += 0.2
                    body.bond[w, x] += 0.1
                    self.later.append(Stimulus("comfort", w, x, heat))
                else:
                    acts[w].append("strike")
                    self.heat_of[(w, x)] = heat
                continue
            total = left[w, x] + right[w, x]
            side = (left[w, x] - right[w, x]) / max(total, 1e-9)
            turn[w] = float(np.clip(side * SCENT_CONTRAST * (total / (total + SCENT_HALF)), -1, 1)) if total > 0 else 0.0
            want[w] = 1.0
        return turn, want, acts


if __name__ == "__main__":
    from brain.personality import preset, stack
    from brain.physiology import Physiology

    def garden(labels, seed=0):
        b = Physiology(len(labels), stack([preset(x) for x in labels]), hunger=0.2, thirst=0.2)
        b.bond = np.zeros((len(labels), len(labels)))
        return b, Reactions(len(labels), np.random.default_rng(seed))

    def tally(b, r, s, trials=600):
        """How often each duck makes each response to the same stimulus, starting fresh every time."""
        seen = [dict() for _ in range(r.n)]
        near, light = np.ones((r.n, r.n)), np.ones(r.n)
        for _ in range(trials):
            r.intent, r.later = [None] * r.n, []
            mood = b.anger.copy(), b.sorrow.copy(), b.fear.copy(), b.joy.copy()
            r.react(b, [s], near, light, 0.0)
            for w in range(r.n):
                done = ("intent " + r.intent[w][0] + " " + str(r.intent[w][1])) if r.intent[w] else \
                    "anger" if b.anger[w] > mood[0][w] else "sadden" if b.sorrow[w] > mood[1][w] else \
                    "fear" if b.fear[w] > mood[2][w] else "cheer" if b.joy[w] > mood[3][w] else "nothing"
                seen[w][done] = seen[w].get(done, 0) + 1
            b.anger[:], b.sorrow[:], b.fear[:], b.joy[:] = mood
        return seen

    top = lambda d: max((v, k) for k, v in d.items() if k != "nothing")[1]

    # The Angry Individual (Bailey and Katchabaw): a hostile, dominant duck among others. A match gets angry
    # back, a kind duck that is not dominant turns sad, an unkind one that is not dominant turns afraid.
    b, r = garden(["Bully", "Gentle", "Naughty", "Bully"])
    b.k["timidity"] = b.k["timidity"].copy()
    b.k["timidity"][2] = 0.9  # a skittish Naughty
    b.k["aggressiveness"] = b.k["aggressiveness"].copy()
    b.k["aggressiveness"][2] = 0.3
    seen = tally(b, r, Stimulus("angry", 0))
    assert (top(seen[1]), top(seen[2]), top(seen[3])) == ("sadden", "fear", "anger"), seen

    # A shove, and who does what about it: the victim's kind friend comforts it, its bold friend goes for the
    # Bully, the Bully's own friend piles on, and the victim itself, being timid, mostly lets it go or flees.
    b, r = garden(["Bully", "Scaredy", "Gentle", "Naughty", "Bully"])
    b.bond[2, 1] = b.bond[3, 1] = 0.8  # the Gentle and the Naughty are the Scaredy's friends
    b.bond[4, 0], b.bond[4, 1] = 0.8, -0.5  # the second Bully is the first one's friend and has it in for the Scaredy
    seen = tally(b, r, Stimulus("shove", 0, 1))
    assert top(seen[2]) == "intent comfort 1" and top(seen[3]) == "intent shove 0" and top(seen[4]) == "intent shove 1", seen
    assert seen[1].get("nothing", 0) > seen[1].get("intent shove 0", 0), seen[1]

    # Nothing is ruled out: even the Gentle, seeing its friend shoved, sometimes goes for the Bully.
    assert seen[2].get("intent shove 0", 0) > 0, seen[2]

    # Chains cool and crowds damp: the third duck to get caught up about the same duck is much less likely
    # to than the first, and a response two links down a chain is a quarter as likely.
    b, r = garden(["Gentle"] * 5)
    b.bond[:] = 0.8
    np.fill_diagonal(b.bond, 0)
    first = sum(v for k, v in tally(b, r, Stimulus("fall", 0))[1].items() if k != "nothing")
    r.intent = [None, None, ("comfort", 0, 99.0, 1.0), ("comfort", 0, 99.0, 1.0), None]
    rng, count = np.random.default_rng(1), 0
    for _ in range(600):
        r.rng, r.later = rng, []
        r.react(b, [Stimulus("fall", 0)], np.ones((5, 5)), np.ones(5), 0.0)
        count += r.intent[1] is not None
        r.intent[1] = None
    assert count < 0.25 * first, (count, first)

    # Walking over: an intent turns the duck towards its duck and, on arrival, comforts it.
    b, r = garden(["Gentle", "Scaredy"])
    r.intent[0] = ("comfort", 1, 10.0, 0.5)
    left, right = np.array([[0, 0.3], [0, 0]]), np.array([[0, 0.1], [0, 0]])
    far = {"near_id": np.array([-1.0, -1.0]), "near_left": np.zeros(2), "near_right": np.zeros(2)}
    turn, want, acts = r.steer(b, far, left, right, 0.0)
    assert turn[0] > 0.5 and want[0] == 1 and acts == [[], []], (turn, want, acts)
    b.sorrow[1] = 0.6
    there = {"near_id": np.array([1.0, 0.0]), "near_left": np.array([0.7, 0.0]), "near_right": np.zeros(2)}
    turn, want, acts = r.steer(b, there, left, right, 1.0)
    assert acts[0] == ["emote_comfort"] and r.intent[0] is None and b.sorrow[1] < 0.35 and r.later[0].kind == "comfort"
    print("ok  the angry individual is met with sadness, fear and anger by type; a shove brings comfort, a defender and a"
          " pile-on by personality and ties; chains cool and crowds damp; an intent walks a duck over and comforts")
