"""Bodily state per duck: drives and Chao-style emotions, shaped by personality knobs.

Drives rise on their own clocks and are satisfied by world events; emotions jump on events and fade.
As in the Chao games, knobs mostly set those rates. The body acts on the brain by
scaling sensory gains and adding mood input (`sense_gains`, `aggression_tone`), and shapes how the
decoder moves the duck (`motor`).

Aggression is a mood, not a reflex: a tonic input into pC1d/e raises aIPg, and touch then turns that
into an attack (decoder). The mood is personality times a trigger: hunger near food, or anger from
being headbutted.
"""
import numpy as np

from brain.personality import KNOB_DEFAULT, KNOBS

# From just fed to fully hungry, at appetite 0.5: one garden day (world/fields.py DAY_S). Faster, and a
# body walking 0.06 m/s between dish and pond cannot keep up with five ducks' bites and sips.
HUNGER_RISE_S = 600.0
BITE_FULLNESS = 0.1
EAT_GROWS = 0.5  # how much more a fully practised eater takes from a bite
THIRST_RISE_S = 800.0  # kept at four thirds of hunger's
SIP_QUENCH = 0.1
FATIGUE_M = 30.0  # metres of walking from rested to exhausted, at energy 0.5
REST_S = 60.0  # standing still from exhausted to rested
# Losing a smell. A fly that walks out of an odor it was following slows down and turns hard, a local
# search that puts it back in the plume (Alvarez-Salvado et al. 2018, the OFF response). Without it a duck
# walks straight upwind past a dish to one side, because going upwind never corrects sideways.
# How long the smell it had stays with it, and so how long it searches for one it lost. A walking fly keeps
# searching for tens of seconds after losing a food cue; at 5 s a duck that overshot gave up too soon.
SCENT_MEMORY_S = 30.0
# A duck can only lose a smell it had: below this its memory of one counts as none, or a duck in a garden
# with no food would search forever. 0.05 is a tenth of the smell sense's half level.
SCENT_FLOOR = 0.05
SEARCH_TURN, SEARCH_SLOW = 3.0, 0.5  # at a smell fully lost: wander this many times wider, walk this much slower
# A need leaves a duck alone until it passes CONTENT_BELOW and has all of its attention by PRESSING_AT, a
# ramp and not a switch. Every place a like gives way to a need uses this, so a slightly peckish duck still
# swims, keeps company and shows its personality.
CONTENT_BELOW, PRESSING_AT = 0.4, 0.8
# How much of a walk boredom alone is worth, against a need or a want at 1. A bored duck ambles; at 1 every
# duck marched half its waking life, since boredom sits near full for most of a quiet day.
BORED_WANDER = 0.4
AMUSED = 0.3  # how much boredom a song or a dance takes off
PLAY_BORED, PLAY_TIRED = 0.25, 0.7  # boredom at which play starts to appeal, and fatigue at which it has stopped
REST_BELOW = 0.15  # a duck whose hunger, thirst and boredom are all under this has no reason to move
NIGHT_SLEEPINESS = 2.0  # sleep pressure builds this many times faster once the sun is down
DAY_WAKING = 1.0  # daylight cancels an ordinary duck's build entirely, so it stays up all day; a sleepy one still naps
AWAKE_S = 600.0  # awake time before sleep pressure is full, at sleepiness 0.5
SLEEP_S = 120.0  # asleep time to clear full sleep pressure
BODY_TEMP_TAU_S, WATER_C = 30.0, 18.0
COMFORT_C, COMFORT_BAND_C, COMFORT_SPAN_C = 24.0, 2.0, 4.0
BORED_S = 120.0  # uneventful time to full boredom, at boredom rate 0.5
LONELY_S = 60.0  # time without touching another duck before a sociable duck starts to mope
AGGR_TONE_MAX = 0.8  # pC1d/e input at full aggressiveness and full trigger; aIPg about 1.2 Hz at a dish
FOOD_NEAR_HALF = 0.5  # food odor at which "near food" is 0.5


def pressing(need):
    """How far a need has a duck's attention, 0 to 1."""
    return np.clip((np.asarray(need, float) - CONTENT_BELOW) / (PRESSING_AT - CONTENT_BELOW), 0, 1)


def _knob(personality, name, n):
    return np.broadcast_to(np.asarray(personality.get(name, KNOB_DEFAULT), float), n)


class Physiology:
    def __init__(self, n: int, personality: dict | None = None, hunger=0.5, thirst=0.5, provoked=0.0,
                 body_temp=COMFORT_C):
        self.k = {name: _knob(personality or {}, name, n) for name in KNOBS}
        full = lambda v: np.broadcast_to(np.asarray(v, float), n).copy()
        self.hunger, self.thirst = full(hunger), full(thirst)
        self.fatigue, self.sleep_pressure, self.boredom = full(0.0), full(0.0), full(0.0)
        self.body_temp = full(body_temp)
        self.asleep = np.zeros(n, bool)
        self.alone_s = np.zeros(n)
        self.swimming = np.zeros(n, bool)
        self.joy, self.fear, self.sorrow = full(0.0), full(0.0), full(0.0)
        self.anger = full(provoked)
        self.scent, self.scent_was = full(0.0), full(0.0)
        # what a duck has got good at, grown by doing it (brain/social.py); a save keeps them
        self.swim_skill, self.walk_skill = full(0.0), full(0.0)
        self.eat_skill, self.dance_skill, self.fight_skill = full(0.0), full(0.0), full(0.0)
        self.fashion_skill, self.music_skill = full(0.0), full(0.0)

    def step(self, dt: float, f, escaped, speed):
        """f: frame records (structured array); escaped and speed: per duck, from the last step."""
        k = self.k
        ate, drank, bumped, swimming = (f[name] > 0 for name in ("ate", "drank", "bumped", "swimming"))
        self.swimming = np.asarray(swimming)  # kept for what a duck shows of itself (brain/emotes.py)
        touching = (f["touch_left"] + f["touch_right"]) > 0
        ambient = (f["temp_left"] + f["temp_right"]) / 2
        escaped = np.asarray(escaped, bool)
        speed = np.abs(np.asarray(speed, float))

        self.scent = (f["odor_left"] + f["odor_right"]) / 2.0
        self.scent_was = self.scent_was + (self.scent - self.scent_was) * min(dt / SCENT_MEMORY_S, 1.0)
        # a practised eater gets more out of the same bite: it knows how to get the fruit down
        self.hunger = np.clip(self.hunger + dt / HUNGER_RISE_S * (0.5 + k["appetite"])
                              - BITE_FULLNESS * (1 + EAT_GROWS * self.eat_skill) * ate, 0, 1)
        self.thirst = np.clip(self.thirst + dt / THIRST_RISE_S - SIP_QUENCH * drank, 0, 1)
        tire = speed * dt / FATIGUE_M * (1.5 - k["energy"])
        rest = np.where(speed < 0.01, dt / REST_S, 0.0) * (1 + self.asleep)
        self.fatigue = np.clip(self.fatigue + tire - rest, 0, 1)
        # night builds sleep pressure faster; daylight wears it off
        light = np.asarray(f["light"], float)
        dark = 1.0 - light
        build = dt / AWAKE_S * (0.5 + k["sleepiness"]) * (1 + NIGHT_SLEEPINESS * dark)
        self.sleep_pressure = np.clip(
            self.sleep_pressure + np.where(self.asleep, -dt / SLEEP_S, build)
            - dt / AWAKE_S * DAY_WAKING * light, 0, 1)
        target = np.where(swimming, WATER_C, ambient)
        self.body_temp += (target - self.body_temp) * dt / BODY_TEMP_TAU_S

        # A song or a dance nearby entertains a sociable duck, puts off one that keeps to itself, and
        # provokes an aggressive one whatever it thinks of company.
        show = f["show"] > 0
        fan = np.clip(2 * k["sociability"] - 1, 0, 1)
        put_off = np.clip(1 - 2 * k["sociability"], 0, 1)
        cross = np.clip((k["aggressiveness"] - 0.6) / 0.4, 0, 1)
        self.boredom = np.clip(self.boredom - 0.7 * AMUSED * fan * (1 - cross) * show, 0, 1)
        kicked = (f["kicked"] > 0) | (f["drummed"] > 0)  # a ball kicked or a drum tapped: play either way
        eventful = ate | drank | bumped | escaped | touching | kicked
        self.boredom = np.clip(np.where(eventful, self.boredom - 0.2, self.boredom + dt / BORED_S * (0.5 + k["boredom_rate"])), 0, 1)
        self.alone_s = np.where(touching, 0.0, self.alone_s + dt)

        # emotions jump on events and fade on personality clocks (Chao: aggressiveness speeds Fear's fade and
        # slows Anger's; curiosity speeds Sorrow's)
        fade = lambda x, tau: x * np.exp(-dt / tau)
        self.joy = np.clip(fade(self.joy, 10.0) + 0.3 * (ate | drank) + 0.3 * k["playfulness"] * kicked
                           + 0.2 * fan * (1 - cross) * show, 0, 1)
        # being shoved frightens a timid duck and angers an aggressive one
        self.fear = np.clip(fade(self.fear, 3 + 15 * k["timidity"] + 5 * (1 - k["aggressiveness"]))
                            + 0.8 * escaped + bumped * k["timidity"], 0, 1)
        self.anger = np.clip(np.where(bumped, self.anger + k["aggressiveness"],
                                      fade(self.anger, 3 + 15 * k["aggressiveness"]))
                             + 0.25 * np.maximum(cross, 0.5 * put_off) * show, 0, 1)
        lonely = (self.alone_s > LONELY_S) & (k["sociability"] > 0.5)
        mope = dt / 30.0 * lonely * (k["sociability"] - 0.5) * 2 + 0.3 * bumped * k["timidity"]
        self.sorrow = np.clip(fade(self.sorrow, 5 + 20 * (1 - k["curiosity"])) + mope, 0, 1)

        falls_asleep = ~self.asleep & (self.sleep_pressure >= 1) & (self.fear < 0.2)
        wakes = self.asleep & ((self.sleep_pressure <= 0.1) | bumped | escaped)
        self.asleep = (self.asleep | falls_asleep) & ~wakes
        return falls_asleep, wakes

    def at_water(self, humidity):
        """How close to water the air says it is, 0 to 1: past half saturated the pond is a step or two
        away and the humidity neurons' own left and right do the rest."""
        return np.clip((humidity - 0.5) / 0.5, 0, 1)

    def amuse(self, i: int) -> None:
        """Duck i has just sung, danced or played: that is something happening, which is what boredom wants."""
        self.boredom[i] = max(self.boredom[i] - AMUSED, 0.0)
        self.joy[i] = min(self.joy[i] + 0.1, 1.0)

    def play(self) -> np.ndarray:
        """How much a duck wants to play, 0 to 1: a playful duck, bored, and not worn out. It is a like, so
        it gives way as a need presses, as music and company do (brain/server.py `at_ease`)."""
        bored = np.clip((self.boredom - PLAY_BORED) / (1 - PLAY_BORED), 0, 1)
        rested = np.clip(1 - self.fatigue / PLAY_TIRED, 0, 1)
        return self.k["playfulness"] * bored * rested * ~self.asleep

    def discomfort(self):
        """(too hot, too cold) in 0-1, with a comfort band widened by heat tolerance on the hot side."""
        hot_edge = COMFORT_C + COMFORT_BAND_C + 4 * (self.k["heat_tolerance"] - 0.5)
        hot = np.clip((self.body_temp - hot_edge) / COMFORT_SPAN_C, 0, 1)
        cold = np.clip((COMFORT_C - COMFORT_BAND_C - self.body_temp) / COMFORT_SPAN_C, 0, 1)
        return hot, cold

    def cooling(self) -> tuple[np.ndarray, np.ndarray]:
        """How much a hot duck wants the pond and the shade, each 0 to 1. Water love picks which, and like a
        swim it gives way to hunger."""
        hot = self.discomfort()[0] * (1 - 0.8 * pressing(self.hunger)) * ~self.asleep
        return hot * self.k["water_love"], hot * (1 - self.k["water_love"])

    def sense_gains(self) -> dict[str, np.ndarray]:
        """Multipliers on encoder levels, by input set name (sides share a gain)."""
        k = self.k
        hot, cold = self.discomfort()
        # A frightened duck goes off its food, which is what lets one duck drive another off a dish.
        # Hunger and thirst share one set of steering neurons, so each need's senses are scaled by its
        # share of the two: 1.0 apiece when level, 2.0 and 0 when one is everything.
        need = self.hunger + self.thirst + 1e-9
        hungry, thirsty = 2 * self.hunger / need, 2 * self.thirst / need
        food = (0.5 + self.hunger) * (0.75 + 0.5 * k["appetite"]) * (1 - 0.6 * self.fear) * hungry
        # A hot duck cools off in the pond or under the tree, and water love picks which, so the two
        # never pull against each other. Thirst is a need; liking water and cooling off in it are likes,
        # and yield to hunger as the swim urge does.
        hungry_now = pressing(self.hunger)
        water = (self.thirst * thirsty + hot * k["water_love"] * (1 - 0.8 * hungry_now)
                 + 0.5 * k["water_love"] * (1 - hungry_now))
        # Companionship is a like: sociability sets how much another duck's smell draws this one,
        # and it fades as hunger or thirst press, or a huddle starves together.
        companionship = 2 * k["sociability"] * (1 - pressing(np.maximum(self.hunger, self.thirst)))
        care = 1 - 0.6 * k["carelessness"]
        # cold sensors steer toward cold, so a hot duck turns up its cold sense to find shade, and a cold
        # duck its heat sense; a water lover skips the shade and heads for the pond
        return {
            # Smell keeps a floor, because a full duck still notices a dish. Taste has none: tasting sets
            # the feeding intent, which stops a duck where it stands, so a full duck must not taste food.
            "orn_food": food,
            "sugar": self.hunger * (0.75 + 0.5 * k["appetite"]),
            # likewise water is only worth tasting when thirsty, or a sated duck parks at the shore to "eat"
            "water_taste": self.thirst,
            "moist_air": water,
            "cold": 1 + 2 * hot * (1 - k["water_love"]), "heat": 1 + 2 * cold,
            "orn_danger": care,
            "orn_pheromone": companionship,
            "vision": (0.5 + k["timidity"]) * care * (1 + self.fear),  # brain/vision.py, not a set
        }

    def motor(self, wants=0.0, damp=0.0) -> dict[str, np.ndarray]:
        """How the body shapes movement: speed and wander scales, and whether the duck approaches touches."""
        k = self.k
        # how much of the smell it was following is gone, 0 to 1, weighed by wanting it
        lost = np.clip(1 - self.scent / np.maximum(self.scent_was, 1e-6), 0, 1) * (self.scent_was > SCENT_FLOOR)
        lost = lost * pressing(self.hunger)
        swim_urge = np.clip(k["water_love"] * (1 + self.discomfort()[0]) * (1 - 0.8 * pressing(self.hunger)), 0, 1)
        return {
            "speed": ((0.4 + 1.2 * k["energy"]) * (1 - 0.6 * self.fatigue) * (1 - 0.5 * self.sorrow)
                      * (1 - SEARCH_SLOW * lost)),
            "wander": (0.5 + k["curiosity"]) * (1 + self.boredom) * (1 + (SEARCH_TURN - 1) * lost),
            "zoomies": (self.boredom > 0.8) & (k["energy"] > 0.6) & (self.fatigue < 0.3),
            "social": np.clip((k["sociability"] - 0.5) * 2, 0, 1),
            "asleep": self.asleep,
            # Heat multiplies a duck's taste for water rather than adding to it, so the knob still acts
            # when hot. Hunger gets a duck out of the pond, where it cannot eat.
            "swim_urge": swim_urge,
            "swim_thirst_weight": 1 - 0.5 * k["water_love"],  # water lovers wade in even a little thirsty
            # Whether a duck has any reason to be walking; with none it stands, and rests. Needs count once
            # they press. `wants` is anything else pulling at it, 0 to 1, that the server knows and the body
            # does not (music, to a duck that loves or hates it). A swim counts for a water lover that can
            # tell water is near (`damp` is how humid the air is). Boredom is what eventually gets it going.
            "restlessness": np.maximum(
                pressing(np.maximum(self.hunger, self.thirst)),
                np.clip((np.maximum.reduce([BORED_WANDER * self.boredom, np.broadcast_to(wants, self.hunger.shape),
                                            swim_urge * damp]) - REST_BELOW) / 0.5, 0, 1)),
        }


def aggression_tone(aggressiveness, hunger, food_odor, provoked, kindness=KNOB_DEFAULT):
    """pC1d/e input level per duck."""
    near_food = food_odor / (food_odor + FOOD_NEAR_HALF)
    calm = 1 - np.maximum(np.asarray(kindness) - KNOB_DEFAULT, 0)
    return AGGR_TONE_MAX * aggressiveness * calm * np.clip(hunger * near_food + provoked, 0, 1)
