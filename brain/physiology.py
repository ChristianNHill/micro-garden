"""Bodily state per duck: drives and Chao-style emotions, shaped by personality knobs (PLAN.md Gates 4c, 5).

Drives rise on their own clocks and are satisfied by world events; emotions jump on events and fade.
Following the Chao games (RESEARCH.md §9), knobs mostly set those rates. The body acts on the brain by
scaling sensory gains and adding mood input (`sense_gains`, `aggression_tone`); it also shapes how the
decoder moves the duck (`motor`).

Aggression is a mood, not a reflex: a tonic input into pC1d/e raises aIPg, and touch then turns that
into an attack (decoder). The mood is personality times a trigger: hunger near food, or anger from
being headbutted (Chris, 2026-09-16).
"""
import numpy as np

from brain.personality import KNOB_DEFAULT, KNOBS

# From just fed to fully hungry, at appetite 0.5: one garden day (world/fields.py DAY_S), the scale sleep
# already runs on. At half a day, with thirst to match, five ducks needed about 200 bites and 150 sips in
# twenty minutes and a body that walks 0.06 m/s between a dish and a pond 2.4 m apart can manage about 80
# of each: the soak had every duck starving or parched most of the time however well it found its way
# (Gate 9b; Claude's call while Chris was out, 2026-09-19, for him to ratify).
HUNGER_RISE_S = 600.0
BITE_FULLNESS = 0.1
THIRST_RISE_S = 800.0  # kept at four thirds of hunger's
SIP_QUENCH = 0.1
FATIGUE_M = 30.0  # metres of walking from rested to exhausted, at energy 0.5
REST_S = 60.0  # standing still from exhausted to rested
# Losing a smell. A fly that walks out of an odor it was following slows down and turns hard for a few
# seconds, a local search that puts it back in the plume (Alvarez-Salvado et al. 2018, the OFF response).
# Without it a duck walks up the wind in a straight line and straight past a dish a hand's width to one
# side, because going upwind never corrects sideways (2026-09-19: 8 of 8 aligned upwind, 1 of 8 arrived).
# How long the smell it had stays with it, and so how long it searches for one it lost. A walking fly keeps
# a local search going for tens of seconds after losing a food cue, not five: at 5 s a duck that overshot
# the dish gave up and walked off before it had turned round (Gate 4, 13/20 found; at 30 s, 17/20).
SCENT_MEMORY_S = 30.0
# A duck can only lose a smell it had. Below this its memory of one counts as none: without the floor a
# duck that had never smelled anything read as having lost everything, and searched, at half speed and
# turning three times as wide, for as long as there was nothing to smell, which is every garden without
# food in it (found at Gate 8b, 2026-09-20). 0.05 is a tenth of the smell sense's half level.
SCENT_FLOOR = 0.05
# There was also a search for when the smell was strong, "close to food", set on the smell of one dish. Four
# fruits' plumes add up, so it read "close" over 19 to 45% of the garden, out to 2.7 m downwind, and ducks
# that far off stopped following the wind and dithered; and with one dish Gate 4 does better without it, 20/20
# at 27.6 s against 17/20 at 38.4 s, because the search for a lost smell already covers the last step. Removed
# 2026-09-20.
SEARCH_TURN, SEARCH_SLOW = 3.0, 0.5  # at a smell fully lost: wander this many times wider, walk this much slower
# A duck does not have to be full to be happy (Chris, 2026-09-20). A need leaves a duck alone until it
# passes CONTENT_BELOW and has all of its attention by PRESSING_AT, a ramp and not a switch. Every place a
# like gives way to a need uses this, so a duck that is a bit peckish still swims, keeps company and
# dawdles in a smell it likes, and its personality has most of the day to show in. It used to give way in
# proportion to hunger from zero: at hunger 0.4 a water lover's love of water was already down a third.
CONTENT_BELOW, PRESSING_AT = 0.4, 0.8
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
        self.joy, self.fear, self.sorrow = full(0.0), full(0.0), full(0.0)
        self.anger = full(provoked)
        self.scent, self.scent_was = full(0.0), full(0.0)

    def step(self, dt: float, f, escaped, speed):
        """f: frame records (structured array); escaped and speed: per duck, from the last step."""
        k = self.k
        ate, drank, bumped, swimming = (f[name] > 0 for name in ("ate", "drank", "bumped", "swimming"))
        touching = (f["touch_left"] + f["touch_right"]) > 0
        ambient = (f["temp_left"] + f["temp_right"]) / 2
        escaped = np.asarray(escaped, bool)
        speed = np.abs(np.asarray(speed, float))

        self.scent = (f["odor_left"] + f["odor_right"]) / 2.0
        self.scent_was = self.scent_was + (self.scent - self.scent_was) * min(dt / SCENT_MEMORY_S, 1.0)
        self.hunger = np.clip(self.hunger + dt / HUNGER_RISE_S * (0.5 + k["appetite"]) - BITE_FULLNESS * ate, 0, 1)
        self.thirst = np.clip(self.thirst + dt / THIRST_RISE_S - SIP_QUENCH * drank, 0, 1)
        tire = speed * dt / FATIGUE_M * (1.5 - k["energy"])
        rest = np.where(speed < 0.01, dt / REST_S, 0.0) * (1 + self.asleep)
        self.fatigue = np.clip(self.fatigue + tire - rest, 0, 1)
        # Night is what makes a duck sleepy; daylight is what wears the sleepiness off. Without this a
        # duck simply ran down like a clock and slept whenever its 10 minutes were up (PLAN.md Gate 3).
        light = np.asarray(f["light"], float)
        dark = 1.0 - light
        build = dt / AWAKE_S * (0.5 + k["sleepiness"]) * (1 + NIGHT_SLEEPINESS * dark)
        self.sleep_pressure = np.clip(
            self.sleep_pressure + np.where(self.asleep, -dt / SLEEP_S, build)
            - dt / AWAKE_S * DAY_WAKING * light, 0, 1)
        target = np.where(swimming, WATER_C, ambient)
        self.body_temp += (target - self.body_temp) * dt / BODY_TEMP_TAU_S

        eventful = ate | drank | bumped | escaped | touching
        self.boredom = np.clip(np.where(eventful, self.boredom - 0.2, self.boredom + dt / BORED_S * (0.5 + k["boredom_rate"])), 0, 1)
        self.alone_s = np.where(touching, 0.0, self.alone_s + dt)

        # emotions jump on events and fade on personality clocks (Chao: aggressiveness speeds Fear's fade and
        # slows Anger's; curiosity speeds Sorrow's)
        fade = lambda x, tau: x * np.exp(-dt / tau)
        self.joy = np.clip(fade(self.joy, 10.0) + 0.3 * (ate | drank), 0, 1)
        # Being shoved frightens a timid duck and angers an aggressive one. It used to anger every duck
        # alike, so a Scaredy walked straight back to the dish it had just been driven off and a Bully
        # displaced nobody (Gate 8, 2026-09-18).
        self.fear = np.clip(fade(self.fear, 3 + 15 * k["timidity"] + 5 * (1 - k["aggressiveness"]))
                            + 0.8 * escaped + bumped * k["timidity"], 0, 1)
        self.anger = np.clip(np.where(bumped, self.anger + k["aggressiveness"],
                                      fade(self.anger, 3 + 15 * k["aggressiveness"])), 0, 1)
        lonely = (self.alone_s > LONELY_S) & (k["sociability"] > 0.5)
        mope = dt / 30.0 * lonely * (k["sociability"] - 0.5) * 2 + 0.3 * bumped * k["timidity"]
        self.sorrow = np.clip(fade(self.sorrow, 5 + 20 * (1 - k["curiosity"])) + mope, 0, 1)

        falls_asleep = ~self.asleep & (self.sleep_pressure >= 1) & (self.fear < 0.2)
        wakes = self.asleep & ((self.sleep_pressure <= 0.1) | bumped | escaped)
        self.asleep = (self.asleep | falls_asleep) & ~wakes
        return falls_asleep, wakes

    def at_water(self, humidity):
        """How close to water the air says it is, 0 to 1: past half saturated the pond is a step or two
        away and the humidity neurons' own left and right do the rest (Gate 4b)."""
        return np.clip((humidity - 0.5) / 0.5, 0, 1)

    def discomfort(self):
        """(too hot, too cold) in 0-1, with a comfort band widened by heat tolerance on the hot side."""
        hot_edge = COMFORT_C + COMFORT_BAND_C + 4 * (self.k["heat_tolerance"] - 0.5)
        hot = np.clip((self.body_temp - hot_edge) / COMFORT_SPAN_C, 0, 1)
        cold = np.clip((COMFORT_C - COMFORT_BAND_C - self.body_temp) / COMFORT_SPAN_C, 0, 1)
        return hot, cold

    def sense_gains(self) -> dict[str, np.ndarray]:
        """Multipliers on encoder levels, by input set name (sides share a gain)."""
        k = self.k
        hot, cold = self.discomfort()
        # A frightened duck goes off its food, which is what lets one duck drive another off a dish
        # Hunger and thirst share one set of steering neurons, so they are weighed against each other
        # before they get there: each need's senses are scaled by its share of the two, 1.0 apiece when
        # they are level and up to 2.0 and 0 when one is everything. Unweighed, a starving duck with a
        # little thirst steered by humid air as hard as ever, and five of them spent two days at the pond
        # averaging 2 m from food that sat uneaten (Gate 9b, 2026-09-19).
        need = self.hunger + self.thirst + 1e-9
        hungry, thirsty = 2 * self.hunger / need, 2 * self.thirst / need
        food = (0.5 + self.hunger) * (0.75 + 0.5 * k["appetite"]) * (1 - 0.6 * self.fear) * hungry
        # A hot duck has two ways to cool down and water love decides which it reaches for: one wades
        # in, another sits under the tree. Both are what a duck does, so it is a knob and not a bug
        # (Chris, 2026-09-18). It also stops the two pulling against each other, which is what made a
        # heat-intolerant duck walk away from a sunny pond once it could see the water (Gate 5).
        # Thirst and heat pull a duck to water because it needs to be there. Liking water is a
        # pleasure, and hunger outranks a pleasure: the gain used to floor at 0.25 + half the knob
        # whatever else was true, so a duck with nothing to drink and everything to eat still read the
        # pond louder than a dish two metres off and never went (audit, 2026-09-19).
        # Cooling off in the pond is a like too, and yields to hunger by the factor the swim urge does:
        # ungated, a hot water lover followed damp air while it starved (Gate 9b, Carefree).
        hungry_now = pressing(self.hunger)
        water = (self.thirst * thirsty + hot * k["water_love"] * (1 - 0.8 * hungry_now)
                 + 0.5 * k["water_love"] * (1 - hungry_now))
        # Companionship is a like, not a need. Sociability sets how much another duck's smell draws this one
        # (ARCHITECTURE.md 2.4; until now no knob touched it and every duck was drawn alike), and it
        # fades once hunger or thirst passes halfway, or a huddle holds itself together while it starves.
        companionship = 2 * k["sociability"] * (1 - pressing(np.maximum(self.hunger, self.thirst)))
        care = 1 - 0.6 * k["carelessness"]
        # cold sensors steer toward cold (Gate 4b probe), so a hot duck turns up its cold sense to find
        # shade, and a cold duck its heat sense; a water lover skips the shade and heads for the pond
        return {
            # Smell keeps a floor, because a full duck still notices a dish. Taste does not: at
            # (0.5 + hunger) a duck with no appetite at all still fed 96% of ticks and took 58 bites,
            # exactly as many as a starving one, because tasting food sets the feeding intent and the
            # feeding intent stops a duck where it stands (audit, 2026-09-19).
            "orn_food": food,
            "sugar": self.hunger * (0.75 + 0.5 * k["appetite"]),
            # Water is only worth tasting when a duck is thirsty. At 0.5 + thirst a sated duck still
            # tasted the shore as food, which fired its proboscis neurons, which set the feeding intent,
            # which stops a duck where it stands: five hungry ducks parked at the pond and never went to
            # eat (found by watching the sim, 2026-09-19; no gate puts a pond and a dish in one garden).
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
            # Heat multiplies a duck's taste for water rather than standing in for it. Added, heat
            # alone pinned the urge at 1.0 for every duck, so one that hates water waded in exactly as
            # readily as one that loves it and the knob could not act at all (Gate 5, 2026-09-18).
            # Hunger gets a duck out of the pond. Without it a water lover paddles at a fifth of its
            # speed, cannot eat while swimming, and has only a small chance every few seconds of
            # choosing to leave, so it can sit there and starve (audit, 2026-09-19).
            "swim_urge": swim_urge,
            "swim_thirst_weight": 1 - 0.5 * k["water_love"],  # water lovers wade in even a little thirsty
            # Whether a duck has any reason to be walking. BASE_VX used to be added unconditionally,
            # from before the ducks had eyes and anything reached DNp09, so one could never stand still:
            # fatigue only ever climbed, because resting needs speed under 0.01 (Chris, watching the
            # sim, 2026-09-19). A duck with nothing it wants stands about instead, and boredom is what
            # eventually gets it going again.
            # `wants` is anything else pulling at it, 0 to 1, which the server knows and the body does
            # not: music, to a duck that loves or hates it. Without it a fed, watered duck stood and
            # listened from across the garden, liking the tune or not (Gate 8b, 2026-09-20).
            # A swim is a want too, for a duck that likes water and can tell there is some about (`damp`
            # is how humid the air is): a fed, watered water lover stood on the shore until it got bored.
            "restlessness": np.clip((np.maximum.reduce([self.hunger, self.thirst, self.boredom,
                                                        np.broadcast_to(wants, self.hunger.shape),
                                                        swim_urge * damp])
                                     - REST_BELOW) / 0.5, 0, 1),
        }


def aggression_tone(aggressiveness, hunger, food_odor, provoked, kindness=KNOB_DEFAULT):
    """pC1d/e input level per duck."""
    near_food = food_odor / (food_odor + FOOD_NEAR_HALF)
    calm = 1 - np.maximum(np.asarray(kindness) - KNOB_DEFAULT, 0)
    return AGGR_TONE_MAX * aggressiveness * calm * np.clip(hunger * near_food + provoked, 0, 1)
