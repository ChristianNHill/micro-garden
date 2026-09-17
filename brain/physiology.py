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

HUNGER_RISE_S = 300.0  # from just fed to fully hungry, at appetite 0.5
BITE_FULLNESS = 0.1
THIRST_RISE_S = 400.0
SIP_QUENCH = 0.1
FATIGUE_M = 30.0  # metres of walking from rested to exhausted, at energy 0.5
REST_S = 60.0  # standing still from exhausted to rested
AWAKE_S = 600.0  # awake time before sleep pressure is full, at sleepiness 0.5
SLEEP_S = 120.0  # asleep time to clear full sleep pressure
BODY_TEMP_TAU_S, WATER_C = 30.0, 18.0
COMFORT_C, COMFORT_BAND_C, COMFORT_SPAN_C = 24.0, 2.0, 4.0
BORED_S = 120.0  # uneventful time to full boredom, at boredom rate 0.5
LONELY_S = 60.0  # time without touching another duck before a sociable duck starts to mope
AGGR_TONE_MAX = 0.8  # pC1d/e input at full aggressiveness and full trigger; aIPg about 1.2 Hz at a dish
FOOD_NEAR_HALF = 0.5  # food odor at which "near food" is 0.5

KNOB_DEFAULT = 0.5


def _knob(personality, name, n):
    return np.broadcast_to(np.asarray(personality.get(name, KNOB_DEFAULT), float), n)


class Physiology:
    def __init__(self, n: int, personality: dict | None = None, hunger=0.5, thirst=0.5, provoked=0.0,
                 body_temp=COMFORT_C):
        self.n = n
        self.k = {name: _knob(personality or {}, name, n) for name in (
            "aggressiveness", "timidity", "curiosity", "sociability", "kindness", "appetite", "energy",
            "sleepiness", "heat_tolerance", "water_love", "boredom_rate", "carelessness")}
        full = lambda v: np.broadcast_to(np.asarray(v, float), n).copy()
        self.hunger, self.thirst = full(hunger), full(thirst)
        self.fatigue, self.sleep_pressure, self.boredom = full(0.0), full(0.0), full(0.0)
        self.body_temp = full(body_temp)
        self.asleep = np.zeros(n, bool)
        self.alone_s = np.zeros(n)
        self.joy, self.fear, self.sorrow = full(0.0), full(0.0), full(0.0)
        self.anger = full(provoked)  # "provoked" before Gate 5

    @property
    def provoked(self):
        return self.anger

    def step(self, dt: float, f=None, ate=None, bumped=None, escaped=None, speed=None):
        """f: frame records (structured array). The keyword arrays let callers without frames step too."""
        k, n = self.k, self.n
        zeros = np.zeros(n)
        get = lambda name, alt: (f[name] if f is not None else alt)
        ate = get("ate", zeros if ate is None else ate) > 0
        drank = get("drank", zeros) > 0
        bumped = get("bumped", zeros if bumped is None else bumped) > 0
        touching = (get("touch_left", zeros) + get("touch_right", zeros)) > 0
        swimming = get("swimming", zeros) > 0
        ambient = (get("temp_left", self.body_temp) + get("temp_right", self.body_temp)) / 2
        escaped = zeros > 0 if escaped is None else np.asarray(escaped, bool)
        speed = zeros if speed is None else np.abs(np.asarray(speed, float))

        self.hunger = np.clip(self.hunger + dt / HUNGER_RISE_S * (0.5 + k["appetite"]) - BITE_FULLNESS * ate, 0, 1)
        self.thirst = np.clip(self.thirst + dt / THIRST_RISE_S - SIP_QUENCH * drank, 0, 1)
        tire = speed * dt / FATIGUE_M * (1.5 - k["energy"])
        rest = np.where(speed < 0.01, dt / REST_S, 0.0) * (1 + self.asleep)
        self.fatigue = np.clip(self.fatigue + tire - rest, 0, 1)
        self.sleep_pressure = np.clip(
            self.sleep_pressure + np.where(self.asleep, -dt / SLEEP_S, dt / AWAKE_S * (0.5 + k["sleepiness"])), 0, 1)
        target = np.where(swimming, WATER_C, ambient)
        self.body_temp += (target - self.body_temp) * dt / BODY_TEMP_TAU_S

        eventful = ate | drank | bumped | escaped | touching
        self.boredom = np.clip(np.where(eventful, self.boredom - 0.2, self.boredom + dt / BORED_S * (0.5 + k["boredom_rate"])), 0, 1)
        self.alone_s = np.where(touching, 0.0, self.alone_s + dt)

        # emotions jump on events and fade on personality clocks (Chao: aggressiveness speeds Fear's fade and
        # slows Anger's; curiosity speeds Sorrow's)
        fade = lambda x, tau: x * np.exp(-dt / tau)
        self.joy = np.clip(fade(self.joy, 10.0) + 0.3 * (ate | drank), 0, 1)
        self.fear = np.clip(fade(self.fear, 3 + 15 * k["timidity"] + 5 * (1 - k["aggressiveness"])) + 0.8 * escaped, 0, 1)
        self.anger = np.where(bumped, 1.0, fade(self.anger, 3 + 15 * k["aggressiveness"]))
        lonely = (self.alone_s > LONELY_S) & (k["sociability"] > 0.5)
        mope = dt / 30.0 * lonely * (k["sociability"] - 0.5) * 2 + 0.3 * bumped * k["timidity"]
        self.sorrow = np.clip(fade(self.sorrow, 5 + 20 * (1 - k["curiosity"])) + mope, 0, 1)

        falls_asleep = ~self.asleep & (self.sleep_pressure >= 1) & (self.fear < 0.2)
        wakes = self.asleep & ((self.sleep_pressure <= 0.1) | bumped | escaped)
        self.asleep = (self.asleep | falls_asleep) & ~wakes
        return falls_asleep, wakes

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
        food = (0.5 + self.hunger) * (0.75 + 0.5 * k["appetite"])
        # 1.0 for a comfortable thirst-0.5, love-0.5 duck; heat pulls toward water whatever the love
        water = 0.5 + self.thirst + k["water_love"] * 0.5 - 0.25 + hot
        care = 1 - 0.6 * k["carelessness"]
        # cold sensors steer toward cold (Gate 4b probe), so a hot duck turns up its cold sense to find
        # shade, and a cold duck its heat sense
        return {
            "orn_food": food, "sugar": food,  # sugar and water share the sugar/water taste neurons
            "water_taste": 0.5 + self.thirst,
            "moist_air": water,
            "cold": 1 + 2 * hot, "heat": 1 + 2 * cold,
            "orn_danger": care,
            "LPLC2": (0.5 + k["timidity"]) * care * (1 + self.fear),
        }

    def motor(self) -> dict[str, np.ndarray]:
        """How the body shapes movement: speed and wander scales, and whether the duck approaches touches."""
        k = self.k
        return {
            "speed": (0.4 + 1.2 * k["energy"]) * (1 - 0.6 * self.fatigue) * (1 - 0.5 * self.sorrow),
            "wander": (0.5 + k["curiosity"]) * (1 + self.boredom),
            "zoomies": (self.boredom > 0.8) & (k["energy"] > 0.6) & (self.fatigue < 0.3),
            "social": np.clip((k["sociability"] - 0.5) * 2, 0, 1),
            "asleep": self.asleep,
            "swim_urge": np.clip(k["water_love"] + self.discomfort()[0], 0, 1),
            "swim_thirst_weight": 1 - 0.5 * k["water_love"],  # water lovers wade in even a little thirsty
        }


def aggression_tone(aggressiveness, hunger, food_odor, provoked, kindness=KNOB_DEFAULT):
    """pC1d/e input level per duck."""
    near_food = food_odor / (food_odor + FOOD_NEAR_HALF)
    calm = 1 - 0.5 * (np.asarray(kindness) - KNOB_DEFAULT) * 2 * (np.asarray(kindness) > KNOB_DEFAULT)
    return AGGR_TONE_MAX * aggressiveness * calm * np.clip(hunger * near_food + provoked, 0, 1)
