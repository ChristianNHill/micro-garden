"""Bodily state per duck. Gate 5 grows this into the full drive set; Gate 4c needs hunger and provocation.

Aggression is a mood, not a reflex: a tonic input into pC1d/e (persistent social arousal) raises aIPg,
and touch then turns that into an attack (decoder). The mood is personality times a trigger: a hungry
duck near food defends it, and a duck that was just headbutted is provoked (Chris, 2026-09-16).
"""
import numpy as np

HUNGER_RISE_S = 300.0  # from just fed to fully hungry
BITE_FULLNESS = 0.1  # hunger removed per bite
PROVOKE_TAU_S = 10.0
AGGR_TONE_MAX = 0.8  # pC1d/e input at full aggressiveness and full trigger; aIPg about 1.2 Hz at a dish
FOOD_NEAR_HALF = 0.5  # food odor at which "near food" is 0.5


class Physiology:
    def __init__(self, n: int, hunger=0.0, provoked=0.0):
        self.hunger = np.broadcast_to(np.asarray(hunger, float), n).copy()
        self.provoked = np.broadcast_to(np.asarray(provoked, float), n).copy()

    def step(self, dt: float, ate: np.ndarray, bumped: np.ndarray) -> None:
        self.hunger = np.clip(self.hunger + dt / HUNGER_RISE_S - BITE_FULLNESS * ate, 0.0, 1.0)
        self.provoked = np.where(bumped, 1.0, self.provoked * np.exp(-dt / PROVOKE_TAU_S))


def aggression_tone(aggressiveness, hunger, food_odor, provoked):
    """pC1d/e input level per duck."""
    near_food = food_odor / (food_odor + FOOD_NEAR_HALF)
    return AGGR_TONE_MAX * aggressiveness * np.clip(hunger * near_food + provoked, 0, 1)
