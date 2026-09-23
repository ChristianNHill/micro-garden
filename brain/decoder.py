"""Motor decoder: descending-neuron spikes -> smoothed rates -> microduck intent.

- DNp09 drives forward, moonwalker (MDN) backward.
- DNa02, the odor-steering DNs and DNp12/DNp44 (humidity) turn toward their own side. DNg48 fires
  opposite a touch, so read the same way a touched duck turns away.
- ESCAPE_SPIKES giant fiber spikes within ESCAPE_WINDOW ticks start an escape (the giant fiber idles, so it takes a burst).
- Proboscis motor neurons mean feeding.
- DNp32 fires for stink, more on the stink's side. A stink-averse duck bolts: it runs and turns gently
  away from the busier DNp32 (strong turning made ducks circle in the stink). A stink-loving duck slows
  and turns gently toward it. On meeting a stink a duck picks one, lingering with probability
  stink_affinity, and keeps it until the stink is gone, so the knob is a scale. This is an explicit
  per-duck readout choice, not the brain.
- The body (brain/physiology.py `motor`, set by the server each body step in `self.body`) scales speed
  and spontaneous wandering, triggers zoomies, stops an asleep duck, and makes sociable ducks turn
  toward touch. At the shore a duck drinks or wades in, chosen on arrival and every WADE_REROLL_TICKS
  after, with a chance that rises with swim urge and falls with thirst. A swimming duck paddles nearly
  in place; one that chose not to walks back out.
- DNge091 fires on the side the wind comes from and turns the duck into it; the body only lets it hear
  the wind while it smells food it wants, which is how a fly finds food.
- Explicit: a frightened duck turns away from the other ducks and runs; an aggressive one (aIPg) turns
  after them. Both ride the explicit turn toward other ducks' smell that companionship uses.
- aIPg is aggression (its mood input is set in physiology). While it is active the touch turn flips
  toward the other duck, and a touching duck attacks (runs at it and headbutts) with a chance per tick
  that scales with aggression.

Ducks walk at BASE_VX (scaled by the body's restlessness) and the brain can push them up to RUN_VX.
Spontaneous turning (WANDER_*) is the same for real and shuffled brains, so any difference between
them comes from the brain.
"""
import numpy as np
import pandas as pd
import torch

from brain.lif import DEVICE, DT_MS

TAU_MS = 100.0
BASE_VX, RUN_VX = 0.11, 0.3  # m/s: an amble that reads as walking on screen, and a run
VX_PER_HZ = 0.02  # m/s per Hz of forward minus backward drive
VYAW_PER_HZ = 1.0  # rad/s per Hz of left minus right steering DNs, positive = turn left
ESCAPE_TICKS = 50  # 0.5 s of running backward
# An escape is 5 giant fiber spikes in 300 ms. The giant fiber idles at about 2 Hz, so a looser bar
# (3 in 100 ms) fires by chance many times an hour. A clap (brain/server.py CLAP_S) puts 7 to 9 in the window.
ESCAPE_SPIKES, ESCAPE_WINDOW = 5, 30
WANDER_VYAW, WANDER_TICKS = 0.5, 50  # spontaneous turn rate, redrawn every 0.5 s
FEED_HZ = 1.0  # proboscis MN rate that means "eat"
# DNp32 is one neuron per side: a stray spike adds 0.5 Hz to its 2 s average, stink holds it near 1 Hz.
STINK_FLOOR_HZ, STINK_FULL_HZ, STINK_TAU_MS = 0.3, 0.8, 2000.0
STINK_VYAW_PER_HZ = 0.5  # rad/s per Hz of left minus right DNp32
# Companionship, explicit like music: a duck turns toward or away from the side the other ducks smell
# stronger on, by sociability and `fondness` (brain/plasticity.py). The brain's own left-right for a smell
# is a fraction of a hertz under 1.5 Hz of steering noise.
COMPANIONSHIP_VYAW = 1.5
# How far fear counts towards running from the other ducks, and aggression towards turning after them.
# At 0 they no longer steer a duck by the others: the control to measure them against.
FLEE, CHASE = 1.0, 1.0
# How much faster a fully practised duck is, in water and on land. On land it is the walk and the run alike:
# a practised walker ambles quicker and flees or charges quicker when it is scared or angry.
SWIM_GROWS, WALK_GROWS = 0.6, 0.4
BOND_VYAW, HAND_VYAW, COMFORT_VYAW = 1.5, 1.5, 2.0  # rad/s at full bond, full trust, and full kindness at a cry beside it
ROOM_FROM, ROOM_GONE = 0.5, 0.65  # `near_*` summed (1 - metres apart): company stops pulling from 0.5 m to 0.35 m
CROWDED_VYAW = 1.5  # rad/s aside from a duck nearer than that
INTENT_VYAW = 2.0  # rad/s towards a duck it has set out to comfort or to shove (brain/reactions.py)
COOL_VYAW = 2.0  # rad/s towards the eye its relief is in, at full heat
PLAY_VYAW = 2.0  # rad/s towards the eye a ball is in, at full wish to play
PURSUE_VX = 0.2  # m/s an angry duck closes on another at, short of a run, which overshoots
MUSIC_VYAW = 1.5  # rad/s toward the louder ear at full music affinity, and away from it at none
GROOM_HZ = 0.4  # grooming DN rate at which a duck may shed a hat
PREEN_REROLL_TICKS = 500  # a hatted duck reconsiders its hat every 5 s
PREEN_P = 0.3  # chance of shedding at each of those, at no vanity at all; a scale, not a threshold
FEAR_STOPS_FEEDING = 0.5  # fear at which a duck goes off its food
STINK_LINGER = 0.5  # a stink lover slows to this fraction of its speed in the stink
# aIPg mean rate; silent without the mood input. The full-scale rate is what the aggressiveness knob
# reaches at 1.0, measured: 0.0, 0.95, 2.63, 3.47, 4.37 Hz across the dial.
AGGR_FLOOR_HZ, AGGR_FULL_HZ, AGGR_TAU_MS = 0.05, 4.4, 1000.0
WADE_REROLL_TICKS = 500  # a duck at the shore reconsiders wading in every 5 s
# Chance per tick of a headbutt while touching, at full aggression (graded, not a threshold). At 0.01 a
# fully aggressive duck strikes about once a second and the dial spreads (20.0, 4.8, 1.7, 1.3, 1.0 s).
ATTACK_P = 0.01
TOUCH_HZ = 1.0  # DNg48 left plus right rate that means another duck is touching

FWD, BACK, STEER_L, STEER_R, GF, FEED, STINK_L, STINK_R, TOUCH_L, TOUCH_R, AIPG, GROOM, WIND_L, WIND_R = range(14)


class Decoder:
    def __init__(self, ann: pd.DataFrame, sets: dict[str, np.ndarray], batch: int, seed: int = 0,
                 stink_affinity=0.0):
        side = ann["side"].to_numpy()

        def by_side(ix):
            return ix[side[ix] == "left"], ix[side[ix] == "right"]

        steer = np.concatenate([sets["DNa02"], sets["odor_steer"], sets["moist_steer"]])
        members = [sets["DNp09"], sets["moonwalker"], *by_side(steer), sets["giant_fiber"], sets["proboscis_mn"],
                   *by_side(sets["danger_valence"]), *by_side(sets["touch_steer"]), sets["aIPg"],
                   sets["grooming_dn"], *by_side(sets["wind_steer"])]
        self.idx = torch.from_numpy(np.concatenate(members)).to(DEVICE)
        self.group = np.repeat(np.arange(len(members)), [len(m) for m in members])
        self.size = np.array([len(m) for m in members], np.float32)
        self.tau = np.full(len(members), TAU_MS)
        self.tau[[STINK_L, STINK_R]] = STINK_TAU_MS
        self.tau[AIPG] = AGGR_TAU_MS
        # weigh touch against the other steering DNs as if they were one averaged group
        self.touch_share = self.size[TOUCH_L] / (self.size[STEER_L] + self.size[TOUCH_L])
        self.rates = np.zeros((batch, len(members)), np.float32)  # Hz, smoothed
        self.stink_affinity = np.broadcast_to(np.asarray(stink_affinity, float), batch).copy()
        self.rng = np.random.default_rng(seed)
        self.wander = np.zeros(batch)
        self.escape_left = np.zeros(batch, int)
        self.in_stink = np.zeros(batch, bool)
        self.at_water = np.zeros(batch, bool)
        self.wades = np.zeros(batch, bool)
        self.body = {}  # set by the server each body step; see Physiology.motor
        self.lingers = np.zeros(batch, bool)
        self.gf_recent = np.zeros((ESCAPE_WINDOW, batch))

        self.ticks = 0

    def update(self, spk: torch.Tensor) -> list[dict]:
        """Feed one tick's (batch, n) spike mask; return one intent dict per brain."""
        s = spk[:, self.idx].cpu().numpy()
        per_group = np.zeros_like(self.rates)
        for g in range(len(self.size)):
            per_group[:, g] = s[:, self.group == g].sum(axis=1) / self.size[g]
        self.rates += DT_MS / self.tau * (per_group * 1000 / DT_MS - self.rates)
        r = self.rates.T

        if self.ticks % WANDER_TICKS == 0:
            self.wander = self.rng.uniform(-WANDER_VYAW, WANDER_VYAW, len(self.rates))
        self.ticks += 1
        self.gf_recent = np.roll(self.gf_recent, 1, axis=0)
        self.gf_recent[0] = s[:, self.group == GF].sum(axis=1)
        onset = (self.gf_recent.sum(axis=0) >= ESCAPE_SPIKES) & (self.escape_left == 0)
        self.escape_left = np.where(onset, ESCAPE_TICKS, np.maximum(self.escape_left - 1, 0))

        n = len(self.rates)
        b = lambda key, default: np.broadcast_to(np.asarray(self.body.get(key, default)), n)
        asleep, swimming, shore = b("asleep", False), b("swimming", False), b("at_shore", False)
        water = shore | swimming
        visit = water & (~self.at_water | (self.ticks % WADE_REROLL_TICKS == 0))
        urge = b("swim_urge", 0.5) * (1 - b("thirst", 0.5) * b("swim_thirst_weight", 1.0))
        self.wades = np.where(visit, self.rng.random(n) < urge, self.wades & water)
        self.at_water = water

        stink = np.clip((np.maximum(r[STINK_L], r[STINK_R]) - STINK_FLOOR_HZ) / (STINK_FULL_HZ - STINK_FLOOR_HZ), 0, 1)
        meeting = (stink > 0) & ~self.in_stink
        self.lingers = np.where(meeting, self.rng.random(len(stink)) < self.stink_affinity, self.lingers)
        self.in_stink = stink > 0
        # lingering in a smell keeps a duck in it; hunger breaks that loop, as it does for the pond
        avoid, like = stink * ~self.lingers, stink * self.lingers * (1 - b("hunger", 0.0))
        aggression = np.clip((r[AIPG] - AGGR_FLOOR_HZ) / (AGGR_FULL_HZ - AGGR_FLOOR_HZ), 0, 1)
        # Stop to eat only with something under the beak (`tasting`): the proboscis neurons also fire to
        # touch, so two ducks leaning on each other would otherwise both stop and starve.
        feeding = ((r[FEED] > FEED_HZ) & ~self.wades & ~swimming & ~asleep & b("tasting", True)
                   & (b("fear", 0.0) < FEAR_STOPS_FEEDING)
                   & ~b("sharing", False))  # a kind duck that is not starving leaves the food to one crying for it
        attack = ((r[TOUCH_L] + r[TOUCH_R] > TOUCH_HZ) & (self.rng.random(n) < ATTACK_P * aggression)
                  & ~swimming & ~asleep)
        zoomies = b("zoomies", False)
        # A hat itches, and the grooming neurons say so; vanity stops a duck shaking it off. Rolled every
        # few seconds, not every tick, or no hat survives its first second.
        preen = ((r[GROOM] > GROOM_HZ) & b("hatted", False) & ~asleep
                 & (self.ticks % PREEN_REROLL_TICKS == 0)
                 & (self.rng.random(n) < PREEN_P * (1 - b("vanity", 0.5))))

        vx = np.clip(BASE_VX * b("restlessness", 1.0) + VX_PER_HZ * (r[FWD] - r[BACK]),
                     -RUN_VX, RUN_VX) * b("speed", 1.0)
        vx = np.where(zoomies, RUN_VX, vx)
        # a duck that chose to swim paddles nearly in place, the more so the stronger its urge; one that
        # did not walks back out at its normal pace
        vx = np.where(swimming & self.wades, vx * (1 - 0.9 * b("swim_urge", 0.5)), vx)
        vx = np.where(self.escape_left > 0, -RUN_VX, vx)
        vx = vx + (RUN_VX - vx) * avoid
        # A fly that smells food surges: it turns upwind and speeds up (Alvarez-Salvado et al. 2018).
        # Explicit: a duck picks up its feet as far as it is following a plume.
        vx = vx + (RUN_VX - np.maximum(vx, 0)) * b("surge", 0.0) * (vx > 0)
        vx = vx * (1 - (1 - STINK_LINGER) * like)
        vx = np.where(feeding, 0.0, vx)  # stop to eat
        # Explicit: a frightened duck runs from the other ducks, as far as it can smell any. An angry one
        # turns after them (below) but does not run at them, since running overshoots; it runs when it strikes.
        ducks_near = np.clip((b("duck_left", 0.0) + b("duck_right", 0.0)) / 2, 0, 1)
        fled, chased = FLEE * np.clip(b("fear", 0.0), 0, 1), CHASE * aggression
        vx = vx + (RUN_VX - np.maximum(vx, 0)) * fled * ducks_near * (vx >= 0)
        # an angry duck hurries at PURSUE_VX towards a duck it smells but is not touching yet
        apart = ~(r[TOUCH_L] + r[TOUCH_R] > TOUCH_HZ)
        vx = vx + (PURSUE_VX - np.minimum(np.maximum(vx, 0), PURSUE_VX)) * chased * ducks_near * apart * (vx >= 0)
        vx = np.where(attack, RUN_VX, vx)
        # practice shows: a duck that has swum a lot is quicker in the water, one that has walked a lot quicker on land
        vx = vx * np.where(swimming, 1 + SWIM_GROWS * b("swim_skill", 0.0), 1 + WALK_GROWS * b("walk_skill", 0.0))
        vx = np.where(asleep, 0.0, vx)

        share = self.touch_share
        toward_touch = np.maximum(aggression, b("social", 0.0))
        # The pooled readout carries a built-in turn of about 0.4 rad/s. It is left in: subtracting a running
        # average also subtracts humid air's lasting left-right signal and halves water-finding.
        steer = (1 - share) * (r[STEER_L] - r[STEER_R]) + share * (r[TOUCH_L] - r[TOUCH_R]) * (1 - 2 * toward_touch)
        wander = self.wander * b("wander", 1.0) * np.where(zoomies, 2.0, 1.0)
        # Explicit: a duck with a taste for music turns toward the louder ear, one without turns away,
        # and one in the middle does neither.
        taste = 2 * b("music_affinity", 0.5) - 1
        # sociability is where a duck starts; fondness is what it has learned, and a lesson fully learned
        # is worth the whole dial
        liking = np.clip(2 * b("sociability", 0.5) - 1 + b("fondness", 0.0) + b("sleepy_together", 0.0), -1, 1) * b("at_ease", 1.0)
        # Fear turns a duck away from the others and a fight after them. Company is wanted to arm's length:
        # `near_*` is 1 at no distance and 0 at a metre, the pull to a liked duck is gone by 0.35 m, and a
        # duck nearer than that to anyone, and not cross, turns aside.
        beside = b("near_left", 0.0) + b("near_right", 0.0)
        room = 1 - np.clip((beside - ROOM_FROM) / (ROOM_GONE - ROOM_FROM), 0, 1)
        # a duck on its way to comfort or shove someone does not step aside from it
        crowded = np.clip((beside - ROOM_GONE) / 0.1, 0, 1) * (1 - aggression) * b("at_ease", 1.0) * (1 - b("intent", 0.0))
        liking = np.where(liking > 0, liking * room, liking)
        liking = np.clip(liking + chased - 2 * fled, -1, 1)
        # Fleeing a stink adds a turn away from it and leaves the rest of the steering alone.
        # DNge091, read like the other steering neurons, turns a duck upwind at the same gain; it only
        # fires as far as the duck smells food it wants (brain/server.py).
        vyaw = ((wander + VYAW_PER_HZ * (steer + r[WIND_L] - r[WIND_R]))
                + STINK_VYAW_PER_HZ * (r[STINK_L] - r[STINK_R]) * (like - avoid)
                + MUSIC_VYAW * taste * (b("music_left", 0.0) - b("music_right", 0.0))
                + COMPANIONSHIP_VYAW * liking * (b("duck_left", 0.0) - b("duck_right", 0.0))
                + PLAY_VYAW * b("play", 0.0) * (b("ball_left", 0.0) - b("ball_right", 0.0))
                # a hot duck turns for the pond or the shade, whichever its water love picks (Physiology.cooling)
                + COOL_VYAW * (b("cool_left", 0.0) - b("cool_right", 0.0))
                + INTENT_VYAW * b("intent_turn", 0.0)  # explicit: towards the duck it is going to
                # explicit social turns (brain/social.py): towards friends and away from grudges, towards a
                # trusted hand and away from a distrusted one, and for a kind duck towards one crying; all
                # only as far as nothing is pressing
                + b("at_ease", 1.0) * (BOND_VYAW * np.where(b("bond_near", 0.0) > 0, room, 1.0)
                                       * (b("bond_turn", 0.0) + b("bond_near", 0.0) * (b("near_left", 0.0) - b("near_right", 0.0)))
                                       - CROWDED_VYAW * crowded * (b("near_left", 0.0) - b("near_right", 0.0))
                                       + HAND_VYAW * b("hand_trust", 0.0) * (b("hand_left", 0.0) - b("hand_right", 0.0))
                                       + COMFORT_VYAW * b("comfort", 0.0) * (b("cry_left", 0.0) - b("cry_right", 0.0))))
        vyaw = np.where(asleep, 0.0, vyaw)
        return [
            {"vx": float(vx[b]), "vy": 0.0, "vyaw": float(vyaw[b]), "escape": bool(onset[b]),
             "feed": bool(feeding[b]), "attack": bool(attack[b]), "preen": bool(preen[b]),
             "zoomies": bool(zoomies[b])}
            for b in range(len(self.rates))
        ]


if __name__ == "__main__":
    # The explicit turns, on a silent brain: no connectome needed, two made-up neurons a set.
    names = ["DNa02", "odor_steer", "moist_steer", "DNp09", "moonwalker", "giant_fiber", "proboscis_mn",
             "danger_valence", "touch_steer", "aIPg", "grooming_dn", "wind_steer"]
    sets = {name: np.array([2 * i, 2 * i + 1]) for i, name in enumerate(names)}
    ann = pd.DataFrame({"side": ["left", "right"] * len(names)})
    silent = torch.zeros((1, 2 * len(names)), dtype=torch.bool, device=DEVICE)

    def turn(**body) -> float:
        dec = Decoder(ann, sets, 1)
        dec.wander = np.zeros(1)
        dec.ticks = 1  # past the tick that redraws the wander
        dec.body = {"sociability": 0.5, **body}
        return dec.update(silent)[0]["vyaw"]

    far, close, touching = (turn(bond_near=1.0, near_left=x) for x in (0.3, 0.65, 0.75))
    assert far > 0.3 and abs(close) < far / 2 and touching < 0, (far, close, touching)
    assert turn(bond_near=-1.0, near_left=0.6) < 0, "a grudge beside it still turns a duck away"
    assert turn(cool_left=0.5) > 0.5 > -0.5 > turn(cool_right=0.5)
    print(f"ok  a friend on the left turns a duck {far:+.2f} rad/s at 0.7 m, {close:+.2f} at 0.35 m and {touching:+.2f} "
          "touching; a hot duck turns for the eye its relief is in")
