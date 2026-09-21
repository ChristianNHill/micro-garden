"""Motor decoder: descending-neuron spikes -> smoothed rates -> microduck intent (PLAN.md Gates 2, 4, 4b, 4c).

- DNp09 drives forward, moonwalker (MDN) backward.
- DNa02, the odor-steering DNs and DNp12/DNp44 (humidity) turn toward their own side. DNg48 fires
  opposite a touch, so read the same way a touched duck turns away.
- ESCAPE_SPIKES giant fiber spikes within 100 ms start an escape (single spikes happen during odor).
- Proboscis motor neurons mean feeding.
- DNp32 fires for stink, more on the stink's side. A stink-averse duck bolts: it runs, other turning is
  suppressed, and it turns gently away from the busier DNp32 (strong turning made ducks circle in the
  stink, Gate 4b). A stink-loving duck slows down and turns gently toward it instead. Each time a duck
  meets stink it picks one of the two, lingering with probability stink_affinity, and keeps that choice
  until the stink is gone: that makes the knob a scale (a fixed blend flipped like a switch). This is a
  per-duck readout choice (Chris, 2026-09-16), not the brain changing.
- The body (brain/physiology.py `motor`, set by the server each body step in `self.body`) scales speed
  and spontaneous wandering, triggers zoomies, stops an asleep duck, and makes sociable ducks turn
  toward touch. At the pond's shore a duck either drinks or wades in to swim, chosen on arrival and
  every WADE_REROLL_TICKS after with a chance that rises with its swim urge and falls with thirst; a duck that chose to swim paddles nearly in
  place, the more so the stronger its urge, and one that did not walks back out.
- DNge091 fires on the side the wind comes from and turns the duck into it; the body only lets it hear
  the wind while it smells food it wants, which is how a fly finds food (Gate 9b).
- aIPg is aggression (its mood input is set in physiology). While it is active the touch turn flips
  toward the other duck, and a touching duck attacks (runs at it and headbutts) with a chance per tick
  that scales with aggression.

Gate 4 (Chris, 2026-09-16): ducks walk at BASE_VX by default and the brain can push them up to
RUN_VX, because nothing reaches DNp09 before the duck has eyes. Spontaneous turning (WANDER_*) is
the same for real and shuffled brains, so any difference between them comes from the brain.
"""
import numpy as np
import pandas as pd
import torch

from brain.lif import DEVICE, DT_MS

TAU_MS = 100.0
BASE_VX, RUN_VX = 0.08, 0.3  # m/s
VX_PER_HZ = 0.02  # m/s per Hz of forward minus backward drive
VYAW_PER_HZ = 1.0  # rad/s per Hz of left minus right steering DNs, positive = turn left
ESCAPE_TICKS = 50  # 0.5 s of running backward
# giant fiber spikes within 10 ticks (100 ms). Sun and dry air alone reach 2 in 5% of windows and 3 in
# none; looming reaches 3 in about 20% (Gate 4b).
# Measured over lone ducks, 2026-09-20: three spikes land in a window by chance 0.44 times a duck-minute
# blind and 1.19 with its eyes open (its own turning sweeps the garden across the retina), while a hand
# swooping in from 1.6 m trips it 18% of the time. So most startles are at nothing. Chris, 2026-09-20: keep
# them; a duck that jumps at shadows is a duck. Four spikes would all but end the false ones (0.06) and the
# real ones with them (1%).
ESCAPE_SPIKES, ESCAPE_WINDOW = 3, 10
WANDER_VYAW, WANDER_TICKS = 0.5, 50  # spontaneous turn rate, redrawn every 0.5 s
FEED_HZ = 1.0  # proboscis MN rate that means "eat"
# DNp32 is one neuron per side: a stray spike adds 0.5 Hz to its 2 s average, stink holds it near 1 Hz.
STINK_FLOOR_HZ, STINK_FULL_HZ, STINK_TAU_MS = 0.3, 0.8, 2000.0
STINK_VYAW_PER_HZ = 0.5  # rad/s per Hz of left minus right DNp32
# Companionship: a duck turns toward the side the other ducks smell stronger on, or away from it, as far as
# its sociability and what it has learned about them say (brain/plasticity.py `fondness`). Explicit, like
# music, and for the same reason: the brain's own left and right for a smell are a fraction of a hertz under
# 1.5 Hz of steering noise (PLAN.md Gate 9b screens), so sociability never showed. It turns a duck as hard as
# a tune does.
COMPANIONSHIP_VYAW = 1.5
MUSIC_VYAW = 1.5  # rad/s toward the louder ear at full music affinity, and away from it at none
GROOM_HZ = 0.4  # grooming DN rate at which a duck is fussing with its head enough to shed a hat
PREEN_REROLL_TICKS = 500  # a hatted duck reconsiders the thing on its head every 5 s, as it does wading
PREEN_P = 0.3  # chance of shedding at each of those, at no vanity at all; a scale, not a threshold
FEAR_STOPS_FEEDING = 0.5  # a frightened duck goes off its food, which is how one duck drives another off
STINK_LINGER = 0.5  # a stink lover slows to this fraction of its speed in the stink
# aIPg mean rate; silent without the mood input. The full-scale rate is what the aggressiveness knob
# actually reaches at 1.0, measured: 0.0, 0.95, 2.63, 3.47, 4.37 Hz across the dial. It was 1.0 back
# when the senses were noisy, which clipped every knob from 0.5 up to the same value and turned the
# dial into a switch (Gate 4c, 2026-09-18).
AGGR_FLOOR_HZ, AGGR_FULL_HZ, AGGR_TAU_MS = 0.05, 4.4, 1000.0
WADE_REROLL_TICKS = 500  # a duck at the shore reconsiders wading in every 5 s
# Chance per tick of a headbutt while touching, at full aggression (graded, not a threshold). 0.1 meant
# ten strikes a second, which is not a duck, and it put every setting above aggression 0.2 inside the
# ~0.7 s it takes the touch rate to climb past TOUCH_HZ, so the dial could not spread. At 0.01 a fully
# aggressive duck strikes about once a second and the dial runs 20.0, 4.8, 1.7, 1.3, 1.0 s (Gate 4c).
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
        # touch and the other steering DNs were one averaged group before Gate 4c; keep that weighting
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
        # Lingering in a smell it likes keeps a duck in the smell, which keeps it lingering. Hunger is
        # what breaks that loop, as it is for the pond (audit, 2026-09-19).
        avoid, like = stink * ~self.lingers, stink * self.lingers * (1 - b("hunger", 0.0))
        aggression = np.clip((r[AIPG] - AGGR_FLOOR_HZ) / (AGGR_FULL_HZ - AGGR_FLOOR_HZ), 0, 1)
        # A duck stops to eat when there is something under its beak. The proboscis neurons also fire to
        # touch, so two sociable ducks leaning on each other each set the other's feeding intent, which
        # stopped them both where they stood, in a corner, starving (Gate 9b, 2026-09-19).
        feeding = ((r[FEED] > FEED_HZ) & ~self.wades & ~swimming & ~asleep & b("tasting", True)
                   & (b("fear", 0.0) < FEAR_STOPS_FEEDING))
        attack = ((r[TOUCH_L] + r[TOUCH_R] > TOUCH_HZ) & (self.rng.random(n) < ATTACK_P * aggression)
                  & ~swimming & ~asleep)
        zoomies = b("zoomies", False)
        # A hat itches, and the grooming neurons say so. Vanity is what stops a duck shaking it off.
        # Decided every few seconds rather than every tick: rolled per tick, even vanity 0.95 got six
        # thousand chances to shed in a minute and no hat survived its first second (Gate 8b).
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
        # A fly that smells food surges: it turns upwind and it speeds up (Alvarez-Salvado et al. 2018). A
        # duck following a plume at its ambling 0.06 m/s took over a minute to cross a garden whose food
        # lay uneaten, so it picks up its feet as far as it is following one, as it does fleeing a stink.
        vx = vx + (RUN_VX - np.maximum(vx, 0)) * b("surge", 0.0) * (vx > 0)
        vx = vx * (1 - (1 - STINK_LINGER) * like)
        vx = np.where(feeding, 0.0, vx)  # stop to eat
        vx = np.where(attack, RUN_VX, vx)
        vx = np.where(asleep, 0.0, vx)

        share = self.touch_share
        toward_touch = np.maximum(aggression, b("social", 0.0))
        # The pooled readout carries a built-in turn of about 0.4 rad/s (PLAN.md Gate 9b screen). Taking
        # its own 30 s average as zero removed it and halved water-finding with it, 105 sips a garden to
        # 50 over three gardens: humid air is the one smell whose left and right are strong and lasting,
        # and a lasting signal is exactly what a moving zero subtracts. The turn stays.
        steer = (1 - share) * (r[STEER_L] - r[STEER_R]) + share * (r[TOUCH_L] - r[TOUCH_R]) * (1 - 2 * toward_touch)
        wander = self.wander * b("wander", 1.0) * np.where(zoomies, 2.0, 1.0)
        # Music: a duck with a taste for it turns toward the louder ear, one without turns away, and a
        # duck in the middle does neither. Whether it likes music is the knob; the sound is the garden's.
        taste = 2 * b("music_affinity", 0.5) - 1
        # sociability is where a duck starts; fondness is what life has done to that, and a lesson fully
        # learned is worth the whole of the dial
        liking = np.clip(2 * b("sociability", 0.5) - 1 + b("fondness", 0.0), -1, 1)
        # Fleeing a stink adds a turn away from it; it does not stop a duck steering by everything
        # else. The (1 - avoid) factor here used to scale down all the rest, so a hungry duck within
        # smell of the demo garden's stink patch lost a third of its steering toward the dish, ran past
        # at speed and never came back: it got to 0.41 m and ended 2.28 m away (audit, 2026-09-19).
        # DNge091 fires on the side the wind comes from, so read like the other steering neurons it
        # turns a duck upwind, at the same gain. It only fires as far as the duck smells food it wants
        # (brain/server.py), so this is a duck following its nose up the wind and nothing else.
        vyaw = ((wander + VYAW_PER_HZ * (steer + r[WIND_L] - r[WIND_R]))
                + STINK_VYAW_PER_HZ * (r[STINK_L] - r[STINK_R]) * (like - avoid)
                + MUSIC_VYAW * taste * (b("music_left", 0.0) - b("music_right", 0.0))
                + COMPANIONSHIP_VYAW * liking * (b("duck_left", 0.0) - b("duck_right", 0.0)))
        vyaw = np.where(asleep, 0.0, vyaw)
        return [
            {"vx": float(vx[b]), "vy": 0.0, "vyaw": float(vyaw[b]), "escape": bool(onset[b]),
             "feed": bool(feeding[b]), "attack": bool(attack[b]), "preen": bool(preen[b])}
            for b in range(len(self.rates))
        ]
