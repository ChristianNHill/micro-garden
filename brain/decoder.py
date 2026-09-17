"""Motor decoder: descending-neuron spikes -> smoothed rates -> microduck intent (PLAN.md Gates 2, 4, 4b, 4c).

- DNp09 drives forward, moonwalker (MDN) backward.
- DNa02, the odor-steering DNs and DNp12/DNp44 (humidity) turn toward their own side. DNg48 fires
  opposite a touch, so read the same way a touched duck turns away.
- ESCAPE_SPIKES giant fiber spikes within 100 ms start an escape (single spikes happen during odor).
- Proboscis motor neurons mean feeding.
- DNp32 fires for stink, more on the stink's side. A stink-averse duck bolts: it runs, other turning is
  suppressed, and it turns gently away from the busier DNp32 (strong turning made ducks circle in the
  stink, Gate 4b). A stink-loving duck (stink_affinity 1) slows down and turns gently toward it instead;
  that is a per-duck readout choice (Chris, 2026-09-16), not the brain changing.
- aIPg is aggression (its mood input is set in physiology). While it is active the touch turn flips
  toward the other duck, and an aggressive duck that is touching another attacks: runs at it and
  headbutts.

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
ESCAPE_SPIKES, ESCAPE_WINDOW = 3, 10
WANDER_VYAW, WANDER_TICKS = 0.5, 50  # spontaneous turn rate, redrawn every 0.5 s
FEED_HZ = 1.0  # proboscis MN rate that means "eat"
# DNp32 is one neuron per side: a stray spike adds 0.5 Hz to its 2 s average, stink holds it near 1 Hz.
STINK_FLOOR_HZ, STINK_FULL_HZ, STINK_TAU_MS = 0.3, 0.8, 2000.0
STINK_VYAW_PER_HZ = 0.5  # rad/s per Hz of left minus right DNp32
STINK_LINGER = 0.5  # a stink lover slows to this fraction of its speed in the stink
AGGR_FLOOR_HZ, AGGR_FULL_HZ, AGGR_TAU_MS = 0.3, 0.8, 1000.0  # aIPg mean rate
ATTACK_AGGRESSION = 0.5
TOUCH_HZ = 1.0  # DNg48 left plus right rate that means another duck is touching

FWD, BACK, STEER_L, STEER_R, GF, FEED, STINK_L, STINK_R, TOUCH_L, TOUCH_R, AIPG = range(11)


class Decoder:
    def __init__(self, ann: pd.DataFrame, sets: dict[str, np.ndarray], batch: int, seed: int = 0,
                 stink_affinity=0.0):
        side = ann["side"].to_numpy()

        def by_side(ix):
            return ix[side[ix] == "left"], ix[side[ix] == "right"]

        steer = np.concatenate([sets["DNa02"], sets["odor_steer"], sets["moist_steer"]])
        members = [sets["DNp09"], sets["moonwalker"], *by_side(steer), sets["giant_fiber"], sets["proboscis_mn"],
                   *by_side(sets["danger_valence"]), *by_side(sets["touch_steer"]), sets["aIPg"]]
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

        stink = np.clip((np.maximum(r[STINK_L], r[STINK_R]) - STINK_FLOOR_HZ) / (STINK_FULL_HZ - STINK_FLOOR_HZ), 0, 1)
        avoid, like = stink * (1 - self.stink_affinity), stink * self.stink_affinity
        aggression = np.clip((r[AIPG] - AGGR_FLOOR_HZ) / (AGGR_FULL_HZ - AGGR_FLOOR_HZ), 0, 1)
        feeding = r[FEED] > FEED_HZ
        attack = (r[TOUCH_L] + r[TOUCH_R] > TOUCH_HZ) & (aggression > ATTACK_AGGRESSION)

        vx = np.clip(BASE_VX + VX_PER_HZ * (r[FWD] - r[BACK]), -RUN_VX, RUN_VX)
        vx = np.where(self.escape_left > 0, -RUN_VX, vx)
        vx = vx + (RUN_VX - vx) * avoid
        vx = vx * (1 - (1 - STINK_LINGER) * like)
        vx = np.where(feeding, 0.0, vx)  # stop to eat
        vx = np.where(attack, RUN_VX, vx)

        share = self.touch_share
        steer = (1 - share) * (r[STEER_L] - r[STEER_R]) + share * (r[TOUCH_L] - r[TOUCH_R]) * (1 - 2 * aggression)
        vyaw = ((self.wander + VYAW_PER_HZ * steer) * (1 - avoid)
                + STINK_VYAW_PER_HZ * (r[STINK_L] - r[STINK_R]) * (like - avoid))
        return [
            {"vx": float(vx[b]), "vy": 0.0, "vyaw": float(vyaw[b]), "escape": bool(onset[b]),
             "feed": bool(feeding[b]), "attack": bool(attack[b])}
            for b in range(len(self.rates))
        ]
