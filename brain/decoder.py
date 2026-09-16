"""Motor decoder: descending-neuron spikes -> smoothed rates -> microduck intent (PLAN.md Gates 2, 4).

DNp09 drives forward, moonwalker (MDN) backward, DNa02 and the odor-steering DNs turn toward their
own side, a burst of ESCAPE_SPIKES giant fiber spikes within 100 ms starts an escape (single spikes
happen during odor), proboscis motor neurons mean feeding. Gains and TAU_MS are the tuning surface.

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
ESCAPE_SPIKES, ESCAPE_WINDOW = 2, 10  # giant fiber spikes within 10 ticks (100 ms)
WANDER_VYAW, WANDER_TICKS = 0.5, 50  # spontaneous turn rate, redrawn every 0.5 s
FEED_HZ = 1.0  # proboscis MN rate that means "eat"


class Decoder:
    def __init__(self, ann: pd.DataFrame, sets: dict[str, np.ndarray], batch: int, seed: int = 0):
        side = ann["side"].to_numpy()
        steer = np.concatenate([sets["DNa02"], sets["odor_steer"]])
        members = [
            sets["DNp09"],
            sets["moonwalker"],
            steer[side[steer] == "left"],
            steer[side[steer] == "right"],
            sets["giant_fiber"],
            sets["proboscis_mn"],
        ]
        self.idx = torch.from_numpy(np.concatenate(members)).to(DEVICE)
        self.group = np.repeat(np.arange(len(members)), [len(m) for m in members])
        self.size = np.array([len(m) for m in members], np.float32)
        self.rates = np.zeros((batch, len(members)), np.float32)  # Hz, smoothed
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
        self.rates += DT_MS / TAU_MS * (per_group * 1000 / DT_MS - self.rates)

        if self.ticks % WANDER_TICKS == 0:
            self.wander = self.rng.uniform(-WANDER_VYAW, WANDER_VYAW, len(self.rates))
        self.ticks += 1
        self.gf_recent = np.roll(self.gf_recent, 1, axis=0)
        self.gf_recent[0] = s[:, self.group == 4].sum(axis=1)
        onset = (self.gf_recent.sum(axis=0) >= ESCAPE_SPIKES) & (self.escape_left == 0)
        self.escape_left = np.where(onset, ESCAPE_TICKS, np.maximum(self.escape_left - 1, 0))

        fwd, back, left, right, _, feed = self.rates.T
        vx = np.clip(BASE_VX + VX_PER_HZ * (fwd - back), -RUN_VX, RUN_VX)
        vx = np.where(self.escape_left > 0, -RUN_VX, vx)
        feeding = feed > FEED_HZ
        vx = np.where(feeding, 0.0, vx)  # stop to eat
        vyaw = self.wander + VYAW_PER_HZ * (left - right)
        return [
            {"vx": float(vx[b]), "vy": 0.0, "vyaw": float(vyaw[b]),
             "escape": bool(onset[b]), "feed": bool(feeding[b])}
            for b in range(len(self.rates))
        ]
