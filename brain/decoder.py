"""Motor decoder: descending-neuron spikes -> smoothed rates -> microduck intent (PLAN.md Gate 2).

DNp09 drives forward, moonwalker (MDN) backward, DNa02 turns toward its own side, giant fiber
means escape, proboscis motor neurons mean feeding. Gains and TAU_MS are the tuning surface.
"""
import numpy as np
import pandas as pd
import torch

from brain.lif import DEVICE, DT_MS

TAU_MS = 100.0
VX_PER_HZ = 0.01  # m/s
VYAW_PER_HZ = 0.05  # rad/s, positive = turn left


class Decoder:
    def __init__(self, ann: pd.DataFrame, sets: dict[str, np.ndarray], batch: int):
        side = ann["side"].to_numpy()
        dna02 = sets["DNa02"]
        members = [
            sets["DNp09"],
            sets["moonwalker"],
            dna02[side[dna02] == "left"],
            dna02[side[dna02] == "right"],
            sets["giant_fiber"],
            sets["proboscis_mn"],
        ]
        self.idx = torch.from_numpy(np.concatenate(members)).to(DEVICE)
        self.group = np.repeat(np.arange(len(members)), [len(m) for m in members])
        self.size = np.array([len(m) for m in members], np.float32)
        self.rates = np.zeros((batch, len(members)), np.float32)  # Hz, smoothed
        self.escape_now = np.zeros(batch, bool)

    def update(self, spk: torch.Tensor) -> list[dict]:
        """Feed one tick's (batch, n) spike mask; return one intent dict per brain."""
        s = spk[:, self.idx].cpu().numpy()
        per_group = np.zeros_like(self.rates)
        for g in range(len(self.size)):
            per_group[:, g] = s[:, self.group == g].sum(axis=1) / self.size[g]
        alpha = DT_MS / TAU_MS
        self.rates += alpha * (per_group * 1000 / DT_MS - self.rates)
        self.escape_now = per_group[:, 4] > 0  # the giant fiber is a single-spike command
        fwd, back, left, right, _, feed = self.rates.T
        return [
            {"vx": float(VX_PER_HZ * (fwd[b] - back[b])), "vy": 0.0,
             "vyaw": float(VYAW_PER_HZ * (left[b] - right[b])), "escape": bool(self.escape_now[b]),
             "feed": float(feed[b])}
            for b in range(len(self.rates))
        ]
