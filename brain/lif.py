"""Batched leaky integrate-and-fire over the connectome on torch (PLAN.md Gate 1).

Per-tick rule (snedea/flybrain): V = 0.95 V + synaptic input + external input; spike at V >= 1,
reset to 0, refractory 3 ticks. Gate 1 decisions (2026-09-16): torch on MPS, DT_MS = 10, so the
membrane time constant is 200 ms and refractory 30 ms; each synapse adds SYN_GAIN of threshold
(flybrain's 0.15 * w / max|w| did not propagate); glutamate stays excitatory for now.
"""
import numpy as np
import torch
from scipy import sparse

DT_MS = 10.0
LEAK, THRESH, REFRAC = 0.95, 1.0, 3
SYN_GAIN = 0.005
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class LIF:
    """N brains sharing one weight matrix. Sparse COO matmul (CSR has no MPS kernel in torch 2.14)."""

    def __init__(self, W: sparse.csr_matrix, batch: int, device: str = DEVICE):
        coo = (W * SYN_GAIN).astype(np.float32).tocoo()
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        self.W = torch.sparse_coo_tensor(idx, torch.from_numpy(coo.data), W.shape).coalesce().to(device)
        n = W.shape[0]
        self.dev = device
        self.V = torch.zeros((batch, n), device=device)
        self.syn = torch.zeros((batch, n), device=device)
        self.ref = torch.zeros((batch, n), dtype=torch.int8, device=device)
        self.n_spikes = torch.zeros((batch, n), dtype=torch.int32, device=device)

    def step(self, in_b: np.ndarray, in_n: np.ndarray, in_v: np.ndarray | float = THRESH) -> torch.Tensor:
        """Advance one tick with external input in_v added at (in_b, in_n). Returns the (batch, n) spike mask."""
        V = self.V
        V.mul_(LEAK).add_(self.syn)
        b = torch.from_numpy(in_b).to(self.dev)
        n = torch.from_numpy(in_n).to(self.dev)
        v = torch.from_numpy(np.broadcast_to(np.float32(in_v), in_b.shape).copy()).to(self.dev)
        V.index_put_((b, n), v, accumulate=True)
        V.masked_fill_(self.ref > 0, 0.0)
        spk = V >= THRESH
        V.masked_fill_(spk, 0.0)
        self.ref.sub_(1).clamp_(min=0).masked_fill_(spk, REFRAC)
        self.n_spikes += spk
        self.syn = torch.sparse.mm(self.W, spk.T.float()).T
        return spk

    def counts(self) -> np.ndarray:
        return self.n_spikes.cpu().numpy()


def poisson_kicks(rng: np.random.Generator, batch: int, targets: np.ndarray, p: float):
    """Kick indices for one tick: each (instance, target) fires with probability p."""
    k = rng.binomial(batch * len(targets), p)
    flat = rng.integers(0, batch * len(targets), k)
    return flat // len(targets), targets[flat % len(targets)]
