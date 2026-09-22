"""Batched leaky integrate-and-fire over the connectome on torch.

Per-tick rule (snedea/flybrain): V = LEAK V + synaptic input + external input; spike at
V >= THRESH + adaptation, reset to 0, refractory REFRAC ticks. Each synapse adds SYN_GAIN of threshold
per synapse count; inhibitory ones are scaled by INH_RATIO. Each spike raises that neuron's threshold
by ADAPT_INC, decaying by ADAPT_DECAY per tick (spike-frequency adaptation, about 200 ms).
SYN_GAIN and ADAPT_INC are tuned so activity dies within 0.5 s of odor off and odor still reaches the
steering DNs.
"""
import numpy as np
import torch
from scipy import sparse

DT_MS = 10.0
LEAK, THRESH, REFRAC = 0.95, 1.0, 3
SYN_GAIN = 0.01
INH_RATIO = 1.0
ADAPT_INC, ADAPT_DECAY = 1.0, 0.95
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class LIF:
    """N brains sharing one weight matrix. Sparse COO matmul (CSR has no MPS kernel in torch 2.14)."""

    def __init__(self, W: sparse.csr_matrix, batch: int, device: str = DEVICE):
        coo = W.tocoo()
        data = (coo.data * SYN_GAIN * np.where(coo.data < 0, INH_RATIO, 1.0)).astype(np.float32)
        idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
        self.W = torch.sparse_coo_tensor(idx, torch.from_numpy(data), W.shape).coalesce().to(device)
        n = W.shape[0]
        self.dev = device
        self.V = torch.zeros((batch, n), device=device)
        self.syn = torch.zeros((batch, n), device=device)
        self.adapt = torch.zeros((batch, n), device=device)
        self.ref = torch.zeros((batch, n), dtype=torch.int8, device=device)
        self.n_spikes = torch.zeros((batch, n), dtype=torch.int32, device=device)
        self.rest_current = torch.zeros(n, device=device)
        # Per-neuron threshold above the shared one. Kenyon cells fire only when several inputs
        # coincide, which keeps the odor code sparse.
        self.thresh_offset = torch.zeros(n, device=device)

    def step(self, in_b: np.ndarray, in_n: np.ndarray, graded: tuple[torch.Tensor, torch.Tensor] | None = None,
             plastic=None) -> torch.Tensor:
        """Advance one tick with a threshold-sized kick at each (in_b, in_n). Returns the (batch, n) spike mask.

        graded is (neuron indices, release in [0, 1] per brain) for non-spiking cells; their release
        replaces their spike this tick, 1.0 being one spike's worth of transmitter.

        plastic is a brain.plasticity.Plasticity with per-duck Kenyon cell -> MBON synapses. It must own
        that block (see `strip`) or it counts twice.
        """
        V = self.V
        V.mul_(LEAK).add_(self.syn)
        b = torch.from_numpy(in_b).to(self.dev)
        n = torch.from_numpy(in_n).to(self.dev)
        V.index_put_((b, n), torch.full(b.shape, THRESH, device=self.dev), accumulate=True)
        V.masked_fill_(self.ref > 0, 0.0)
        spk = V >= THRESH + self.adapt + self.thresh_offset
        V.masked_fill_(spk, 0.0)
        self.ref.sub_(1).clamp_(min=0).masked_fill_(spk, REFRAC)
        if ADAPT_INC:
            self.adapt.mul_(ADAPT_DECAY).add_(spk, alpha=ADAPT_INC)
        self.n_spikes += spk
        release = spk.float()
        if graded is not None:
            release[:, graded[0]] = graded[1]
        self.syn = torch.sparse.mm(self.W, release.T).T - self.rest_current
        if plastic is not None:
            self.syn += plastic.current(spk)
        return spk

    def calibrate(self, indices: torch.Tensor, rest_release: float) -> None:
        """Take the current a set of graded cells delivers at rest as the zero point.

        Graded cells release all the time, so without this a blank scene fires the giant fiber. Only
        departures from rest drive anything.
        """
        rest = torch.zeros(1, self.W.shape[1], device=self.dev)
        rest[:, indices] = rest_release
        self.rest_current = torch.sparse.mm(self.W, rest.T).T[0]

    def counts(self) -> np.ndarray:
        return self.n_spikes.cpu().numpy()
