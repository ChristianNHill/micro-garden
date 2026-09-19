"""Batched leaky integrate-and-fire over the connectome on torch (PLAN.md Gate 1).

Per-tick rule (snedea/flybrain): V = LEAK V + synaptic input + external input; spike at
V >= THRESH + adaptation, reset to 0, refractory REFRAC ticks. Gate 1 decisions (2026-09-16): torch on
MPS, DT_MS = 10, each excitatory synapse adds SYN_GAIN of threshold (flybrain's 0.15 * w / max|w| did
not propagate). Glutamate became inhibitory at Gate 4.

Gate 4 model work: inhibitory synapses are scaled by INH_RATIO relative to excitatory ones, and each
spike raises that neuron's threshold by ADAPT_INC, decaying by ADAPT_DECAY per tick (spike-frequency
adaptation, about 200 ms). The sweep that chose SYN_GAIN 0.01 and ADAPT_INC 1.0 (from 0.005 and 0):
activity dies within 0.5 s of odor off, odor reaches steering DNs, Gate 2 still holds.
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
        # Per-neuron threshold, above the shared one. Kenyon cells need it: in the fly each fires only
        # when several of its few inputs coincide, and that is what keeps the odor code sparse.
        self.thresh_offset = torch.zeros(n, device=device)

    def step(self, in_b: np.ndarray, in_n: np.ndarray, graded: tuple[torch.Tensor, torch.Tensor] | None = None,
             plastic=None) -> torch.Tensor:
        """Advance one tick with a threshold-sized kick at each (in_b, in_n). Returns the (batch, n) spike mask.

        graded is (neuron indices, release in [0, 1] per brain) for cells that do not spike: the optic
        lobe's neurons are graded in the fly, and flyvis models them that way (brain/vision.py). Their
        release replaces their spike this tick, 1.0 being as much transmitter as one spike carries.

        plastic is a brain.plasticity.Plasticity, whose Kenyon cell -> MBON synapses are per duck and
        so cannot live in the shared matrix. It must own that block (see `strip`) or it counts twice.
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

        Graded cells release all the time, so a blank scene still pushed the whole brain (Gate 6: the
        giant fiber fired as often with nothing to see as with a looming disc). A fly's brain is not
        wound up by an empty grey world, so the resting release is subtracted and only departures
        from it drive anything.
        """
        rest = torch.zeros(1, self.W.shape[1], device=self.dev)
        rest[:, indices] = rest_release
        self.rest_current = torch.sparse.mm(self.W, rest.T).T[0]

    def counts(self) -> np.ndarray:
        return self.n_spikes.cpu().numpy()
