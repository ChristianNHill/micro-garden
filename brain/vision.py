"""Visual front end: flyvis runs retina to T4/T5, its columns drive the matching FlyWire cells.

Light cannot go straight into FlyWire's photoreceptors: that synapse is histamine (FlyWire predicts
acetylcholine), and early vision needs graded, tonically active, temporally filtered cells that a
spiking model does not have. flyvis (trained on optic flow) stands in for the optic lobe, and the
connectome does the rest (LPLC2, LC9, and on inward).

flyvis drives every FlyWire cell of a type it also models, column by column. T4/T5 alone is not
enough: they are only a quarter of LPLC2's excitatory input.

One flyvis network runs every eye as a batch entry. Eye 0 is the left eye and drives FlyWire's
left-side optic lobe, eye 1 the right.

ponytail: the driven cells keep their own FlyWire input too, so the spiking optic lobe still runs
underneath; and one network serves both eyes, so mirror symmetry between the lobes is not modelled.
Revisit if vision behaves oddly by side.
"""
import numpy as np
import torch

from body.stub2d.retina import HEX_AZ, HEX_EL, HEX_U, HEX_V, N_HEX

NETWORK = "flow/0000/000"  # one member of flyvis's trained ensemble; they differ in tuning, not in kind
DT_S = 0.02  # flyvis warns above 1/50; the body step is exactly that
GREY = 0.5
SETTLE_STEPS = 50  # 1 s of grey to find the true resting activity; flyvis converges by about 0.5 s
# Release per RMS of visual drive, once the eye has adapted. Dimensionless, so it holds across scenes
# whose raw drive differs 120-fold.
VIS_GAIN = 0.05
# Adaptation time. One scale per duck, shared by both eyes so left and right stay comparable for
# steering. Slower than a loom (about 1.5 s) so it tracks the scene, not the event.
VIS_ADAPT_S = 3.0
# Floor on the adapted scale, which caps the eye's gain. A blank field has drive RMS about 1e-5
# (numerical residue), a real scene about 0.05; without a floor a blank retina fires the giant fiber.
VIS_FLOOR = 0.005
# Resting release. A hyperpolarised graded cell releases less, which disinhibits its targets, so
# rest sits mid-range and a cell can withhold as much as it adds. Too low, and the clamp at zero
# makes LPLC2 prefer a receding disc to a looming one. LIF.calibrate subtracts the resting drive.
VIS_TONIC = 0.5


def _hex_rows() -> np.ndarray:
    """Row index per hex column, dorsal row 0."""
    step = np.diff(np.unique(np.round(HEX_EL, 6))).min()
    return np.round((HEX_EL.max() - HEX_EL) / step).astype(int)


def _rank(x: np.ndarray) -> np.ndarray:
    """Position of each element in [0, 1) by rank, ties broken arbitrarily but stably."""
    out = np.empty(len(x))
    out[np.argsort(x, kind="stable")] = (np.arange(len(x)) + 0.5) / len(x)
    return out


def hex_column_of(azim: np.ndarray, elev: np.ndarray) -> np.ndarray:
    """Hex column for each neuron, from where it sits on the retinotopic sheet.

    Rank-matched row by row: the most dorsal neurons get the most dorsal hex row, and within a row the
    most frontal get the most frontal columns. Rank order survives the optic lobe's curvature, which a
    plane fit does not.

    ponytail: retinotopic in order, not in degrees. The exact map is the Codex Visual Columns file;
    drop it in and replace this.
    """
    rows = _hex_rows()
    sizes = np.bincount(rows)
    edges = np.concatenate([[0], np.cumsum(sizes) / sizes.sum()])
    row_of = np.clip(np.searchsorted(edges, _rank(elev), "right") - 1, 0, len(sizes) - 1)
    col = np.empty(len(azim), int)
    for r in range(len(sizes)):
        in_row = np.flatnonzero(row_of == r)
        if len(in_row) == 0:
            continue
        hexes = np.flatnonzero(rows == r)[np.argsort(HEX_AZ[rows == r])]  # frontal first
        col[in_row] = hexes[(_rank(azim[in_row]) * len(hexes)).astype(int)]
    return col


def _columns_of(net, node_type: np.ndarray, cell_type: str) -> np.ndarray:
    """flyvis node index per hex column for one cell type. Types thinner than the lattice (Am, Lawf1,
    the Tm5s) do not fill every column, so a gap borrows its nearest neighbour's column."""
    at = np.full(N_HEX, -1)
    nodes = np.flatnonzero(node_type == cell_type)
    hex_of = {(u, v): i for i, (u, v) in enumerate(zip(HEX_U, HEX_V))}
    at[[hex_of[(u, v)] for u, v in zip(net.connectome.nodes.u[:][nodes], net.connectome.nodes.v[:][nodes])]] = nodes
    gaps, have = np.flatnonzero(at < 0), np.flatnonzero(at >= 0)
    if len(gaps):
        d = np.hypot(HEX_AZ[gaps, None] - HEX_AZ[have], HEX_EL[gaps, None] - HEX_EL[have])
        at[gaps] = at[have[d.argmin(1)]]
    return at


def _sheet_coords(ann, cell_type: str, side: str):
    """FlyWire row indices for one type on one side, with each neuron's (rightward, upward) sheet position.

    FlyWire coordinates: x is medial-lateral, y grows ventrally, z grows posteriorly (checked against
    Kenyon cells being dorsal and the sugar GRNs ventral, the antennae anterior and the calyx posterior).
    Rightward in an eye's own field means posterior for the right eye and anterior for the left.
    """
    rows = np.flatnonzero((ann["cell_type"].fillna("") == cell_type) & (ann["side"] == side))
    pos = ann.iloc[rows]
    azim = pos["pos_z"].to_numpy(float) * (1 if side == "right" else -1)
    return rows, azim, -pos["pos_y"].to_numpy(float)


class Vision:
    """flyvis stepping in lockstep with the body, one state per duck per eye."""

    def __init__(self, ann, batch: int, network: str = NETWORK, device: str = "cpu"):
        import flyvis  # a few seconds to import; only vision needs it

        self.net = flyvis.NetworkView(network).init_network()
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.params = self.net._param_api()
        self.batch = batch
        self.state = self.net.steady_state(1.0, DT_S, 2 * batch, value=GREY)
        self.rest = torch.zeros_like(self.state.nodes.activity)

        node_type = np.array([t.decode() for t in self.net.connectome.nodes.type[:]])
        # cell types both name: 50 of flyvis's 65, about 55k FlyWire cells, photoreceptors excluded
        self.types = sorted(set(node_type) & set(ann["cell_type"].dropna()))
        columns = {t: _columns_of(self.net, node_type, t) for t in self.types}
        idx, src, eye_of = [], [], []
        for eye, side in enumerate(("left", "right")):
            for t in self.types:
                rows, azim, elev = _sheet_coords(ann, t, side)
                idx.append(rows)
                src.append(columns[t][hex_column_of(azim, elev)])
                eye_of.append(np.full(len(rows), eye))
        self.neurons = np.concatenate(idx)  # FlyWire row index per driven neuron
        self.index = torch.from_numpy(self.neurons).to(device)
        self.device = device
        self._src = torch.from_numpy(np.concatenate(src))
        self._eye = torch.from_numpy(np.concatenate(eye_of))
        self.scale = None
        # steady_state and a grey image through the stimulus differ by about 0.03, so rest is measured
        # down the path the eye uses: a blank scene must read as no drive.
        for _ in range(SETTLE_STEPS):
            self.step(np.full((batch, 2, N_HEX), GREY, np.float32))
        self.rest = self.state.nodes.activity.clone()
        # unseeded, so adaptation starts from the first real frame, not from blank grey
        self.scale = None

    def step(self, lum: np.ndarray, gain=1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """One body step of vision. lum is (batch, 2, 721); returns LIF.step's graded argument.

        gain scales the departure from rest per brain (a timid duck watches harder, brain/physiology.py),
        never the resting release.
        """
        x = torch.from_numpy(np.ascontiguousarray(lum, np.float32)).reshape(2 * self.batch, 1, 1, N_HEX)
        self.net.stimulus.zero(2 * self.batch, 1)
        self.net.stimulus.add_input(x)
        self.state = self.net._next_state(self.params, self.state, self.net.stimulus()[:, 0], DT_S)
        drive = (self.state.nodes.activity - self.rest).reshape(self.batch, 2, -1)
        raw = drive[:, self._eye, self._src]
        rms = raw.pow(2).mean(1, keepdim=True).sqrt()
        a = DT_S / VIS_ADAPT_S
        self.scale = rms if self.scale is None else (1 - a) * self.scale + a * rms
        g = torch.as_tensor(gain, dtype=torch.float32).reshape(-1, 1)
        level = raw / self.scale.clamp(min=VIS_FLOOR) * (VIS_GAIN * g) + VIS_TONIC
        return self.index, level.clamp_(0, 1).to(self.device)
