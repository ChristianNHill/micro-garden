"""Packs the connectome for the browser, plus a reference run for web/lif.js to match.

Run: uv run python -m web.pack

Writes into web/data/, which git ignores because the FlyWire data is not redistributed from this repo:
  brain.bin       the signed weights, grouped by presynaptic neuron so a spike walks its own row
  sets.json       the named neuron sets, the sided ones split by side, and the Kenyon cell threshold
  reference.bin   what brain/lif.py does with the schedule in REFERENCE: spikes per tick, then per neuron

brain.bin layout, little-endian, every block 4-byte aligned:
  "MGB1", u32 neurons, u32 synapses, u32 distinct weights
  f32[distinct]   each distinct weight once, exactly as brain/lif.py holds it
  i32[neurons+1]  where each presynaptic neuron's synapses start
  i32[synapses]   postsynaptic neuron of each synapse, ascending within a neuron
  u16[synapses]   which distinct weight each synapse has
"""
import json
import struct
from pathlib import Path

import numpy as np
from scipy import sparse

from brain.data import load_connectome, named_sets
from brain.encoder import MAX_HZ
from brain.lif import DT_MS, INH_RATIO, LIF, SYN_GAIN
from brain.plasticity import KC_THRESHOLD, sparsen
from brain.server import SIDED

OUT = Path(__file__).resolve().parent / "data"
# (first tick, last tick + 1, {set: level}). Quiet, then food smelled on the left with wind from the
# left, then sugar, then a loom, so the check covers smell, wind, taste and the escape path.
REFERENCE = [
    (0, 100, {}),
    (100, 300, {"orn_food_left": 0.8, "orn_food_right": 0.2, "jo_push_left": 0.5, "jo_pull_right": 0.5}),
    (300, 400, {"sugar_grn": 1.0}),
    (400, 500, {"LPLC2": 0.8}),
]


def sided_sets(ann) -> dict[str, np.ndarray]:
    """named_sets plus a left and a right half of each sided set, as the brain server splits them."""
    sets = named_sets(ann)
    side = ann["side"].to_numpy()
    for name in SIDED:
        for s in ("left", "right"):
            sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
    return sets


def weights(W: sparse.csr_matrix) -> sparse.csc_matrix:
    """W scaled the way LIF.__init__ scales it, float32, one column per presynaptic neuron."""
    coo = W.tocoo()
    data = (coo.data * SYN_GAIN * np.where(coo.data < 0, INH_RATIO, 1.0)).astype(np.float32)
    return sparse.csc_matrix((data, (coo.row, coo.col)), shape=W.shape)


def write_brain(W: sparse.csr_matrix) -> None:
    C = weights(W)
    C.sort_indices()
    values, which = np.unique(C.data, return_inverse=True)
    assert len(values) < 2 ** 16, len(values)
    with open(OUT / "brain.bin", "wb") as f:
        f.write(b"MGB1" + struct.pack("<III", C.shape[0], C.nnz, len(values)))
        f.write(values.astype("<f4").tobytes())
        f.write(C.indptr.astype("<i4").tobytes())
        f.write(C.indices.astype("<i4").tobytes())
        f.write(which.astype("<u2").tobytes())


def release(level: float) -> float:
    """What brain/encoder.py's graded() releases for a level, with no noise."""
    return float(np.float32(np.clip(level * MAX_HZ * DT_MS / 1000, 0, 1)))


def write_reference(W, sets) -> None:
    """Run the Python model through REFERENCE on the CPU and keep its spikes."""
    import torch

    brain = LIF(W, 1, device="cpu")
    sparsen(brain, sets)
    none = np.empty(0, np.int64)
    per_tick = []
    for start, end, levels in REFERENCE:
        idx = np.concatenate([sets[k] for k in levels]) if levels else none
        rel = np.concatenate([np.full(len(sets[k]), release(v), np.float32) for k, v in levels.items()]) \
            if levels else np.empty(0, np.float32)
        drive = (torch.from_numpy(idx), torch.from_numpy(rel)[None, :])
        for _ in range(start, end):
            per_tick.append(int(brain.step(none, none, graded=drive).sum()))
    with open(OUT / "reference.bin", "wb") as f:
        f.write(struct.pack("<II", len(per_tick), W.shape[0]))
        f.write(np.array(per_tick, "<u4").tobytes())
        f.write(brain.counts()[0].astype("<u4").tobytes())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    W, ann = load_connectome()
    sets = sided_sets(ann)
    write_brain(W)
    (OUT / "sets.json").write_text(json.dumps({
        "sets": {k: v.tolist() for k, v in sets.items()},
        "kc_threshold": KC_THRESHOLD,
        "reference": [{"from": a, "to": b, "release": {k: release(v) for k, v in lv.items()}}
                      for a, b, lv in REFERENCE],
    }))
    write_reference(W, sets)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:14} {p.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
