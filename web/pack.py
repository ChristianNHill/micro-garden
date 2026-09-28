"""Packs what the browser garden needs, from the same Python objects the native garden builds.

Run: uv run python -m web.pack

Writes into web/data/, which git ignores because the FlyWire data is not redistributed from this repo:
  brain.bin.gz      the connectome without its Kenyon cell -> MBON block, that block for learning, the named
                    sets, the decoder's groups, the neurons the brain view shows, and which cells the eyes drive
  eyes.bin.gz       flyvis's optic lobe network, settled on grey, for brain/vision.py's eyes
  reference.bin.gz  what the Python model does with fixed input, for web/check.js to hold the browser to
  parity.json.gz    body, world and readout cases from the Python modules, for the same
  robot.json, clips.json, LICENSE-microduck   the duck's rig and motions, copied from viewer/godot

Each file is a bundle (see `write_bundle`): a JSON header naming each array, then the arrays.
"""
import gzip
import json
import shutil
import struct
from pathlib import Path

import numpy as np
from scipy import sparse

from brain.data import load_connectome, named_sets
from brain.encoder import MAX_HZ
from brain.lif import DT_MS, INH_RATIO, LIF, SYN_GAIN
from brain.plasticity import KC_THRESHOLD, Plasticity, sparsen, strip
from brain.server import SIDED

OUT = Path(__file__).resolve().parent / "data"
GODOT = Path(__file__).resolve().parent.parent / "viewer" / "godot"
# (first tick, last tick + 1, {set: level}, reward, punish). Quiet, food smelled on the left with wind from
# the left, sugar with a reward, then a loom with a punishment, so the check covers smell, wind, taste,
# the escape path and learning.
REFERENCE = [
    (0, 100, {}, 0.0, 0.0),
    (100, 300, {"orn_food_left": 0.8, "orn_food_right": 0.2, "jo_push_left": 0.5, "jo_pull_right": 0.5}, 0.0, 0.0),
    (300, 400, {"sugar_grn": 1.0, "orn_food_left": 0.5, "orn_food_right": 0.5}, 1.0, 0.0),
    (400, 500, {"LPLC2": 0.8, "orn_food_left": 0.5, "orn_food_right": 0.5}, 0.0, 1.0),
]
EYE_STEPS = 60  # of the reference loom for the eyes
EYE_KEPT = (10, 30, 59)  # steps at which the eyes' whole state is kept


def write_bundle(path: Path, arrays: dict[str, np.ndarray], meta: dict | None = None) -> None:
    """"MGB2", u32 header length, the JSON header, then each array 8-byte aligned, gzipped whole. The header
    maps each name to its dtype, shape and byte offset, and carries `meta` as is."""
    header, blobs, at = {"arrays": {}, "meta": meta or {}}, [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        assert a.dtype.str[1:] in ("f4", "f8", "i4", "u4", "i2", "u2", "i1", "u1"), (name, a.dtype)
        a = a.astype(a.dtype.newbyteorder("<"))
        header["arrays"][name] = {"dtype": a.dtype.str[1:], "shape": list(a.shape), "offset": at}
        blobs.append(a.tobytes())
        at += len(blobs[-1])
        blobs.append(b"\0" * (-at % 8))
        at += len(blobs[-1])
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * (-(8 + len(text)) % 8)
    with gzip.open(path, "wb", compresslevel=9) as f:
        f.write(b"MGB2" + struct.pack("<I", len(text)) + text + b"".join(blobs))


def by_source(post: np.ndarray, pre: np.ndarray, w: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Synapses grouped by presynaptic neuron, targets ascending, each weight an index into the distinct ones.
    Targets are stored as the gap from the one before, which zips far smaller than the targets themselves."""
    C = sparse.csc_matrix((w, (post, pre)), shape=(n, n))
    C.sort_indices()
    values, which = np.unique(C.data, return_inverse=True)
    assert len(values) < 2 ** 16, len(values)
    gap = np.diff(C.indices, prepend=0).astype(np.int32)
    gap[C.indptr[:-1][np.diff(C.indptr) > 0]] = C.indices[C.indptr[:-1][np.diff(C.indptr) > 0]]  # each neuron's first is absolute
    return {"start": C.indptr.astype(np.int32), "gap": gap, "which": which.astype(np.uint16), "values": values.astype(np.float32)}


def sided_sets(ann) -> dict[str, np.ndarray]:
    """named_sets plus a left and a right half of each sided set, as the brain server splits them."""
    sets = named_sets(ann)
    side = ann["side"].to_numpy()
    for name in SIDED:
        for s in ("left", "right"):
            sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
    return sets


def release(level: float) -> float:
    """What brain/encoder.py's graded() releases for a level, with no noise."""
    return float(np.float32(np.clip(level * MAX_HZ * DT_MS / 1000, 0, 1)))


def brain_bundle(W, ann, sets):
    from brain.brainview import BrainView, GROUPS, group_of
    from brain.decoder import Decoder
    from brain.vision import Vision

    stripped = strip(W, sets).tocoo()
    data = (stripped.data * SYN_GAIN * np.where(stripped.data < 0, INH_RATIO, 1.0)).astype(np.float32)
    arrays = {f"W.{k}": v for k, v in by_source(stripped.row, stripped.col, data, W.shape[0]).items()}
    plastic = Plasticity(W, sets, 1, "cpu")
    arrays |= {"plastic.post": plastic.post.numpy().astype(np.int32), "plastic.pre": plastic.pre.numpy().astype(np.int32),
               "plastic.pre_local": plastic.pre_local.numpy().astype(np.int32), "plastic.base": plastic.base.numpy(),
               "plastic.reward_gate": plastic.reward_gate.numpy().astype(np.uint8),
               "plastic.punish_gate": plastic.punish_gate.numpy().astype(np.uint8)}
    arrays |= {f"set.{k}": v.astype(np.int32) for k, v in sets.items()}
    dec = Decoder(ann, sets, 1)
    arrays["decoder.idx"], arrays["decoder.group"] = dec.idx.cpu().numpy().astype(np.int32), dec.group.astype(np.int32)
    view = BrainView(ann, sets, "cpu")
    xy = ann[["pos_x", "pos_y"]].to_numpy(float)[view.idx]
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)).max()
    arrays |= {"view.idx": view.idx.astype(np.int32), "view.xy": xy.astype(np.float32).ravel(),
               "view.group": group_of(ann)[view.idx].astype(np.uint8)}
    eyes = Vision(ann, 1)
    arrays |= {"vision.neurons": eyes.neurons.astype(np.int32), "vision.src": eyes._src.numpy().astype(np.int32),
               "vision.eye": eyes._eye.numpy().astype(np.uint8)}
    meta = {"neurons": int(W.shape[0]), "kc_threshold": KC_THRESHOLD, "decoder_groups": len(dec.size),
            "view_groups": list(GROUPS), "sets": sorted(sets)}
    return arrays, meta, eyes


def eyes_bundle(eyes):
    """flyvis's network as it steps (brain/vision.py): activity, bias and time constant per cell, the weighted
    edges, which cells take the image, and the state after settling on grey."""
    import torch

    net, p = eyes.net, eyes.params
    t = lambda x: np.asarray(x.detach() if isinstance(x, torch.Tensor) else x)
    src = np.asarray(net.connectome.edges.source_index[:])
    tgt = np.asarray(net.connectome.edges.target_index[:])
    n = len(net.connectome.nodes.type[:])
    arrays = {f"net.{k}": v for k, v in by_source(tgt, src, t(p.edges.weight).astype(np.float32), n).items()}
    arrays |= {"net.tau": t(p.nodes.time_const).astype(np.float32), "net.bias": t(p.nodes.bias).astype(np.float32),
               "net.input": np.asarray(net.stimulus.input_index).astype(np.int32),
               "settled": eyes.state.nodes.activity[0].numpy().astype(np.float32),
               "rest": eyes.rest[0].numpy().astype(np.float32)}
    return arrays, {"nodes": n}


def loom(step: int) -> np.ndarray:
    """(1, 2, 721): a dark disc growing in the left eye's centre, grey elsewhere; the eyes' reference input."""
    from body.stub2d.retina import HEX_AZ, HEX_EL, N_HEX
    lum = np.full((1, 2, N_HEX), 0.5, np.float32)
    lum[0, 0, np.hypot(HEX_AZ, HEX_EL) < 0.02 + 0.012 * step] = 0.05
    return lum


def reference_bundle(W, sets, eyes):
    import torch

    brain = LIF(strip(W, sets), 1, device="cpu")
    sparsen(brain, sets)
    plastic = Plasticity(W, sets, 1, "cpu")
    none = np.empty(0, np.int64)
    per_tick = []
    for start, end, levels, reward, punish in REFERENCE:
        idx = np.concatenate([sets[k] for k in levels]) if levels else none
        rel = np.concatenate([np.full(len(sets[k]), release(v), np.float32) for k, v in levels.items()]) \
            if levels else np.empty(0, np.float32)
        drive = (torch.from_numpy(idx), torch.from_numpy(rel)[None, :])
        for _ in range(start, end):
            spk = brain.step(none, none, graded=drive, plastic=plastic)
            plastic.step(spk, np.array([reward]), np.array([punish]))
            per_tick.append(int(spk.sum()))
    arrays = {"per_tick": np.array(per_tick, np.uint32), "counts": brain.counts()[0].astype(np.uint32),
              "plastic.w": plastic.w[0].numpy(), "plastic.slow": plastic.slow[0].numpy()}
    # The eyes, from their settled state, through a loom; the whole state kept at a few steps.
    state = eyes.state
    for step in range(EYE_STEPS):
        idx, level = eyes.step(loom(step))
        if step in EYE_KEPT:
            arrays[f"eyes.{step}"] = eyes.state.nodes.activity.numpy().astype(np.float32).ravel()
    arrays["eyes.level"] = level[0].numpy().astype(np.float32)
    eyes.state = state
    meta = {"schedule": [{"from": a, "to": b, "release": {k: release(v) for k, v in lv.items()}, "reward": r, "punish": p}
                         for a, b, lv, r, p in REFERENCE],
            "eye_steps": EYE_STEPS, "eye_kept": list(EYE_KEPT)}
    return arrays, meta


def main() -> None:
    from web.parity import cases

    OUT.mkdir(exist_ok=True)
    for old in ("brain.bin", "sets.json", "reference.bin"):  # the first packing's files
        (OUT / old).unlink(missing_ok=True)
    W, ann = load_connectome()
    sets = sided_sets(ann)
    arrays, meta, eyes = brain_bundle(W, ann, sets)
    write_bundle(OUT / "brain.bin.gz", arrays, meta)
    write_bundle(OUT / "eyes.bin.gz", *eyes_bundle(eyes))
    write_bundle(OUT / "reference.bin.gz", *reference_bundle(W, sets, eyes))
    with gzip.open(OUT / "parity.json.gz", "wt") as f:
        json.dump(cases(), f)
    for name in ("robot.json", "clips.json", "LICENSE-microduck"):
        shutil.copy(GODOT / name, OUT / name)
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:18} {p.stat().st_size / 1e6:6.1f} MB")


if __name__ == "__main__":
    main()
