"""The brain, to look at: which neurons to show, where each sits, and which of them just fired.

A viewer cannot be sent 139,000 neurons fifty times a second, so it is shown SHOWN of them: the neurons the
garden names (the senses going in, the mushroom body's cast), every descending and motor neuron, and a random
draw of the rest, so the picture has the whole brain's shape and the cells that matter are all in it. Their
places are written once to a file the viewer reads (front view: x across, y down, as FlyWire has them), with
a group each for colour. After that a step's news is only the indices that spiked.

The places are FlyWire's and are not redistributed: the file is made here, from the data the user downloaded,
into ~/.cache/micro-garden.
"""
import json
from pathlib import Path

import numpy as np
import torch

SHOWN = 12000
POINTS = Path.home() / ".cache" / "micro-garden" / "brain_points.json"
GROUPS = ("optic lobe", "mushroom body", "senses", "descending, motor", "central")


def group_of(ann) -> np.ndarray:
    cls = ann["class"].fillna("").astype(str)
    g = np.full(len(ann), 4)
    g[cls.str.contains(r"^(?:ME|LO|LA|LOP)|visual|optic", regex=True)] = 0
    g[cls.str.contains("Kenyon|MBON|DAN|MBIN", regex=True)] = 1
    g[cls.str.contains("sensory|olfactory|gustatory|hygro|thermo", regex=True)] = 2
    # the annotations give descending neurons no class; their cell types all begin DN
    g[ann["cell_type"].fillna("").astype(str).str.startswith("DN").to_numpy() | cls.str.contains("motor").to_numpy()] = 3
    return g


class BrainView:
    def __init__(self, ann, sets: dict, device, seed: int = 0):
        rng = np.random.default_rng(seed)
        groups = group_of(ann)
        named = np.unique(np.concatenate([np.asarray(v).ravel() for v in sets.values()]))
        named = named[rng.permutation(len(named))][:SHOWN // 2]
        named = np.union1d(named, np.flatnonzero(groups == 3))  # and every neuron that goes down to the legs
        rest = np.setdiff1d(np.arange(len(ann)), named)
        self.idx = np.sort(np.concatenate([named, rng.choice(rest, SHOWN - len(named), replace=False)]))
        self.index = torch.from_numpy(self.idx).to(device)
        self.fired = torch.zeros(SHOWN, device=device)
        self.duck = -1  # whose brain is being watched, or nobody's
        xy = ann[["pos_x", "pos_y"]].to_numpy(float)[self.idx]
        xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0)).max()  # one scale for both, so the brain keeps its shape
        POINTS.parent.mkdir(parents=True, exist_ok=True)
        POINTS.write_text(json.dumps({"x": xy[:, 0].round(4).tolist(), "y": xy[:, 1].round(4).tolist(),
                                      "group": groups[self.idx].tolist(), "groups": GROUPS,
                                      "of": len(ann)}, separators=(",", ":")))

    def tick(self, spk: torch.Tensor) -> None:
        """One brain tick's spikes, (ducks, neurons). Costs nothing while nobody is watching."""
        if self.duck >= 0:
            self.fired += spk[self.duck, self.index]

    def take(self) -> list[int]:
        """Which shown neurons fired since the last take, as positions in the points file."""
        if self.duck < 0:
            return []
        out = self.fired.nonzero().flatten().cpu().tolist()
        self.fired.zero_()
        return out


if __name__ == "__main__":
    from brain.data import load_connectome, named_sets
    W, ann = load_connectome()
    view = BrainView(ann, named_sets(ann), "cpu")
    points = json.loads(POINTS.read_text())
    assert len(points["x"]) == SHOWN == len(set(view.idx.tolist())) and 0 <= min(points["y"]) and max(points["x"]) <= 1
    spk = torch.zeros(2, len(ann))
    spk[1, view.idx[[5, 77]]] = 1
    view.tick(spk); assert view.take() == [], "nobody is watching"
    view.duck = 1
    view.tick(spk); view.tick(spk)
    assert view.take() == [5, 77] and view.take() == []
    counts = np.bincount(points["group"], minlength=5)
    print(f"ok  {SHOWN} of {points['of']} neurons shown: " + ", ".join(f"{n} {g}" for g, n in zip(GROUPS, counts))
          + f"; {POINTS.stat().st_size / 1024:.0f} KB")
