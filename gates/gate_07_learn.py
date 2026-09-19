"""Gate 7: an odor paired with sugar changes what the mushroom body says about it.

Run: uv run python -m gates.gate_07_learn

Two odors, told apart the way a fly tells them apart: different glomeruli. Odor A is ORN_DM1 and
ORN_DM2, odor B is ORN_DM4 and ORN_DM5, both drawn from the vinegar-attractive set the ducks already
use. A is paired with sugar for TRIALS trials, B is never paired, and B is the control that says any
change is learning rather than drift.

The readout is the MBON population, not a walk toward a smell. The 2D world carries one food odor
field, so two odors cannot be put in two places until other ducks start smelling of themselves at
Gate 8; and the MBONs are where the mechanism acts, so a null there would mean nothing downstream
could work either. Depression at Kenyon cell -> MBON is what learning is here: after pairing, the
paired odor should drive the MBONs less, which is how a fly stops being told to leave something alone.

Two brains run side by side and see exactly the same odors; only the first gets dopamine. Comparing
them, rather than one brain before against after, is what makes the measurement mean anything: the
first attempt did the latter and the unpaired control odor fell further than the paired one, because
spike-frequency adaptation builds over a long run and drags every rate down with it. A yoked control
subtracts that exactly.

Specificity is read at the synapses rather than off the MBON rate. With a sparse code only about 5% of
the 21,438 synapses belong to any one odor, so the mean weight barely moves even when exactly the right
ones are wiped out, and 96 MBONs firing under half a hertz cannot resolve the difference either. The
weights themselves can: odor A's own synapses should be depressed and odor B's should not.

STATUS 2026-09-18: PASSING. A duck learns about one smell in particular.
- Odor A's own synapses end at 0.526 of baseline and odor B's at 0.660, on a brain that saw exactly
  the same odors as its yoked control and differed only in getting dopamine. The control's weights are
  1.000 to three places, and everything climbs back to 0.983 over five minutes.
- Two things had to be true first, and neither was.
  1. A sparse odor code. 28% of Kenyon cells answered any odor and two odors shared 79% of their cells.
     `plasticity.sparsen` raises their threshold until about 5% answer, the figure measured in the fly.
  2. A repeatable one. Even sparsened, the same odor twice lit different cells (overlap 0.59) about as
     often as two different odors did (0.44), because `encode` fires a random subset of the receptors
     each tick. Driving them graded instead, at the same mean current, makes the same odor give the
     same cells every time (1.00) while two odors still differ (0.48). The connectome was never the
     problem: it drives 826 Kenyon cells from odor A alone and 532 from B alone, correlation 0.21.
- Not yet true: the MBON rates do not show it (odor A 0.47 against its control's 0.52 Hz, odor B 0.38
  against 0.47). 96 cells under half a hertz is a handful of spikes, which is why the checks read the
  synapses, but it does mean the learning is not yet visible in the signal that would steer a duck.
  Gate 8 gives ducks each other's odors and a place to walk toward, which is where that gets tested.
"""
import sys
import time

import numpy as np

from brain.data import load_connectome, named_sets
from brain.encoder import graded
from brain.lif import DEVICE, LIF
from brain.plasticity import RECOVER_S, Plasticity, sparsen, strip
from gates.episodes import verdict

ODOR_LEVEL = 0.6
TRIALS, TRIAL_TICKS, GAP_TICKS = 12, 100, 50  # 1 s of odor with sugar, 0.5 s between
TEST_TICKS = 200  # 2 s of odor alone, no dopamine
REWARD = 1.0
EMPTY = np.empty(0, np.int64)
FORGET_MIN = (0, 1, 3, 5)  # minutes of quiet to log the forgetting curve at


def odor_sets(ann) -> dict:
    """Two odors as two pairs of glomeruli, plus the MBON readout."""
    ct = ann["cell_type"].fillna("")
    return {name: np.flatnonzero(ct.str.fullmatch(pattern))
            for name, pattern in (("A", r"ORN_(DM1|DM2)"), ("B", r"ORN_(DM4|DM5)"))}


def responders(brain, plastic, sets, odors, which, rng) -> np.ndarray:
    """Which Kenyon cells (as positions in the KC set) answer one odor. No dopamine, so nothing is learnt."""
    kc = sets["kenyon_cells"]
    seen = np.zeros(len(kc), bool)
    for _ in range(TEST_TICKS):
        spk = brain.step(EMPTY, EMPTY, graded=graded({"odor": odors[which]}, {"odor": ODOR_LEVEL}, 2, DEVICE, rng))
        seen |= spk[0, kc].cpu().numpy()
    return np.flatnonzero(seen)


def present(brain, plastic, sets, odors, which, ticks, reward, rng) -> np.ndarray:
    """One odor for `ticks`, dopamine per duck. Returns each duck's MBON population rate in Hz."""
    mbon = sets["MBON"]
    count = np.zeros(len(reward))
    for _ in range(ticks):
        spk = brain.step(EMPTY, EMPTY, plastic=plastic,
                         graded=graded({"odor": odors[which]}, {"odor": ODOR_LEVEL}, len(reward), DEVICE, rng))
        plastic.step(spk, reward, np.zeros_like(reward))
        count += spk[:, mbon].sum(1).cpu().numpy()
    return count / len(mbon) / (ticks * 0.01)


def main() -> int:
    t0 = time.perf_counter()
    W, ann = load_connectome()
    sets = named_sets(ann)
    odors = odor_sets(ann)
    print(f"odor A: {len(odors['A'])} ORNs (DM1, DM2)   odor B: {len(odors['B'])} ORNs (DM4, DM5)")

    brain = LIF(strip(W, sets), 2, DEVICE)  # duck 0 is paired, duck 1 is the yoked control
    plastic = Plasticity(W, sets, 2, DEVICE)
    sparsen(brain, sets)  # without a sparse odor code there is nothing odor-specific to learn
    rng = np.random.default_rng(0)
    paired, none = np.array([REWARD, 0.0]), np.zeros(2)
    kc_a = responders(brain, plastic, sets, odors, "A", rng)
    kc_b = responders(brain, plastic, sets, odors, "B", rng)

    for _ in range(TRIALS):
        present(brain, plastic, sets, odors, "A", TRIAL_TICKS, paired, rng)
        present(brain, plastic, sets, odors, "A", GAP_TICKS, none, rng)
    trained, control = plastic.strength()
    after = {k: present(brain, plastic, sets, odors, k, TEST_TICKS, none, rng) for k in "AB"}

    pre = plastic.pre_local.cpu().numpy()
    w = plastic.w.detach().cpu().numpy() / plastic.base.detach().cpu().numpy()
    a_only, b_only = np.setdiff1d(kc_a, kc_b), np.setdiff1d(kc_b, kc_a)
    own = w[0][np.isin(pre, a_only)].mean()
    others = w[0][np.isin(pre, b_only)].mean()
    print(f"Kenyon cells answering A {len(kc_a)}, B {len(kc_b)}, shared {len(np.intersect1d(kc_a, kc_b))}")
    print(f"weights, paired duck: odor A's own synapses {own:.3f} of baseline, "
          f"odor B's own {others:.3f}   (all synapses {trained:.3f}, control duck {control:.3f})")
    print(f"MBON rate after training, paired against yoked control: " +
          "  ".join(f"odor {k} {after[k][0]:.2f} vs {after[k][1]:.2f} Hz" for k in "AB"))
    curve = []
    for a, b in zip(FORGET_MIN, FORGET_MIN[1:]):
        for _ in range(int((b - a) * 60 * 100)):
            plastic.step(brain.step(EMPTY, EMPTY, plastic=plastic),
                         none, none)
        curve.append((b, plastic.strength()))
    print("forgetting: " + "  ".join(f"{m} min {s[0]:.3f}" for m, s in curve)
          + f"   (recovery constant {RECOVER_S / 60:.0f} min)")
    print(f"({time.perf_counter() - t0:.0f} s wall)")

    return verdict({
        "the yoked control's weights are untouched": control > 0.999,
        "the paired odor's own synapses are depressed": own < 0.9,
        "the unpaired odor's own synapses are spared": others > own + 0.05,
        "the weights recover once the pairing stops": curve[-1][0] > trained,
    })


if __name__ == "__main__":
    sys.exit(main())
