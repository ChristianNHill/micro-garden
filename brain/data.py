"""FlyWire v783 loader: signed sparse connectome plus the named neuron sets.

Reads whichever files are in data/:
  annotations  Codex classification.csv.gz + neurons.csv.gz, else the public
               flywire_annotations Supplemental_file1_neuron_annotations.tsv. Only the public file
               carries pos_x/pos_y/pos_z, which brain/vision.py needs.
  connections  Codex connections.csv.gz, else Zenodo proofread_connections_783.feather
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_SYN = 5  # FlyWire convention; gives the ~2.7M connected pairs
# ORNs release more onto their own side (Gaudry et al. 2013, about 0.7); synapse counts can't show
# that, so contralateral ORN synapses are scaled. 0.3 exaggerates it: at 0.7 the left/right odor
# difference did not reach descending neurons.
ORN_CONTRA = 0.3
INHIBITORY = ["gaba", "glutamate"]  # as Shiu et al. 2024; GABA alone runs away on sustained odor

# Each set is (annotation column, regex matched against the whole value).
NAMED_SETS = {
    "sugar_grn": ("sub_class", r"sugar/water"),  # FlyWire lumps Gr5a/Gr64f sugar GRNs with water GRNs
    "orn_food": ("cell_type", r"ORN_(DM1|DM2|DM4|DM5|VA2)"),  # vinegar-attractive glomeruli
    "orn_danger": ("cell_type", r"ORN_(DA2|V)"),  # geosmin, CO2
    "orn_pheromone": ("sub_class", r"pheromone"),  # other ducks' smell
    "bristle": ("sub_class", r"(eye|head) bristle"),
    "heat": ("cell_type", r"TRN_VP2"),
    "cold": ("cell_type", r"TRN_VP3[ab]"),
    "moist_air": ("cell_type", r"HRN_VP5"),
    "dry_air": ("cell_type", r"HRN_VP4"),
    # sound only: the A and B neurons, not the wind or grooming ones
    "johnstons_organ": ("cell_type", r"JO-[AB].*"),
    # Wind: the E neurons answer an antenna pushed back, the C neurons one pulled forward (Yorozu et
    # al. 2009; Patella and Wilson 2018).
    "jo_push": ("cell_type", r"JO-E.*"),
    "jo_pull": ("cell_type", r"JO-C.*"),
    "lamina_L1": ("cell_type", r"L1"),
    "lamina_L2": ("cell_type", r"L2"),
    "LPLC2": ("cell_type", r"LPLC2"),
    "DNa02": ("cell_type", r"DNa02"),
    "DNp09": ("cell_type", r"DNp09"),
    "giant_fiber": ("cell_type", r"DNp01"),
    "moonwalker": ("cell_type", r"MDN"),
    # ipsilateral to one-sided odor in a held-out seed screen; not from literature
    "odor_steer": ("cell_type", r"DNb05|DNp05"),
    # fires for danger smell (0.6-0.8 Hz) and not for food, heat, cold, humidity, touch, sugar or
    # looming, more on the danger's side, on two held-out seed sets; not from literature
    "danger_valence": ("cell_type", r"DNp32"),
    # fires on the side the wind comes from: 8 cells a side, +3.2 Hz at 45 degrees off the nose, same on
    # a held-out seed, silent in the shuffled brain, almost no built-in left/right offset; not from literature
    "wind_steer": ("cell_type", r"DNge091"),
    # ipsilateral to one-sided humidity on two held-out seed sets; not from literature
    "moist_steer": ("cell_type", r"DNp12|DNp44"),
    # fires for touch only, on the side opposite the touch, two held-out seed sets
    "touch_steer": ("cell_type", r"DNg48"),
    "proboscis_mn": ("sub_class", r"proboscis_motor_neuron"),  # feeding readout; MN9 is unnamed in this release
    "kenyon_cells": ("class", r"Kenyon_Cell"),
    "MBON": ("class", r"MBON"),
    "PAM": ("cell_type", r"PAM\d+"),
    "PPL1": ("cell_type", r"PPL1\d+"),
    "aIPg": ("hemibrain_type", r"aIPg\d"),  # female aggression (Schretter et al. 2020)
    "pC1_aggr": ("cell_type", r"pC1[de]"),  # persistent social arousal / aggression (Deutsch et al. 2020)
    # DN types ranked highest for input from the grooming JO-F neurons and head and eye bristles (direct
    # plus one relay). Data-derived: FlyWire has no aBN/aDN labels.
    "grooming_dn": ("cell_type", r"DNg20|DNg84|DNg15|DNge133"),
}


def load_annotations() -> pd.DataFrame:
    """One row per neuron, indexed by root_id: class, sub_class, cell_type, hemibrain_type, side, nt (lowercase).

    The public TSV also carries pos_x/pos_y/pos_z, the neuron's representative point in FlyWire space.
    """
    cls, neu = DATA / "classification.csv.gz", DATA / "neurons.csv.gz"
    if cls.exists() and neu.exists():
        ann = pd.read_csv(cls, usecols=["root_id", "class", "sub_class", "cell_type", "hemibrain_type", "side"])
        nt = pd.read_csv(neu, usecols=["root_id", "nt_type"]).rename(columns={"nt_type": "nt"})
        ann = ann.merge(nt, on="root_id", how="left")
    else:
        ann = pd.read_csv(
            DATA / "Supplemental_file1_neuron_annotations.tsv", sep="\t",
            usecols=["root_id", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type", "side", "top_nt",
                     "pos_x", "pos_y", "pos_z"],  # brain/vision.py places optic-lobe cells by these
            dtype={"hemibrain_type": str},
        ).rename(columns={"cell_class": "class", "cell_sub_class": "sub_class", "top_nt": "nt"})
    ann["nt"] = ann["nt"].fillna("").str.lower()
    return ann.drop_duplicates("root_id").set_index("root_id")


def load_connections() -> pd.DataFrame:
    """Columns pre, post (root ids), syn_count. May hold several rows per pair (one per neuropil)."""
    codex = DATA / "connections.csv.gz"
    if codex.exists():
        df = pd.read_csv(codex, usecols=["pre_root_id", "post_root_id", "syn_count"])
    else:
        df = pd.read_feather(DATA / "proofread_connections_783.feather",
                             columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    df.columns = ["pre", "post", "syn_count"]
    return df


def load_connectome():
    """Return (W, ann). W[post, pre] = ±synapse count as float32 CSR, sign -1 when pre is INHIBITORY.

    Pairs with fewer than MIN_SYN synapses (summed over neuropils) are dropped, and ORN synapses onto
    the opposite side are scaled by ORN_CONTRA.

    Row/column i is ann.index[i]. Edges touching neurons missing from ann are dropped.
    """
    ann = load_annotations()
    conn = load_connections()
    idx = pd.Index(ann.index)
    pre, post = idx.get_indexer(conn["pre"]), idx.get_indexer(conn["post"])
    keep = (pre >= 0) & (post >= 0)
    sign = np.where(np.isin(ann["nt"].to_numpy(), INHIBITORY), -1.0, 1.0).astype(np.float32)
    n = len(idx)
    w = conn["syn_count"].to_numpy(np.float32)[keep] * sign[pre[keep]]
    W = sparse.csr_matrix((w, (post[keep], pre[keep])), shape=(n, n))  # duplicates summed
    W.data[abs(W.data) < MIN_SYN] = 0
    W.eliminate_zeros()
    side = ann["side"].to_numpy()
    orn = (ann["class"] == "olfactory").to_numpy()
    rows = np.repeat(np.arange(n), np.diff(W.indptr))
    W.data[orn[W.indices] & (side[rows] != side[W.indices])] *= ORN_CONTRA
    return W, ann


def shuffled(W: sparse.csr_matrix, seed: int) -> sparse.csr_matrix:
    """Control brain: presynaptic partners permuted across all edges, each weight travelling with its source.

    Every neuron keeps its in-degree and out-degree; who talks to whom is destroyed.
    """
    coo = W.tocoo()
    perm = np.random.default_rng(seed).permutation(coo.nnz)
    return sparse.csr_matrix((coo.data[perm], (coo.row, coo.col[perm])), shape=W.shape)


def named_sets(ann: pd.DataFrame) -> dict[str, np.ndarray]:
    """Row indices into W for each named set."""
    return {
        name: np.flatnonzero(ann[col].fillna("").str.fullmatch(pattern))
        for name, (col, pattern) in NAMED_SETS.items()
    }


def column_file() -> Path | None:
    """Optic-lobe column assignment file (Codex Visual Columns challenge or Matsliah 2024 SD2), if present."""
    return next(iter(sorted(DATA.glob("*column*"))), None)
