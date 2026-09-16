"""FlyWire v783 loader: signed sparse connectome plus the named neuron sets (PLAN.md Gate 0).

Reads whichever files are in data/:
  annotations  Codex classification.csv.gz + neurons.csv.gz, else the public
               flywire_annotations Supplemental_file1_neuron_annotations.tsv
  connections  Codex connections.csv.gz, else Zenodo proofread_connections_783.feather
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA = Path(__file__).resolve().parent.parent / "data"
MIN_SYN = 5  # FlyWire convention; gives the ~2.7M connected pairs ARCHITECTURE.md assumes

# ARCHITECTURE.md §3.6. Each set is (annotation column, regex matched against the whole value).
NAMED_SETS = {
    "sugar_grn": ("sub_class", r"sugar/water"),  # FlyWire lumps Gr5a/Gr64f sugar GRNs with water GRNs
    "orn_food": ("cell_type", r"ORN_(DM1|DM2|DM4|DM5|VA2)"),  # vinegar-attractive glomeruli
    "orn_danger": ("cell_type", r"ORN_(DA2|V)"),  # geosmin, CO2
    "bristle": ("sub_class", r"(eye|head) bristle"),
    "johnstons_organ": ("cell_type", r"JO-.*"),
    "lamina_L1": ("cell_type", r"L1"),
    "lamina_L2": ("cell_type", r"L2"),
    "LPLC2": ("cell_type", r"LPLC2"),
    "DNa02": ("cell_type", r"DNa02"),
    "DNp09": ("cell_type", r"DNp09"),
    "giant_fiber": ("cell_type", r"DNp01"),
    "moonwalker": ("cell_type", r"MDN"),
    "proboscis_mn": ("sub_class", r"proboscis_motor_neuron"),  # feeding readout; MN9 is unnamed in this release
    "kenyon_cells": ("class", r"Kenyon_Cell"),
    "MBON": ("class", r"MBON"),
    "PAM": ("cell_type", r"PAM\d+"),
    "PPL1": ("cell_type", r"PPL1\d+"),
    "grooming_dn": ("cell_type", r"DNg12_[a-z]"),  # antennal grooming DNs; confirm the type choice at Gate 0 review
}


def load_annotations() -> pd.DataFrame:
    """One row per neuron, indexed by root_id, columns: class, sub_class, cell_type, side, nt (lowercase)."""
    cls, neu = DATA / "classification.csv.gz", DATA / "neurons.csv.gz"
    if cls.exists() and neu.exists():
        ann = pd.read_csv(cls, usecols=["root_id", "class", "sub_class", "cell_type", "side"])
        nt = pd.read_csv(neu, usecols=["root_id", "nt_type"]).rename(columns={"nt_type": "nt"})
        ann = ann.merge(nt, on="root_id", how="left")
    else:
        ann = pd.read_csv(
            DATA / "Supplemental_file1_neuron_annotations.tsv", sep="\t",
            usecols=["root_id", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt"],
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


def load_connectome(ann: pd.DataFrame | None = None):
    """Return (W, ann). W[post, pre] = ±synapse count as float32 CSR, sign -1 when pre is GABAergic.

    Pairs with fewer than MIN_SYN synapses (summed over neuropils) are dropped.

    Row/column i is ann.index[i]. Edges touching neurons missing from ann are dropped.
    """
    ann = load_annotations() if ann is None else ann
    conn = load_connections()
    idx = pd.Index(ann.index)
    pre, post = idx.get_indexer(conn["pre"]), idx.get_indexer(conn["post"])
    keep = (pre >= 0) & (post >= 0)
    sign = np.where(ann["nt"].to_numpy() == "gaba", -1.0, 1.0).astype(np.float32)
    n = len(idx)
    w = conn["syn_count"].to_numpy(np.float32)[keep] * sign[pre[keep]]
    W = sparse.csr_matrix((w, (post[keep], pre[keep])), shape=(n, n))  # duplicates summed
    W.data[abs(W.data) < MIN_SYN] = 0
    W.eliminate_zeros()
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
