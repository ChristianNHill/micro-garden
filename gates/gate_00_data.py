"""Gate 0: FlyWire data loads and every named neuron set is non-empty.

Run: uv run python -m gates.gate_00_data
"""
import sys

import pandas as pd

from brain.data import column_file, load_connectome, named_sets


def main() -> int:
    W, ann = load_connectome()
    n_syn = abs(W).sum()
    print(f"neurons {W.shape[0]:,}  connected pairs {W.nnz:,}  synapses {n_syn:,.0f}  "
          f"inhibitory pairs {(W.data < 0).sum():,}")

    sets = named_sets(ann)
    for name, rows in sets.items():
        types = ann.iloc[rows]["cell_type"].nunique()
        print(f"  {name:16s} {len(rows):6d} neurons  {types:3d} cell types")
    empty = [k for k, v in sets.items() if len(v) == 0]

    cols = column_file()
    if cols is None:
        print("column assignment: ABSENT (Gate 6 falls back to flyvis unless a file lands in data/)")
    else:
        df = pd.read_csv(cols, sep=None, engine="python")
        col = next((c for c in df.columns if "column" in c.lower()), df.columns[-1])
        print(f"column assignment: {cols.name}  {len(df):,} rows  {df[col].nunique():,} columns "
              f"(expect ~23k neurons over 796 columns)")

    if empty:
        print(f"FAIL: empty sets {empty}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
