"""Gate 2: sugar in, feeding out; looming in, escape out. Real brain beside the shuffled control.

Run: uv run python -m gates.gate_02_sugar
Asserts on the real brain only; the shuffled brain is printed for the Gate 2 decision.

2026-09-16 (Chris): the plan's "sugar raises DNp09" cannot hold. DNp09's top inputs are the visual
LC9 and LC31a, and sugar never reaches it. Sugar now tests feeding (proboscis motor neurons, as in
Shiu et al. 2024); walking toward food comes from vision and odor in Gate 4.
Decision: real vs shuffled is plainly visible. Real: sugar fires proboscis MNs, looming fires the
giant fiber in 10 ms. Shuffled: neither.
"""
import sys

import numpy as np

from brain.data import load_connectome, named_sets, shuffled
from brain.decoder import Decoder
from brain.encoder import encode
from brain.lif import DT_MS, LIF

SUGAR_TICKS = 300  # 3 s
LOOM_ONSET, GF_DEADLINE_MS = 5, 50


def sugar_rates(W, ann, sets, level):
    """Mean smoothed decoder group rates (fwd, back, left, right, gf, feed) over the run."""
    rng = np.random.default_rng(0)
    brain, dec = LIF(W, 1), Decoder(ann, sets, 1)
    acc = np.zeros(dec.rates.shape[1])
    for _ in range(SUGAR_TICKS):
        dec.update(brain.step(*encode(rng, sets, {"sugar_grn": level}, 1)))
        acc += dec.rates[0]
    return acc / SUGAR_TICKS, int((brain.counts() > 0).sum())


def gf_latency_ms(W, sets):
    """Time from looming onset to the first giant fiber spike (the decoder's escape rule is stricter)."""
    rng = np.random.default_rng(0)
    brain = LIF(W, 1)
    for t in range(LOOM_ONSET + 20):
        spk = brain.step(*encode(rng, sets, {"LPLC2": float(t >= LOOM_ONSET)}, 1))
        if t >= LOOM_ONSET and spk[0, sets["giant_fiber"]].any():
            return (t - LOOM_ONSET) * DT_MS
    return None


def main() -> int:
    W, ann = load_connectome()
    sets = named_sets(ann)
    results = {}
    for label, M in (("real", W), ("shuffled", shuffled(W, seed=0))):
        off, n_off = sugar_rates(M, ann, sets, 0.0)
        on, n_on = sugar_rates(M, ann, sets, 1.0)
        lat = gf_latency_ms(M, sets)
        results[label] = (off, on, lat)
        print(f"{label}:")
        for name, r, n in (("off", off, n_off), ("on ", on, n_on)):
            print(f"  sugar {name}: proboscis MN {r[5]:.2f} Hz  DNp09 {r[0]:.2f}  MDN {r[1]:.2f}  "
                  f"DNa02 L/R {r[2]:.2f}/{r[3]:.2f}  active neurons {n:,}")
        print(f"  looming: giant fiber latency {'none' if lat is None else f'{lat:.0f} ms'}")

    off, on, lat = results["real"]
    checks = {
        "sugar raises proboscis MN rate": on[5] > off[5] + 0.5,
        f"looming fires giant fiber within {GF_DEADLINE_MS} ms": lat is not None and lat <= GF_DEADLINE_MS,
    }
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
