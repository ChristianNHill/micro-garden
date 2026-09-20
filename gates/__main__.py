"""Every gate in order, one table at the end (PLAN.md, Verification).

Run: uv run python -m gates            all of them, which takes hours
     uv run python -m gates 6 7 9      only those
     uv run python -m gates --fast     only the ones that finish in about a minute

Each gate is its own process, because they are long, they hold UDP ports, and one falling over should
not take the rest with it. The table is the Phase B exit question in one place: which gates pass, what
the number behind each was, and how long it took to find out.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

GATES = ["00_data", "01_tick", "02_sugar", "03_stub", "04_seek", "04b_senses", "04c_temperament",
         "05_personality", "06_vision", "07_learn", "08_social", "08b_toys", "09_persist", "09b_soak"]
FAST = {"00_data", "01_tick", "02_sugar", "03_stub", "06_vision", "07_learn", "09_persist"}
# The line worth putting in the table, per gate. First capture group wins; nothing means no number.
HEADLINE = {
    "00_data": r"(\d[\d,]* neurons[^\n]*)",
    "02_sugar": r"sugar on\s*:\s*(proboscis MN [\d.]+ Hz)",
    "03_stub": r"(odor gradient[^\n]*)",
    "04_seek": r"real\s+(found \d+/\d+\s+median [\d.]+ s)",
    "04b_senses": r"pond\s+real\s+(found \d+/\d+)",
    "04c_temperament": r"aggressiveness [^:]*: (first headbutt after \[[^\]]*\])",
    "05_personality": r"(signature distance: .*)",
    "06_vision": r"loom\s+LPLC2\s+[\d.]+\s*->\s*([\d.]+ Hz)",
    "07_learn": r"weights, paired duck: (odor A's own synapses [\d.]+)",
    "08_social": r"one dish, [^:]*:\s+(share of the bites .*)",
    "08b_toys": r"(a hat stays on [^\n]*)",
    "09b_soak": r"soak: ([^\n]*)",
    "09_persist": r"(three days away catches up in [\d.]+ s)",
}


def run_gate(name: str, extra: list[str]) -> tuple[str, str, float]:
    """Returns (verdict, headline, seconds)."""
    t0 = time.perf_counter()
    out = subprocess.run([sys.executable, "-m", f"gates.gate_{name}", *extra],
                         capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
    took = time.perf_counter() - t0
    text = out.stdout
    verdict = "PASS" if re.search(r"^PASS$", text, re.M) else "FAIL" if re.search(r"^FAIL$", text, re.M) else "ERROR"
    line = ""
    if pattern := HEADLINE.get(name):
        if m := re.search(pattern, text, re.M):
            line = m.group(1).strip()
    if verdict == "ERROR":
        line = (out.stderr.strip().splitlines() or ["no output"])[-1][:60]
    return verdict, line, took


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="gate numbers, e.g. 6 7 9; default is all of them")
    ap.add_argument("--fast", action="store_true", help="only the gates that finish in about a minute")
    args, extra = ap.parse_known_args()

    chosen = GATES
    if args.fast:
        chosen = [g for g in GATES if g in FAST]
    if args.which:
        want = {w.lstrip("0") or "0" for w in args.which}
        chosen = [g for g in chosen if (g.split("_")[0].lstrip("0") or "0") in want or g.split("_")[0] in want]
    if not chosen:
        print("no gates matched")
        return 2

    print(f"{'gate':18s} {'result':7s} {'wall':>8s}  what it measured")
    print("-" * 96)
    results = {}
    for name in chosen:
        verdict, line, took = run_gate(name, extra)
        results[name] = verdict
        mins = f"{took:5.0f} s" if took < 90 else f"{took / 60:5.1f} m"
        print(f"gate {name:13s} {verdict:7s} {mins:>8s}  {line}", flush=True)
    print("-" * 96)
    bad = [n for n, v in results.items() if v != "PASS"]
    print(f"{len(results) - len(bad)} of {len(results)} pass" + (f"; still to fix: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
