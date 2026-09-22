"""Gate 14: the look. Most of this gate is judged by eye; the part a machine can check is the size.

Run: uv run python -m gates.gate_14_look

The visual brief asks for a Dreamcast-sized garden: under 300 KB of assets.
The Godot garden is built from primitives in script, so what counts is its data: the robot's meshes, its
recorded motions, and anything else that is not code. Scripts and shaders do not count, since the budget
is for assets; a texture, model or sound that creeps in later is still caught here.

Judged by eye, not asserted: shimmer under the camera's drift, whether a duck keeps
its silhouette in the halftone, the palette, the ride view, and the quacks.
"""
import os
import sys

from gates.episodes import verdict

PROJECT = os.path.join(os.path.dirname(__file__), "..", "viewer", "godot")
BUDGET_KB = 300
CODE = (".py", ".gd", ".gdshader", ".uid", ".tscn", ".godot")  # code and Godot's own project files, not assets


def main() -> int:
    sizes = {os.path.relpath(os.path.join(d, f), PROJECT): os.path.getsize(os.path.join(d, f))
             for d, _, files in os.walk(PROJECT) if ".godot" not in d.split(os.sep) for f in files
             if not f.endswith(CODE)}
    total = sum(sizes.values()) / 1024
    for name, size in sorted(sizes.items(), key=lambda x: -x[1])[:5]:
        print(f"  {size / 1024:6.1f} KB  {name}")
    print(f"the Godot garden is {total:.0f} KB in {len(sizes)} files")
    return verdict({f"under {BUDGET_KB} KB": 0 < total < BUDGET_KB})


if __name__ == "__main__":
    sys.exit(main())
