"""The garden, in one command: `uv run python -m garden`. Five ducks on five brains, the Godot window over them,
saved when the window is closed and aged on the way back in. Anything body/stub2d/stub.py takes can follow."""
import sys

from body.stub2d.stub import main

sys.argv[1:1] = ["--brain", "--godot"]
main()
