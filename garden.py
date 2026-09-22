"""The garden, in one command: `uv run python -m garden`. Five ducks on five brains in a Godot window,
saved when the window closes and aged on the next start. Any body/stub2d/stub.py argument can follow."""
import sys

from body.stub2d.stub import main

sys.argv[1:1] = ["--brain", "--godot"]
main()
