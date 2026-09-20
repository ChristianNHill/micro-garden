# Micro Garden

Five microducks in a garden, each driven by its own running copy of the fruit fly connectome
(FlyWire v783, about 139k neurons). They smell, see, hear, get hungry and sleepy, learn, and have
personalities. Nothing about what a duck does is scripted; the player changes the garden and the
brains respond.

- `RESEARCH.md`: the goal (top of file) and everything it was decided from.
- `ARCHITECTURE.md`: the design and the decisions ledger (§0).
- `PLAN.md`: the gated build plan, with what each gate actually found.
- `ATTRIBUTION.md`: data and model licenses.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and the two FlyWire files in `data/` (PLAN.md Gate 0).

```
uv run python -m body.stub2d.stub --view --brain
```

Click a duck to follow it and the tree to shake fruit down. Keys: `P` pet the selected duck, `C` clap,
`F` feed by hand at the mouse, `M` music at the mouse, `H` hat on or off. The garden saves when you
close it and is aged by however long you were away when you come back (`--fresh` to start over,
`--learns` to let the ducks learn, `--labels Bully,Napper,...` to choose who hatches).

## Check it

One gate is one runnable check. `uv run python -m gates` runs them all (hours); `--fast` runs the
ones that take about a minute; `uv run python -m gates 6 9` runs those two. Run one GPU job at a time.
