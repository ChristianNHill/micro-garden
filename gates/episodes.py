"""Closed-loop episodes for gate checks: one stub per episode, every duck of every episode driven by one
batched brain, in lockstep over the contract."""
import tempfile

import numpy as np

from body.contract import Client
from body.stub2d.stub import DT, Stub
from brain.server import BrainServer


def run(W, ann, sets, episodes: list[dict], max_s: float, seed: int, port_base: int = 7700, until=None,
        **brain_kwargs):
    """episodes: Stub keyword args per episode, each with pose=(ducks, 3). brain_kwargs go to
    BrainServer (personality, starting physiology), one value per duck across all episodes.

    until(stubs) -> bool stops early. Returns (trajectory [steps, all ducks, 3], stubs); stubs are
    closed but keep their state (eaten, world).
    """
    dirs = [tempfile.TemporaryDirectory(prefix="mg") for _ in episodes]
    stubs, bodies, port = [], [], port_base
    for ep, d in zip(episodes, dirs):
        pose = np.asarray(ep["pose"], float).reshape(-1, 3)
        stubs.append(Stub(len(pose), seed, d.name, **{**ep, "pose": pose, "frame_port": port}))
        bodies += [(f"{d.name}/{name}.sock", port + i) for i, name in enumerate(stubs[-1].names)]
        port += len(pose)
    ctls = [Client(f"{d.name}/control.sock") for d in dirs]
    server = BrainServer(W, ann, sets, bodies, seed, **brain_kwargs)
    for c in ctls:
        c.call("sim.step", n=0)
    traj = []
    for _ in range(int(max_s / DT)):
        server.step(lockstep=True)
        for c in ctls:
            c.call("sim.step", n=1)
        traj.append(np.concatenate([s.pose for s in stubs]))
        if until is not None and until(stubs):
            break
    server.close()
    for c in ctls:
        c.close()
    for s in stubs:
        s.close()
    for d in dirs:
        d.cleanup()
    return np.array(traj), stubs


def ring_poses(rng, n: int, centre, radius: float) -> np.ndarray:
    """n single-duck poses on a circle around centre, facing random ways."""
    a = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a),
                            rng.uniform(-np.pi, np.pi, n)])
