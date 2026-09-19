"""Closed-loop episodes for gate checks: one stub per episode, every duck of every episode driven by one
batched brain, in lockstep over the contract. Also the shared pass/fail report."""
import socket
import tempfile
from contextlib import ExitStack, closing

import numpy as np

from body import frames
from body.contract import Client
from body.stub2d.stub import DT, Stub
from brain.server import BrainServer


def free_port_base(wanted: int, count: int, tries: int = 40) -> int:
    """A block of `count` UDP ports nobody else holds, starting at or after `wanted`.

    Gates bind a port per duck and two running at once used to collide on the default, which kills one
    of them partway through a long run. Asking the operating system is cheaper than remembering.
    """
    for attempt in range(tries):
        base = wanted + attempt * 64
        probes = []
        try:
            for i in range(count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((frames.HOST, base + i))
                probes.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in probes:
                sock.close()
    raise OSError(f"no free block of {count} ports from {wanted}")


def run(W, ann, sets, episodes: list[dict], max_s: float, seed: int, port_base: int = 7700, until=None,
        **brain_kwargs):
    """episodes: Stub keyword args per episode, each with pose=(ducks, 3). brain_kwargs go to
    BrainServer (personality, starting physiology), one value per duck across all episodes.

    until(stubs) -> bool stops early. Returns (trajectory [steps, all ducks, 3], stubs); stubs are
    closed but keep their state (eaten, world).
    """
    with ExitStack() as cleanup:
        port_base = free_port_base(port_base, sum(len(np.asarray(e["pose"], float).reshape(-1, 3))
                                                  for e in episodes))
        stubs, bodies, ctls, port = [], [], [], port_base
        for ep in episodes:
            d = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="mg"))
            pose = np.asarray(ep["pose"], float).reshape(-1, 3)
            stubs.append(cleanup.enter_context(closing(Stub(len(pose), seed, d, **{**ep, "pose": pose, "frame_port": port}))))
            bodies += [(f"{d}/{name}.sock", port + i) for i, name in enumerate(stubs[-1].names)]
            ctls.append(cleanup.enter_context(closing(Client(f"{d}/control.sock"))))
            port += len(pose)
        server = cleanup.enter_context(closing(BrainServer(W, ann, sets, bodies, seed, **brain_kwargs)))
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
    return np.array(traj), stubs


def ring_poses(rng, n: int, centre, radius: float) -> np.ndarray:
    """n single-duck poses on a circle around centre, facing random ways."""
    a = rng.uniform(-np.pi, np.pi, n)
    return np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a),
                            rng.uniform(-np.pi, np.pi, n)])


def verdict(checks: dict[str, bool]) -> int:
    """Print each check and the overall result; returns the exit code."""
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1
