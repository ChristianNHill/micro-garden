"""Gate 3: five stub ducks driven over the contract are deterministic; odor points to food.

Run: uv run python -m gates.gate_03_stub
A seeded random walker sends robot.move to each duck's socket, then sim.step over control.sock,
for 60 simulated seconds. Two runs must end at identical poses.
"""
import sys
import tempfile

import numpy as np

from body import frames
from body.contract import Client
from body.stub2d.stub import DT, Stub
from world.fields import CELL_M, SIZE_M, World

N, SIM_S, HOLD_STEPS = 5, 60.0, 25
PORT_BASE = 7650  # away from the live stub's 7601


def drain(receivers, last):
    for i, r in enumerate(receivers):
        last[i] = frames.latest(r, last[i])


def walk(seed: int, receivers):
    rng = np.random.default_rng(seed)
    last = [None] * N
    with tempfile.TemporaryDirectory(prefix="mg") as d:
        stub = Stub(N, seed, d, frame_port=PORT_BASE)
        start = stub.pose.copy()
        drain(receivers, [None] * N)
        ducks = [Client(f"{d}/{name}.sock") for name in stub.names]
        ctl = Client(f"{d}/control.sock")
        ducks[0].call("robot.head", head_yaw=0.2)
        ducks[1].call("robot.do", skill="ground_pick")
        for _ in range(int(SIM_S / DT / HOLD_STEPS)):
            for c in ducks:
                c.call("robot.move", vx=rng.uniform(-0.1, 0.3), vy=0.0, vyaw=rng.uniform(-1.5, 1.5))
            ctl.call("sim.step", n=HOLD_STEPS)
            drain(receivers, last)
        state = ctl.call("sim.state")
        errors = []
        for method, params in (("robot.fly", {}), ("robot.move", {"speed": 1.0})):
            try:
                ducks[0].call(method, **params)
            except RuntimeError as e:
                errors.append(str(e))
        for c in ducks + [ctl]:
            c.close()
        stub.close()
    return start, np.array(state["pose"]), last, state["t"], errors


def odor_points_to_food(rng) -> tuple[int, int]:
    food = np.array([[3.0, 3.0], [1.0, 1.0], [3.2, 0.8]])
    world = World(food)
    pts = rng.uniform(0.2, SIZE_M - 0.2, (2000, 2))
    dist = np.linalg.norm(pts[:, None] - food[None], axis=-1)
    near, d = dist.argmin(axis=1), np.sort(dist, axis=1)
    use = (d[:, 0] < 0.7 * d[:, 1]) & (d[:, 0] > 3 * CELL_M)
    g = world.odor_gradient(pts[use])
    to_food = food[near[use]] - pts[use]
    good = (g * to_food).sum(axis=1) > 0
    return int(good.sum()), int(use.sum())


def main() -> int:
    receivers = [frames.receiver(PORT_BASE + i) for i in range(N)]
    start, pose_a, frames_a, t, errors = walk(3, receivers)
    _, pose_b, _, _, _ = walk(3, receivers)
    _, pose_c, _, _, _ = walk(4, receivers)
    print(f"simulated {t:.2f} s; final poses run A:\n{np.round(pose_a, 3)}")
    good, total = odor_points_to_food(np.random.default_rng(0))
    print(f"odor gradient points to nearest food at {good}/{total} sample points")
    frame_ok = all(f is not None and abs(f["x"] - p[0]) < 1e-4 and abs(f["y"] - p[1]) < 1e-4
                   for f, p in zip(frames_a, pose_a))
    print(f"errors for bad calls: {errors}")

    checks = {
        "same seed, identical final poses": np.array_equal(pose_a, pose_b),
        "different seed, different poses": not np.array_equal(pose_a, pose_c),
        "every duck moved": (np.linalg.norm(pose_a[:, :2] - start[:, :2], axis=1) > 0.1).all(),
        "last UDP frame matches final pose": frame_ok,
        "unknown method and unknown param refused": len(errors) == 2
        and "-32601" in errors[0] and "-32602" in errors[1],
        "odor gradient points to nearest food everywhere sampled": good == total and total > 500,
    }
    for k, v in checks.items():
        print(f"  {'ok  ' if v else 'FAIL'} {k}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
