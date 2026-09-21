"""Build the Godot garden's duck from the real microduck: robot.json, a few thousand triangles of it.

Run with the simulator's Python, which has MuJoCo (PLAN.md Gate 10 for where the checkouts live):
    ~/.cache/micro-garden/spike/microduck_rl/.venv/bin/python viewer/godot/build_robot.py

The robot's meshes are Pollen Robotics' (microduck_rl, Apache 2.0; ATTRIBUTION.md): 23 MB of STL at a
million triangles, most of them bearings and circuit boards nobody sees. This poses the robot standing, from
its own MuJoCo description, keeps the parts that show, snaps their vertices to a CELL_M grid (which is both
the simplification and the low-poly look), and writes four rigid groups (trunk, head, left and right leg),
each about the joint it swings on, with every part given the nearest ink of the garden's palette. Godot
builds its meshes from that at start-up, so nothing needs importing and the file is the whole asset.
"""
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

RL = Path.home() / ".cache/micro-garden/spike/microduck_rl"
XML = RL / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"
OUT = Path(__file__).parent / "robot.json"
CELL_M = 0.006
UNIT_M = 0.0005  # coordinates are written as whole numbers of this, which keeps the file small
HIDDEN = re.compile(r"bearing|pcb|np_f970|speaker|banana|rigidity|motor_support|power_support|lens_holder")
STAND = {"left_hip_roll": -0.0873, "right_hip_roll": 0.0873, "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
         "left_knee": -0.0049, "right_knee": 0.0049, "left_ankle": 0.4530, "right_ankle": -0.4530,
         "neck_pitch": 0.3491, "head_pitch": 0.3491}  # microduck_constants.HOME_FRAME
INKS = {"cream": (0.96, 0.92, 0.84), "stone": (0.66, 0.64, 0.6), "navy": (0.11, 0.16, 0.30),
        "coral": (0.93, 0.44, 0.36), "mustard": (0.9, 0.71, 0.29)}
# which group a body's parts move with, and the body whose origin that group turns about
GROUPS = {"head": ("neck", ["neck", "neck_pitch", "yaw_roll_motion", "jaw_soft"]),
          "leg_left": ("upper_leg_left", ["upper_leg_left", "leg", "ankle_left"]),
          "leg_right": ("upper_leg_right", ["upper_leg_right", "leg_2", "ankle_right"])}


def to_godot(p: np.ndarray) -> np.ndarray:
    """MuJoCo is z up with y to the left; Godot is y up with z to the right. Forward stays +x."""
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def snapped(verts: np.ndarray, faces: np.ndarray):
    """Vertex clustering: every vertex goes to its grid cell's mean, and triangles that collapse go."""
    cells, inverse = np.unique(np.floor(verts / CELL_M).astype(int), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    merged = np.zeros((len(cells), 3))
    np.add.at(merged, inverse, verts)
    merged /= np.bincount(inverse)[:, None]
    f = inverse[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    f = f[np.unique(np.sort(f, axis=1), axis=0, return_index=True)[1]]
    used, remap = np.unique(f, return_inverse=True)
    return merged[used], remap.reshape(-1, 3)


def main() -> int:
    m = mujoco.MjModel.from_xml_path(str(XML))
    d = mujoco.MjData(m)
    for joint, angle in STAND.items():
        d.qpos[m.joint(joint).qposadr[0]] = angle
    mujoco.mj_forward(m, d)
    group_of = {body: g for g, (_, bodies) in GROUPS.items() for body in bodies}
    soup = {}  # (group, ink) -> [verts], [faces]
    for g in range(m.ngeom):
        mesh = m.mesh(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
        if mesh is None or m.geom_group[g] != 2 or HIDDEN.search(mesh.name):
            continue
        v = m.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
        f = m.mesh_face[mesh.faceadr[0]:mesh.faceadr[0] + mesh.facenum[0]]
        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        rgb = m.mat_rgba[m.geom_matid[g]][:3]
        ink = min(INKS, key=lambda name: np.sum((np.array(INKS[name]) - rgb) ** 2))
        vs, fs, n = soup.setdefault((group_of.get(m.body(m.geom_bodyid[g]).name, "trunk"), ink), ([], [], [0]))
        vs.append(world)
        fs.append(f + n[0])
        n[0] += len(world)
    floor = min(np.concatenate(vs)[:, 2].min() for vs, _, _ in soup.values())  # the soles stand on y = 0
    out, triangles = {}, 0
    for (group, ink), (vs, fs, _) in sorted(soup.items()):
        pivot = d.xpos[m.body(GROUPS[group][0]).id] if group in GROUPS else np.zeros(3)
        pivot = pivot - [0, 0, floor]
        verts, faces = snapped(np.concatenate(vs) - [0, 0, floor] - pivot, np.concatenate(fs))
        triangles += len(faces)
        entry = out.setdefault(group, {"pivot": np.round(to_godot(pivot[None])[0], 4).tolist(), "parts": []})
        entry["parts"].append({"ink": ink, "v": np.round(to_godot(verts) / UNIT_M).astype(int).ravel().tolist(),
                               "i": faces[:, ::-1].ravel().tolist()})  # Godot's front faces wind clockwise, an STL's the other way
    OUT.write_text(json.dumps({"unit": UNIT_M, "groups": out}, separators=(",", ":")))
    height = max((np.array(p["v"]).reshape(-1, 3)[:, 1].max() * UNIT_M + g["pivot"][1]) for g in out.values() for p in g["parts"])
    print(f"{OUT.name}: {triangles} triangles in {sum(len(g['parts']) for g in out.values())} parts, "
          f"{OUT.stat().st_size / 1024:.0f} KB, standing {height:.3f} m tall")
    assert set(out) == {"trunk", "head", "leg_left", "leg_right"} and 1000 < triangles < 20000 and 0.1 < height < 0.4
    return 0


if __name__ == "__main__":
    sys.exit(main())
