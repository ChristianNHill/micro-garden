"""Build the Godot garden's duck from the real microduck: robot.json, a few thousand triangles of it.

Run with the simulator's Python, which has MuJoCo:
    ~/.cache/micro-garden/spike/microduck_rl/.venv/bin/python viewer/godot/build_robot.py

The robot's meshes are Pollen Robotics' (microduck_rl, Apache 2.0; ATTRIBUTION.md): 23 MB of STL, a million
triangles, mostly bearings and circuit boards nobody sees. This poses the robot standing from its MuJoCo
description, keeps the visible parts, snaps their vertices to a CELL_M grid (the simplification and the
low-poly look), and writes each body in its own frame, on its own hinge, under its parent, with each part
given the nearest ink of the garden's palette, so a viewer can pose it from joint angles. Godot builds the
meshes at start-up, so nothing needs importing.
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
UNIT_M = 0.0005  # coordinates are written as whole numbers of this, to keep the file small
HIDDEN = re.compile(r"bearing|pcb|np_f970|speaker|banana|rigidity|motor_support|power_support|lens_holder")
STAND = {"left_hip_roll": -0.0873, "right_hip_roll": 0.0873, "left_hip_pitch": -0.4579, "right_hip_pitch": 0.4579,
         "left_knee": -0.0049, "right_knee": 0.0049, "left_ankle": 0.4530, "right_ankle": -0.4530,
         "neck_pitch": 0.3491, "head_pitch": 0.3491}  # microduck_constants.HOME_FRAME
INKS = {"cream": (0.96, 0.92, 0.84), "stone": (0.66, 0.64, 0.6), "navy": (0.11, 0.16, 0.30),
        "coral": (0.93, 0.44, 0.36), "mustard": (0.9, 0.71, 0.29)}
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
    d.qpos[:3] = 0
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    soup = {}  # (body, ink) -> vertices in that body's own frame, faces
    lowest = np.inf
    for g in range(m.ngeom):
        mesh = m.mesh(m.geom_dataid[g]) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None
        if mesh is None or m.geom_group[g] != 2 or HIDDEN.search(mesh.name):
            continue
        v = m.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
        f = m.mesh_face[mesh.faceadr[0]:mesh.faceadr[0] + mesh.facenum[0]]
        world = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        lowest = min(lowest, world[:, 2].min())
        body = m.geom_bodyid[g]
        local = (world - d.xpos[body]) @ d.xmat[body].reshape(3, 3)
        rgb = m.mat_rgba[m.geom_matid[g]][:3]
        ink = min(INKS, key=lambda name: np.sum((np.array(INKS[name]) - rgb) ** 2))
        vs, fs, n = soup.setdefault((body, ink), ([], [], [0]))
        vs.append(local)
        fs.append(f + n[0])
        n[0] += len(local)
    bodies, triangles = [], 0
    for body in range(1, m.nbody):
        joints = [j for j in range(m.njnt) if m.jnt_bodyid[j] == body and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        assert len(joints) <= 1 and not any(m.jnt_pos[j].any() for j in joints), "one hinge a body, at its origin: how Godot builds it"
        parts = []
        for (owner, ink), (vs, fs, _) in sorted(soup.items()):
            if owner == body:
                verts, faces = snapped(np.concatenate(vs), np.concatenate(fs))
                triangles += len(faces)
                parts.append({"ink": ink, "v": np.round(verts / UNIT_M).astype(int).ravel().tolist(),
                              "i": faces[:, ::-1].ravel().tolist()})  # Godot's front faces wind clockwise, STL's counter-clockwise
        bodies.append({"name": m.body(body).name, "parent": m.body(m.body_parentid[body]).name,
                       "pos": m.body_pos[body].round(5).tolist(), "quat": m.body_quat[body].round(6).tolist(),
                       "joint": None if not joints else {"name": m.joint(joints[0]).name, "axis": m.jnt_axis[joints[0]].round(6).tolist()},
                       "parts": parts})
    # the hat seat: over the middle of the head shell, with up and forward, in the head's frame
    head = m.body("jaw_soft").id
    crown_verts = np.concatenate([np.concatenate(vs) for (owner, ink), (vs, _, _) in soup.items() if owner == head and ink == "cream"])
    to_world = d.xmat[head].reshape(3, 3)
    shell = crown_verts @ to_world.T
    seat = np.array([(shell[:, 0].min() + shell[:, 0].max()) / 2, (shell[:, 1].min() + shell[:, 1].max()) / 2,
                     shell[:, 2].max() + 0.012])  # floating a little
    top = seat @ to_world  # back into the head's frame
    hat = {"body": "jaw_soft", "at": top.round(4).tolist(), "up": to_world.T[:, 2].round(5).tolist(),
           "forward": to_world.T[:, 0].round(5).tolist()}
    # All in MuJoCo's frame (z up); Godot turns the whole rig once. `stand` is the default pose and
    # `stand_z` the trunk height in it.
    OUT.write_text(json.dumps({"unit": UNIT_M, "bodies": bodies, "stand": STAND, "stand_z": round(float(-lowest), 4), "hat": hat},
                              separators=(",", ":")))
    print(f"{OUT.name}: {triangles} triangles on {len(bodies)} bodies, {sum(b['joint'] is not None for b in bodies)} hinges, "
          f"{OUT.stat().st_size / 1024:.0f} KB, trunk {-lowest:.3f} m up when standing")
    assert 1000 < triangles < 20000 and 0.05 < -lowest < 0.3 and sum(b["joint"] is not None for b in bodies) == 14
    return 0


if __name__ == "__main__":
    sys.exit(main())
