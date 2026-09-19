"""The garden rendered onto flyvis's 721-column hex lattice, one eye per side (PLAN.md Gate 6).

Each world object is a vertical cylinder, so a column sees it when the angle between the column's
gaze direction and the object's bearing is under the object's angular radius atan(r / d). Approach
grows that radius: looming comes out of the geometry, not a flag.

Intensities follow flyvis's convention: 0 dark, 0.5 grey background, 1 bright.
"""
import numpy as np
from flyvis.utils.hex_utils import get_hex_coords, hex_to_pixel

from world.fields import DISH_R, DUCK_R, TREE

EXTENT = 15  # flyvis's lattice: 721 columns
N_HEX = 721
OMMATIDIUM_DEG = 5.8  # interommatidial angle, Drosophila
EYE_AZ_DEG = 55.0  # each eye's optical axis, degrees off forward; the two overlap frontally to +-20 deg
BACKGROUND, DISH_I, DUCK_I, TREE_I = 0.5, 1.0, 0.15, 0.0
HAND_I, HAND_R = 0.9, 0.12  # the player's hand, a pale thing about the size of two dishes
POND_I = 0.85  # water reflecting the sky; bright, but not as bright as food
EYE_H = 0.10  # metres off the ground. ponytail: a guess at microduck eye height; measure it at Gate 10
HORIZON_DEG = 1.5  # below this the ground point is past the garden anyway, and one column covers acres of it

HEX_U, HEX_V = get_hex_coords(EXTENT)
_x, _y = hex_to_pixel(HEX_U, HEX_V)
_scale = np.radians(OMMATIDIUM_DEG) / np.sqrt(3)  # hex neighbours sit sqrt(3) apart in pixel units
HEX_AZ = _x * _scale  # rightward in the eye's own field, radians
HEX_EL = _y * _scale  # upward
EYE_AZ = np.radians([EYE_AZ_DEG, -EYE_AZ_DEG])  # left eye looks left (headings are counter-clockwise)


def scene(world, duck_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Visible things as (x, y, radius, intensity) rows, and which duck each row is (-1 if not a duck).

    Smells are invisible and the pond is flat on the ground; dishes, ducks and the tree are not.
    """
    rows = [(x, y, DISH_R, DISH_I) for x, y in world.food]
    rows += [(x, y, DUCK_R, DUCK_I) for x, y in duck_xy]
    rows.append((TREE[0], TREE[1], TREE[2], TREE_I))
    if getattr(world, "hand", None) is not None:
        rows.append((world.hand[0], world.hand[1], HAND_R, HAND_I))
    owner = np.concatenate([np.full(len(world.food), -1), np.arange(len(duck_xy)),
                            np.full(len(rows) - len(world.food) - len(duck_xy), -1)])
    return np.array(rows, float).reshape(-1, 4), owner


def luminance(duck_xy: np.ndarray, heading: np.ndarray, world, light: float = 1.0) -> np.ndarray:
    """(n ducks, 2 eyes, 721) intensities, left eye first. A duck does not see itself.

    `light` dims the whole scene toward the background at night, so contrast goes with the sun.
    """
    rows, owner = scene(world, duck_xy)
    n = len(duck_xy)
    lum = np.full((n, 2, N_HEX), BACKGROUND, np.float32)
    nearest = np.full((n, 2, N_HEX), np.inf)
    # where each column looks, as a world bearing: eye axis minus the column's rightward offset
    col_az = heading[:, None, None] + EYE_AZ[None, :, None] - HEX_AZ[None, None, :]
    for (ox, oy, r, intensity), who in zip(rows, owner):
        dx, dy = ox - duck_xy[:, 0], oy - duck_xy[:, 1]
        d = np.hypot(dx, dy)[:, None, None]
        if who >= 0:
            d[who] = np.inf  # a duck does not see itself
        off = col_az - np.arctan2(dy, dx)[:, None, None]
        off = np.abs(np.arctan2(np.sin(off), np.cos(off)))
        hit = (np.hypot(off, HEX_EL) < np.arctan2(r, np.maximum(d, r))) & (d < nearest)
        lum[hit] = intensity
        nearest = np.where(hit, d, nearest)

    if getattr(world, "pond", None) is not None:
        # Water lies flat, so it has no silhouette to loom: a column below the horizon meets the ground
        # at EYE_H / tan(depression), and the pond is whatever of those points fall inside it. That is
        # a ground-plane raycast, which also gives the near shore its proper perspective for free.
        px, py, pr = world.pond
        ground = np.full(N_HEX, 1e6)  # sky and the horizon itself never reach the ground
        below = HEX_EL < -np.radians(HORIZON_DEG)
        ground[below] = EYE_H / np.tan(-HEX_EL[below])
        gx = duck_xy[:, 0, None, None] + ground * np.cos(col_az)
        gy = duck_xy[:, 1, None, None] + ground * np.sin(col_az)
        wet = (np.hypot(gx - px, gy - py) < pr) & (ground < nearest)
        lum[wet] = POND_I
    return BACKGROUND + (lum - BACKGROUND) * light
