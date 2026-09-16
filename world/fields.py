"""The garden for the 2D stub: diffusing odor, static temperature, food dishes, contact (PLAN.md Gate 3).

Coordinates are metres, origin at a corner, x right, y up. Grid cell (i, j) covers x in [i, i+1) * CELL_M.
"""
import numpy as np

SIZE_M = 4.0
GRID = 64
CELL_M = SIZE_M / GRID
DIFFUSION = 0.2  # per substep, stable below 0.25
DECAY = 0.002  # per substep; decay length sqrt(DIFFUSION / DECAY) = 10 cells, about 0.6 m
SUBSTEPS = 5  # per 20 ms body step
EMIT = 1.0
DISH_R = 0.08
DUCK_R = 0.07
SUN_C, SHADE_C = 30.0, 20.0
TREE = (1.0, 3.0, 0.7)  # shade centre x, y and radius


class World:
    def __init__(self, food_xy):
        self.food = np.asarray(food_xy, float).reshape(-1, 2)
        self.odor = np.zeros((GRID, GRID))
        self.diffuse(2000)  # start near steady state

    def diffuse(self, n: int) -> None:
        src = np.clip((self.food / CELL_M).astype(int), 0, GRID - 1)
        o = self.odor
        for _ in range(n):
            p = np.pad(o, 1, mode="edge")
            o += DIFFUSION * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * o) - DECAY * o
            np.add.at(o, (src[:, 0], src[:, 1]), EMIT)

    def step(self) -> None:
        self.diffuse(SUBSTEPS)

    def odor_at(self, xy) -> np.ndarray:
        """Bilinear sample at (..., 2) positions."""
        u = np.clip(np.asarray(xy, float) / CELL_M - 0.5, 0, GRID - 1.001)
        i, f = u.astype(int), u % 1
        i0, i1, fx, fy = i[..., 0], i[..., 1], f[..., 0], f[..., 1]
        o = self.odor
        return ((1 - fx) * (1 - fy) * o[i0, i1] + fx * (1 - fy) * o[i0 + 1, i1]
                + (1 - fx) * fy * o[i0, i1 + 1] + fx * fy * o[i0 + 1, i1 + 1])

    def odor_gradient(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        dx, dy = np.array([CELL_M, 0.0]), np.array([0.0, CELL_M])
        return np.stack([self.odor_at(xy + dx) - self.odor_at(xy - dx),
                         self.odor_at(xy + dy) - self.odor_at(xy - dy)], axis=-1) / (2 * CELL_M)


def temperature_at(xy) -> np.ndarray:
    """Sunny garden with one shade tree, soft 10 cm edge."""
    d = np.linalg.norm(np.asarray(xy, float) - TREE[:2], axis=-1)
    return SHADE_C + (SUN_C - SHADE_C) / (1 + np.exp(-(d - TREE[2]) / 0.1))


def contacts(duck_xy: np.ndarray, food_xy: np.ndarray):
    """Per duck: number of other ducks touching, and index of the dish it stands on (-1 if none)."""
    d = np.linalg.norm(duck_xy[:, None] - duck_xy[None], axis=-1)
    touch = (d < 2 * DUCK_R).sum(axis=1) - 1
    if len(food_xy) == 0:
        return touch, np.full(len(duck_xy), -1)
    f = np.linalg.norm(duck_xy[:, None] - food_xy[None], axis=-1)
    dish = np.where(f.min(axis=1) < DUCK_R + DISH_R, f.argmin(axis=1), -1)
    return touch, dish
