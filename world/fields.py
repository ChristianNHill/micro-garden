"""The garden for the 2D stub: diffusing smells, temperature, a pond, food dishes, contact (PLAN.md Gates 3, 4b).

Coordinates are metres, origin at a corner, x right, y up. Grid cell (i, j) covers x in [i, i+1) * CELL_M.
Two smells diffuse on their own grids: food (from dishes) and danger (from stink patches).
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
HUMID_FALLOFF_M = 0.4  # humidity halves about every 0.3 m away from the pond edge


class World:
    def __init__(self, food_xy, danger_xy=(), pond=None, bites=1):
        """pond is (x, y, radius) or None. Each dish holds `bites` bites."""
        self.food = np.asarray(food_xy, float).reshape(-1, 2)
        self.bites = np.full(len(self.food), bites)
        self.danger = np.asarray(danger_xy, float).reshape(-1, 2)
        self.pond = pond
        self.odor = np.zeros((GRID, GRID))
        self.danger_odor = np.zeros((GRID, GRID))
        self.diffuse(2000)  # start near steady state

    def diffuse(self, n: int) -> None:
        for grid, sources in ((self.odor, self.food), (self.danger_odor, self.danger)):
            if len(sources) == 0 and not grid.any():
                continue
            src = np.clip((sources / CELL_M).astype(int), 0, GRID - 1)
            for _ in range(n):
                p = np.pad(grid, 1, mode="edge")
                grid += DIFFUSION * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * grid) - DECAY * grid
                np.add.at(grid, (src[:, 0], src[:, 1]), EMIT)

    def step(self) -> None:
        self.diffuse(SUBSTEPS)

    def eat(self, dish: int) -> None:
        """One bite; an empty dish is gone and its odor fades with DECAY."""
        self.bites[dish] -= 1
        if self.bites[dish] <= 0:
            self.food = np.delete(self.food, dish, axis=0)
            self.bites = np.delete(self.bites, dish)

    def odor_at(self, xy, grid=None) -> np.ndarray:
        """Bilinear sample of a smell grid (food by default) at (..., 2) positions."""
        o = self.odor if grid is None else grid
        u = np.clip(np.asarray(xy, float) / CELL_M - 0.5, 0, GRID - 1.001)
        i, f = u.astype(int), u % 1
        i0, i1, fx, fy = i[..., 0], i[..., 1], f[..., 0], f[..., 1]
        return ((1 - fx) * (1 - fy) * o[i0, i1] + fx * (1 - fy) * o[i0 + 1, i1]
                + (1 - fx) * fy * o[i0, i1 + 1] + fx * fy * o[i0 + 1, i1 + 1])

    def odor_gradient(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        dx, dy = np.array([CELL_M, 0.0]), np.array([0.0, CELL_M])
        return np.stack([self.odor_at(xy + dx) - self.odor_at(xy - dx),
                         self.odor_at(xy + dy) - self.odor_at(xy - dy)], axis=-1) / (2 * CELL_M)

    def pond_distance(self, xy) -> np.ndarray:
        """Distance past the pond edge (negative inside); +inf without a pond."""
        xy = np.asarray(xy, float)
        if self.pond is None:
            return np.full(xy.shape[:-1], np.inf)
        return np.linalg.norm(xy - self.pond[:2], axis=-1) - self.pond[2]

    def humidity_at(self, xy) -> np.ndarray:
        """1 at and inside the pond edge, falling off outside."""
        return np.exp(-np.maximum(self.pond_distance(xy), 0) / HUMID_FALLOFF_M)


def temperature_at(xy) -> np.ndarray:
    """Sunny garden with one shade tree, soft 10 cm edge."""
    d = np.linalg.norm(np.asarray(xy, float) - TREE[:2], axis=-1)
    return SHADE_C + (SUN_C - SHADE_C) / (1 + np.exp(-(d - TREE[2]) / 0.1))


def contacts(duck_xy: np.ndarray, heading: np.ndarray, food_xy: np.ndarray):
    """Per duck: other ducks touching on its left and on its right, and the dish it stands on (-1 if none)."""
    rel = duck_xy[None] - duck_xy[:, None]  # [i, j] = j relative to i
    touching = np.linalg.norm(rel, axis=-1) < 2 * DUCK_R
    np.fill_diagonal(touching, False)
    left_side = rel[..., 1] * np.cos(heading)[:, None] - rel[..., 0] * np.sin(heading)[:, None] > 0
    touch_left = (touching & left_side).sum(axis=1)
    touch_right = (touching & ~left_side).sum(axis=1)
    if len(food_xy) == 0:
        return touch_left, touch_right, np.full(len(duck_xy), -1)
    f = np.linalg.norm(duck_xy[:, None] - food_xy[None], axis=-1)
    dish = np.where(f.min(axis=1) < DUCK_R + DISH_R, f.argmin(axis=1), -1)
    return touch_left, touch_right, dish
