"""The garden for the 2D stub: diffusing smells, temperature, a pond, food dishes, contact (PLAN.md Gates 3, 4b).

Coordinates are metres, origin at a corner, x right, y up. Grid cell (i, j) covers x in [i, i+1) * CELL_M.
Two smells diffuse on their own grids: food (from dishes) and danger (from stink patches).
"""
import numpy as np

SIZE_M = 4.0  # the garden every gate is measured in; a World can be another size (the demo garden is)
GRID = 64
CELL_M = SIZE_M / GRID  # the same cell whatever the size, so smells spread alike in any garden
DIFFUSION = 0.2  # per substep, stable below 0.25
# Per substep; the decay length is sqrt(DIFFUSION / DECAY) cells. At 0.002 it was 0.62 m in a 4 m
# garden, so a duck starting where the demo garden puts them, 1.5 to 2.5 m from the only dish, could
# not smell it at all. A longer smell is a shallower one, so DANGER_HALF keeps a faint stink from
# reading as alarm (audit, 2026-09-19).
DECAY = 0.0008  # about 1 m
SUBSTEPS = 5  # per 20 ms body step
EMIT = 1.0
DISH_R = 0.08
DUCK_R = 0.07
SUN_C, SHADE_C = 30.0, 20.0
DAY_S = 600.0  # a whole day and night in simulated seconds; short enough that a gate can watch one
NIGHT_C = 8.0  # how much colder the garden gets when the sun is down
DAWN = 0.15  # fraction of the cycle that dawn and dusk take; the rest is flat day or flat night
TREE = (1.0, 3.0, 0.7)  # shade centre x, y and radius, unless a World puts its tree elsewhere
WIND_FULL_MS = 1.5  # light air; this reads as 1.0 on the antennae
MUSIC_M = 1.2  # music is half as loud every 0.8 m or so; a garden-wide thing, unlike a duck's smell
DUCK_SMELL_M = 0.5  # another duck smells half as strong every 0.35 m or so
HUMID_FALLOFF_M = 0.4  # humidity halves about every 0.3 m away from the pond edge
SHORE_M = 0.1  # a duck whose centre is within this of the pond edge can drink
# Damp air off the pond, carried on the wind like a smell. Per pond cell per substep, set so that two
# metres downwind the air reads about 0.1, which is the damp sense's half level (brain/server.py
# HUMID_HALF), as food two metres downwind reads a little under its own; in still air the pond is only
# its own edge.
POND_DAMP = 0.004
WALL_CLEAR_M = 0.6  # fruit does not land nearer a wall than this: a duck cannot search around what is against one
# A fruit is a meal: ten bites is what takes a starving duck to full (brain/physiology.py BITE_FULLNESS).
# At three a find never filled anyone, so nobody was ever done foraging: ducks had both needs low for 11%
# of their waking time and spent 40 to 55% of it following a plume, and five personalities that mostly act
# through what a duck does with its free time looked alike (Gate 5; Chris, 2026-09-20: personality should
# show as much as possible).
FRUIT_BITES = 10
BALL_R = 0.06  # a ball a duck can push with its chest or kick
BALL_ROLLS_S = 1.2  # how long a rolling ball takes to lose most of its speed on grass; a third of that in water
BALL_BOUNCE = 0.6  # of its speed kept off the fence or a rock
MAX_FOOD = 4  # the tree stops dropping while this much food is on the ground


class World:
    def __init__(self, food_xy, danger_xy=(), pond=None, bites=1, wind=None, wind_turns_s=None, music=None,
                 size=SIZE_M, tree=TREE, rocks=()):
        """pond is (x, y, radius) or None. Each dish holds `bites` bites. wind is the (x, y) velocity
        the air moves at, in m/s, or None for still air: it carries the smells and the pond's damp air
        downwind, so a plume reaches a long way on one side of its source and hardly at all on the
        other. wind_turns_s is how long the breeze takes to swing right round the compass, or None
        for a steady one: in a steady wind whatever lies downwind of the ducks can never be found. music is
        (x, y), something that plays where it has been put, a part of the garden like the pond and the
        stink patch (Chris, 2026-09-20); the player can pick it up and put it down somewhere else."""
        self.size, self.tree = float(size), tuple(tree)  # metres along a side, and the tree's (x, y, shade radius)
        self.balls = np.zeros((0, 4))  # x, y, vx, vy each: toys, which roll (PLAN.md Gate 8b)
        self.rocks = np.asarray(rocks, float).reshape(-1, 3)  # (x, y, radius) each: round, solid, and in the way
        self.grid = round(self.size / CELL_M)
        self.wind0 = self.wind = None if wind is None else np.asarray(wind, float)
        self.wind_turns_s = wind_turns_s
        self.damp = np.zeros((self.grid, self.grid))
        self.food = np.asarray(food_xy, float).reshape(-1, 2)
        self.bites = np.full(len(self.food), bites)
        self.danger = np.asarray(danger_xy, float).reshape(-1, 2)
        self.pond = pond
        self.hand = None  # (x, y) while the player's hand is in the garden (PLAN.md Gate 8)
        self.music = None if music is None else (float(music[0]), float(music[1]))
        self.music_volume = 0.75  # 0 to 1: how loud the box plays, to the ducks as to the player
        self.odor = np.zeros((self.grid, self.grid))
        self.danger_odor = np.zeros((self.grid, self.grid))
        self.diffuse(2000)  # start near steady state

    def _pond_cells(self):
        c = (np.indices((self.grid, self.grid)).reshape(2, -1).T + 0.5) * CELL_M
        return np.argwhere((self.pond_distance(c) < 0).reshape(self.grid, self.grid))

    def diffuse(self, n: int) -> None:
        fields = [(self.odor, self.food, EMIT), (self.danger_odor, self.danger, EMIT)]
        if self.wind is not None and self.pond is not None:
            fields.append((self.damp, None, POND_DAMP))
        for grid, sources, emit in fields:
            if sources is not None and len(sources) == 0 and not grid.any():
                continue
            src = self._pond_cells() if sources is None else np.clip((sources / CELL_M).astype(int), 0, self.grid - 1)
            for _ in range(n):
                p = np.pad(grid, 1, mode="edge")
                grid += DIFFUSION * (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * grid) - DECAY * grid
                if self.wind is not None:
                    # first-order upwind: each cell takes from the neighbour the air comes from. At the
                    # downwind wall nothing comes back, so the smell blows out of the garden.
                    # The air that blows in is clean: padded with the wall's own value, as diffusion is,
                    # the upwind wall kept its smell and five ducks followed it there and stayed.
                    cx, cy = self.wind * (0.02 / SUBSTEPS) / CELL_M  # cells per substep, far below 1
                    q = np.pad(grid, 1)
                    grid -= abs(cx) * (grid - (q[:-2, 1:-1] if cx > 0 else q[2:, 1:-1]))
                    grid -= abs(cy) * (grid - (q[1:-1, :-2] if cy > 0 else q[1:-1, 2:]))
                np.add.at(grid, (src[:, 0], src[:, 1]), emit)

    def step(self, t: float = 0.0) -> None:
        if self.wind0 is not None and self.wind_turns_s:
            a = 2 * np.pi * t / self.wind_turns_s
            self.wind = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ self.wind0
        self.diffuse(SUBSTEPS)

    def eat(self, dish: int) -> None:
        """One bite; an empty dish is gone and its odor fades with DECAY."""
        self.bites[dish] -= 1
        if self.bites[dish] <= 0:
            self.food = np.delete(self.food, dish, axis=0)
            self.bites = np.delete(self.bites, dish)

    def drop_fruit(self, rng: np.random.Generator, n: int = 1) -> int:
        """Fruit falls somewhere under the shade tree's canopy. Returns how many fell."""
        fell = 0
        while fell < n and len(self.food) < MAX_FOOD:
            a, r = rng.uniform(-np.pi, np.pi), rng.uniform(0.2, self.tree[2] + 0.2)
            xy = np.clip(np.array(self.tree[:2]) + r * np.array([np.cos(a), np.sin(a)]), WALL_CLEAR_M, self.size - WALL_CLEAR_M)
            self.food = np.vstack([self.food, xy])
            self.bites = np.append(self.bites, FRUIT_BITES)
            fell += 1
        return fell

    def roll_balls(self, dt: float, duck_xy: np.ndarray, duck_v: np.ndarray) -> None:
        """One step of every ball: it rolls and slows, comes back off the fence and the rocks, and a duck
        that walks into it pushes it ahead at the duck's own pace and a little over."""
        for ball in self.balls:
            pos, v = ball[:2], ball[2:]
            wet = self.pond is not None and self.pond_distance(pos) < 0
            v *= np.exp(-dt / (BALL_ROLLS_S / 3 if wet else BALL_ROLLS_S))
            pos += v * dt
            for axis in range(2):
                if not BALL_R <= pos[axis] <= self.size - BALL_R:
                    pos[axis] = np.clip(pos[axis], BALL_R, self.size - BALL_R)
                    v[axis] *= -BALL_BOUNCE
            for things, reach, bounce in ((self.rocks, None, True), (duck_xy, DUCK_R, False)):
                for k, thing in enumerate(things):
                    r = (thing[2] if reach is None else reach) + BALL_R
                    away = pos - thing[:2]
                    d = np.linalg.norm(away)
                    if d < r:
                        n = away / max(d, 1e-9)
                        pos[:] = thing[:2] + n * r
                        into = v @ n
                        if bounce:
                            v -= (1 + BALL_BOUNCE) * min(into, 0.0) * n
                        else:
                            v += max(duck_v[k] @ n + 0.1 - into, 0.0) * n

    def push_out(self, xy: np.ndarray, clearance: float) -> None:
        """Move any of these (n, 2) points that are inside a rock back to its edge, in place: a rock is a
        wall that happens to be round."""
        for x, y, r in self.rocks:
            away = xy - (x, y)
            d = np.linalg.norm(away, axis=1)
            inside = d < r + clearance
            xy[inside] = (x, y) + away[inside] / np.maximum(d[inside], 1e-9)[:, None] * (r + clearance)

    def odor_at(self, xy, grid=None) -> np.ndarray:
        """Bilinear sample of a smell grid (food by default) at (..., 2) positions."""
        o = self.odor if grid is None else grid
        u = np.clip(np.asarray(xy, float) / CELL_M - 0.5, 0, self.grid - 1.001)
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
        """1 at and inside the pond edge, falling off outside, plus whatever damp air the wind carries."""
        near = np.exp(-np.maximum(self.pond_distance(xy), 0) / HUMID_FALLOFF_M)
        return near if self.wind is None else np.clip(near + self.odor_at(xy, self.damp), 0, 1)


def wind_on(heading, wind) -> tuple[np.ndarray, np.ndarray]:
    """What each duck's antennae feel: (strength 0-1, where it comes from in radians off the nose,
    positive to the left). Wind is where the air goes, so it comes from the opposite way."""
    heading = np.asarray(heading, float)
    if wind is None:
        return np.zeros_like(heading), np.zeros_like(heading)
    source = np.arctan2(-wind[1], -wind[0]) - heading
    return (np.full_like(heading, min(np.hypot(*wind) / WIND_FULL_MS, 1.0)),
            (source + np.pi) % (2 * np.pi) - np.pi)


def music_at(sensor_xy, source) -> np.ndarray:
    """How loud the music is at a point, 1 at the speaker and falling off with distance (Gate 8b)."""
    if source is None:
        return np.zeros(len(np.atleast_2d(sensor_xy)))
    d = np.linalg.norm(np.atleast_2d(np.asarray(sensor_xy, float)) - np.asarray(source, float), axis=-1)
    return np.exp(-d / MUSIC_M)


def duck_odor_at(sensor_xy, duck_xy, exclude: int) -> np.ndarray:
    """How strongly one duck's antenna smells the others (PLAN.md Gate 8).

    Worked out per pair rather than diffused on a grid: ducks move every step, so a grid would have to
    rewrite its sources constantly, and a shared one would have each duck smelling its own emission
    loudest of all. A distance kernel excludes the smeller for nothing and is exact.
    """
    d = np.linalg.norm(np.asarray(duck_xy, float) - np.asarray(sensor_xy, float), axis=-1)
    smell = np.exp(-d / DUCK_SMELL_M)
    smell[exclude] = 0.0
    return smell.sum()


def daylight(t: float) -> float:
    """How light the garden is, 0 at night and 1 in the day, with a dawn and a dusk (PLAN.md Gate 3).

    A flat-topped cycle rather than a sine: a garden should spend most of its day being day, not
    forever on its way to noon.
    """
    phase = (t % DAY_S) / DAY_S
    if phase < 0.5 - DAWN / 2:
        return 1.0
    if phase < 0.5 + DAWN / 2:
        return float(np.clip((0.5 + DAWN / 2 - phase) / DAWN, 0, 1))  # dusk
    if phase < 1.0 - DAWN:
        return 0.0
    return float(np.clip((phase - (1.0 - DAWN)) / DAWN, 0, 1))  # dawn


def temperature_at(xy, light: float = 1.0, tree=TREE) -> np.ndarray:
    """Sunny garden with one shade tree, soft 10 cm edge. The whole garden cools once the sun is down."""
    d = np.linalg.norm(np.asarray(xy, float) - tree[:2], axis=-1)
    warm = SHADE_C + (SUN_C - SHADE_C) / (1 + np.exp(-(d - tree[2]) / 0.1))
    return warm - NIGHT_C * (1.0 - light)


def contacts(duck_xy: np.ndarray, heading: np.ndarray, food_xy: np.ndarray, touch_m: float = 2 * DUCK_R):
    """Per duck: other ducks touching on its left and on its right, and the dish it stands on (-1 if none).
    touch_m is how near two ducks' centres are when they touch: two circles' worth here, arm's length for a
    body that falls over if it is actually bumped."""
    rel = duck_xy[None] - duck_xy[:, None]  # [i, j] = j relative to i
    touching = np.linalg.norm(rel, axis=-1) < touch_m
    np.fill_diagonal(touching, False)
    left_side = rel[..., 1] * np.cos(heading)[:, None] - rel[..., 0] * np.sin(heading)[:, None] > 0
    touch_left = (touching & left_side).sum(axis=1)
    touch_right = (touching & ~left_side).sum(axis=1)
    if len(food_xy) == 0:
        return touch_left, touch_right, np.full(len(duck_xy), -1)
    f = np.linalg.norm(duck_xy[:, None] - food_xy[None], axis=-1)
    dish = np.where(f.min(axis=1) < DUCK_R + DISH_R, f.argmin(axis=1), -1)
    return touch_left, touch_right, dish
