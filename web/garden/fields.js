// world/fields.py in JavaScript: the garden the ducks live in, with diffusing smells, wind, a pond, fruit and toys.
// Coordinates are metres, origin at a corner, x right, y up. Grid cell (i, j) covers x in [i, i+1) * CELL_M and is
// held at i * grid + j. The arithmetic runs in the same order as the Python so the smells come out the same.

import { Rng } from "./rng.js";

export const SIZE_M = 4.0;
export const GRID = 64;
export const CELL_M = SIZE_M / GRID;
export const DIFFUSION = 0.2;
export const DECAY = 0.0008;
export const SUBSTEPS = 5;
export const DT = 0.02;
export const WIND_HOLDS_S = [90.0, 360.0];
export const WIND_WANDER = 0.7;
export const WIND_GUST = [0.3, 1.3];
export const WIND_EASES_S = 40.0;
export const EMIT = 1.0;
export const DISH_R = 0.08;
export const DUCK_R = 0.07;
export const SUN_C = 30.0, SHADE_C = 20.0;
export const DAY_S = 600.0;
export const NIGHT_C = 8.0;
export const DAWN = 0.15;
export const TREE = [1.0, 3.0, 0.7];
export const TRUNK_R = 0.12;  // the fruit tree's trunk, solid like a rock
export const CAN_R = 0.12;  // the garbage can at each stink patch, solid like a rock
export const CLIFF_M = 0.3;  // how far a drawn garden's cliffs stand into the lawn along the north and east edges
export const WIND_FULL_MS = 1.5;
export const MUSIC_M = 1.2;
export const DUCK_SMELL_M = 0.5;
export const HUMID_FALLOFF_M = 0.4;
export const SHORE_M = 0.1;
export const POND_DAMP = 0.004;
export const WALL_CLEAR_M = 0.6;
export const FRUIT_BITES = 10;
export const BALL_R = 0.06;
export const BALL_ROLLS_S = 1.2;
export const BALL_BOUNCE = 0.6;
export const FRUITS = 10;
export const MAX_FOOD = 4;

const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);

export class World {
  constructor({ food_xy, danger_xy = [], pond = null, bites = 1, wind = null, wind_turns_s = null, music = null,
                size = SIZE_M, tree = TREE, rocks = [] }) {
    this.size = size;
    this.tree = [...tree];
    this.drum = null;
    this.instruments = [];  // [x, y, kind]
    this.balls = [];  // [x, y, vx, vy]
    this.rocks = rocks.map(r => [...r]);
    this.grid = Math.round(this.size / CELL_M);
    this.wind0 = wind === null ? null : [...wind];
    this.wind = this.wind0 === null ? null : [...this.wind0];
    this.wind_turns_s = wind_turns_s;
    this.wind_rng = new Rng(7);
    this.wind_next = 0.0;
    this.wind_to = [0.0, 1.0];
    this.wind_at = [0.0, 1.0];
    const G = this.grid;
    this.damp = new Float64Array(G * G);
    this.kind_rng = new Rng(food_xy.length + 17);
    this.set_food(food_xy, bites);
    this.danger = danger_xy.map(d => [...d]);
    this.pond = pond === null ? null : [...pond];
    this.hand = null;
    this.music = music === null ? null : [music[0], music[1]];
    this.music_volume = 0.75;
    this.odor = new Float64Array(G * G);
    this.danger_odor = new Float64Array(G * G);
    this.diffuse(2000);
  }

  _pond_cells() {
    const G = this.grid, out = [];
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      if (this.pond_distance([(i + 0.5) * CELL_M, (j + 0.5) * CELL_M]) < 0) out.push([i, j]);
    }
    return out;
  }

  diffuse(n) {
    const G = this.grid;
    const fields = [[this.odor, this.food, EMIT], [this.danger_odor, this.danger, EMIT]];
    if (this.wind !== null && this.pond !== null) fields.push([this.damp, null, POND_DAMP]);
    const lap = new Float64Array(G * G), q = new Float64Array(G * G);
    for (const [grid, sources, emit] of fields) {
      if (sources !== null && sources.length === 0 && !grid.some(v => v !== 0)) continue;
      const src = sources === null ? this._pond_cells()
        : sources.map(([x, y]) => [clip(Math.trunc(x / CELL_M), 0, G - 1), clip(Math.trunc(y / CELL_M), 0, G - 1)]);
      for (let step = 0; step < n; step++) {
        // edge-padded laplacian, all from the old grid
        for (let i = 0; i < G; i++) {
          const im = i > 0 ? i - 1 : 0, ip = i < G - 1 ? i + 1 : G - 1;
          for (let j = 0; j < G; j++) {
            const jm = j > 0 ? j - 1 : 0, jp = j < G - 1 ? j + 1 : G - 1;
            lap[i * G + j] = grid[im * G + j] + grid[ip * G + j] + grid[i * G + jm] + grid[i * G + jp] - 4 * grid[i * G + j];
          }
        }
        for (let k = 0; k < G * G; k++) grid[k] += DIFFUSION * lap[k] - DECAY * grid[k];
        if (this.wind !== null) {
          // first-order upwind with clean air blowing in; x first, then y against the grid before x moved it
          const cx = this.wind[0] * (0.02 / SUBSTEPS) / CELL_M, cy = this.wind[1] * (0.02 / SUBSTEPS) / CELL_M;
          q.set(grid);
          const ax = Math.abs(cx), ay = Math.abs(cy);
          for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
            const up = cx > 0 ? (i > 0 ? q[(i - 1) * G + j] : 0) : (i < G - 1 ? q[(i + 1) * G + j] : 0);
            grid[i * G + j] -= ax * (grid[i * G + j] - up);
          }
          for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
            const up = cy > 0 ? (j > 0 ? q[i * G + j - 1] : 0) : (j < G - 1 ? q[i * G + j + 1] : 0);
            grid[i * G + j] -= ay * (grid[i * G + j] - up);
          }
        }
        for (const [i, j] of src) grid[i * G + j] += emit;
      }
    }
  }

  step(t = 0.0) {
    if (this.wind0 !== null && this.wind_turns_s) {
      if (t >= this.wind_next) {
        this.wind_next = t + this.wind_rng.uniform(...WIND_HOLDS_S);
        this.wind_to = [this.wind_rng.uniform(-WIND_WANDER, WIND_WANDER), this.wind_rng.uniform(...WIND_GUST)];
      }
      const ease = Math.min(DT / WIND_EASES_S, 1.0);
      this.wind_at = [this.wind_at[0] + (this.wind_to[0] - this.wind_at[0]) * ease,
                      this.wind_at[1] + (this.wind_to[1] - this.wind_at[1]) * ease];
      const a = 2 * Math.PI * t / this.wind_turns_s + this.wind_at[0];
      const [x, y] = this.wind0, c = Math.cos(a), s = Math.sin(a);
      this.wind = [(c * x - s * y) * this.wind_at[1], (s * x + c * y) * this.wind_at[1]];
    }
    this.diffuse(SUBSTEPS);
  }

  set_food(food_xy, bites, kinds = null) {
    this.food = food_xy.map(f => [f[0], f[1]]);
    this.bites = Array.isArray(bites) ? [...bites] : this.food.map(() => bites);
    this.kinds = kinds === null ? this.food.map(() => this.kind_rng.integers(FRUITS)) : [...kinds];
  }

  eat(dish) {
    this.bites[dish] -= 1;
    if (this.bites[dish] <= 0) {
      this.food.splice(dish, 1);
      this.bites.splice(dish, 1);
      this.kinds.splice(dish, 1);
    }
  }

  add_food(xy, bites) {
    this.food.push([xy[0], xy[1]]);
    this.bites.push(Math.trunc(bites));
    this.kinds.push(this.kind_rng.integers(FRUITS));
  }

  drop_fruit(rng, n = 1, most = MAX_FOOD) {
    let fell = 0;
    while (fell < n && this.food.length < most) {
      const a = rng.uniform(-Math.PI, Math.PI), r = rng.uniform(0.2, this.tree[2] + 0.2);
      const xy = [clip(this.tree[0] + r * Math.cos(a), WALL_CLEAR_M, this.size - WALL_CLEAR_M),
                  clip(this.tree[1] + r * Math.sin(a), WALL_CLEAR_M, this.size - WALL_CLEAR_M)];
      this.add_food(xy, FRUIT_BITES);
      fell += 1;
    }
    return fell;
  }

  roll_balls(dt, duck_xy, duck_v) {
    for (const ball of this.balls) {
      const wet = this.pond !== null && this.pond_distance(ball) < 0;
      const slow = Math.exp(-dt / (wet ? BALL_ROLLS_S / 3 : BALL_ROLLS_S));
      ball[2] *= slow; ball[3] *= slow;
      ball[0] += ball[2] * dt; ball[1] += ball[3] * dt;
      for (let axis = 0; axis < 2; axis++) {
        if (!(BALL_R <= ball[axis] && ball[axis] <= this.size - BALL_R)) {
          ball[axis] = clip(ball[axis], BALL_R, this.size - BALL_R);
          ball[2 + axis] *= -BALL_BOUNCE;
        }
      }
      for (const [things, reach, bounce] of [[this.solids(), null, true], [duck_xy, DUCK_R, false]]) {
        things.forEach((thing, k) => {
          const r = (reach === null ? thing[2] : reach) + BALL_R;
          const ax = ball[0] - thing[0], ay = ball[1] - thing[1];
          const d = Math.hypot(ax, ay);
          if (d < r) {
            const nx = ax / Math.max(d, 1e-9), ny = ay / Math.max(d, 1e-9);
            ball[0] = thing[0] + nx * r; ball[1] = thing[1] + ny * r;
            const into = ball[2] * nx + ball[3] * ny;
            if (bounce) {
              const push = (1 + BALL_BOUNCE) * Math.min(into, 0.0);
              ball[2] -= push * nx; ball[3] -= push * ny;
            } else {
              const push = Math.max(duck_v[k][0] * nx + duck_v[k][1] * ny + 0.1 - into, 0.0);
              ball[2] += push * nx; ball[3] += push * ny;
            }
          }
        });
      }
    }
  }

  // Everything round that a duck or a ball cannot pass: the rocks and the tree's trunk.
  solids() { return [...this.rocks, [this.tree[0], this.tree[1], TRUNK_R], ...this.danger.map(([x, y]) => [x, y, CAN_R])]; }

  push_out(xy, clearance) {  // xy: [[x, y], ...], moved in place
    for (const [x, y, r] of this.solids()) {
      for (const p of xy) {
        const ax = p[0] - x, ay = p[1] - y, d = Math.hypot(ax, ay);
        if (d < r + clearance) {
          const s = (r + clearance) / Math.max(d, 1e-9);
          p[0] = x + ax * s; p[1] = y + ay * s;
        }
      }
    }
  }

  odor_at(xy, grid = null) {  // one [x, y]
    const o = grid === null ? this.odor : grid, G = this.grid;
    const ux = clip(xy[0] / CELL_M - 0.5, 0, G - 1.001), uy = clip(xy[1] / CELL_M - 0.5, 0, G - 1.001);
    const i0 = Math.trunc(ux), i1 = Math.trunc(uy), fx = ux % 1, fy = uy % 1;
    return ((1 - fx) * (1 - fy) * o[i0 * G + i1] + fx * (1 - fy) * o[(i0 + 1) * G + i1]
            + (1 - fx) * fy * o[i0 * G + i1 + 1] + fx * fy * o[(i0 + 1) * G + i1 + 1]);
  }

  pond_distance(xy) {
    if (this.pond === null) return Infinity;
    return Math.hypot(xy[0] - this.pond[0], xy[1] - this.pond[1]) - this.pond[2];
  }

  humidity_at(xy) {
    const near = Math.exp(-Math.max(this.pond_distance(xy), 0) / HUMID_FALLOFF_M);
    return this.wind === null ? near : clip(near + this.odor_at(xy, this.damp), 0, 1);
  }
}

// What each antenna feels of the wind: [strength 0-1, where it comes from off the nose, positive left].
export function wind_on(heading, wind) {
  if (wind === null) return [0, 0];
  const source = Math.atan2(-wind[1], -wind[0]) - heading;
  return [Math.min(Math.hypot(wind[0], wind[1]) / WIND_FULL_MS, 1.0), pymod(source + Math.PI, 2 * Math.PI) - Math.PI];
}

export const pymod = (a, b) => ((a % b) + b) % b;

export function music_at(sensor_xy, source) {
  if (source === null) return 0;
  return Math.exp(-Math.hypot(sensor_xy[0] - source[0], sensor_xy[1] - source[1]) / MUSIC_M);
}

export function duck_odor_at(sensor_xy, duck_xy, exclude) {
  let s = 0;
  duck_xy.forEach((d, j) => { if (j !== exclude) s += Math.exp(-Math.hypot(d[0] - sensor_xy[0], d[1] - sensor_xy[1]) / DUCK_SMELL_M); });
  return s;
}

export function daylight(t) {
  const phase = pymod(t, DAY_S) / DAY_S;
  if (phase < 0.5 - DAWN / 2) return 1.0;
  if (phase < 0.5 + DAWN / 2) return clip((0.5 + DAWN / 2 - phase) / DAWN, 0, 1);
  if (phase < 1.0 - DAWN) return 0.0;
  return clip((phase - (1.0 - DAWN)) / DAWN, 0, 1);
}

export function temperature_at(xy, light = 1.0, tree = TREE) {
  const d = Math.hypot(xy[0] - tree[0], xy[1] - tree[1]);
  const warm = SHADE_C + (SUN_C - SHADE_C) / (1 + Math.exp(-(d - tree[2]) / 0.1));
  return warm - NIGHT_C * (1.0 - light);
}

// Per duck: other ducks touching on its left and on its right, and the dish it stands on (-1 if none).
export function contacts(duck_xy, heading, food_xy, touch_m = 2 * DUCK_R) {
  const n = duck_xy.length;
  const touch_left = new Array(n).fill(0), touch_right = new Array(n).fill(0), dish = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const rx = duck_xy[j][0] - duck_xy[i][0], ry = duck_xy[j][1] - duck_xy[i][1];
      if (Math.hypot(rx, ry) < touch_m) {
        if (ry * Math.cos(heading[i]) - rx * Math.sin(heading[i]) > 0) touch_left[i]++; else touch_right[i]++;
      }
    }
    let best = -1, bd = Infinity;
    food_xy.forEach((f, k) => { const d = Math.hypot(duck_xy[i][0] - f[0], duck_xy[i][1] - f[1]); if (d < bd) { bd = d; best = k; } });
    if (best >= 0 && bd < DUCK_R + DISH_R) dish[i] = best;
  }
  return [touch_left, touch_right, dish];
}
