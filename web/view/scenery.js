// viewer/godot/scenery.gd: the garden's setting, after the Chao gardens of Sonic Adventure 2. A lawn in a bowl of
// rounded rock, a waterfall of pale columns down to the pond, palms, a rail fence, sea and sky, every
// shape made by rule from lumps. None of it is in the world: it only draws the walls the ducks already have.
import * as THREE from "three";
import * as Ink from "./ink.js";
import { Rng } from "../garden/rng.js";

export const LAWN = new THREE.Color("#6fc24a");
export const ROCK = new THREE.Color("#a3978a");
export const PALE_ROCK = new THREE.Color("#ddd5c6");
export const WATER = new THREE.Color("#4f9fe0");
export const SEA = new THREE.Color("#3b7fd4");
export const FROND = new THREE.Color("#1f7a4d");
export const WOOD = new THREE.Color("#8a5a3c");
export const SKY_DAY = new THREE.Color("#a9d8f0");
export const SKY_NIGHT = new THREE.Color("#1d2a4d");
const TAU = 2 * Math.PI;

// A rounded column of rock: rings of vertices pushed in and out, narrowing upward, with a domed cap. [sides, cap].
export function lump(radius, height, rng, sides = 9, rings = 4, taper = 0.8, wobble = 0.2, dome = 0.06) {
  const lean = Array.from({ length: sides }, () => rng.uniform(-wobble, wobble));
  const ring = [];
  for (let k = 0; k <= rings; k++) {
    const u = k / rings, points = [];
    for (let j = 0; j < sides; j++) {
      const r = radius * (1 + (taper - 1) * u * u) * (1 + lean[j] + rng.uniform(-wobble, wobble) * 0.5);
      const a = TAU * j / sides;
      points.push([Math.cos(a) * r, height * u, Math.sin(a) * r]);
    }
    ring.push(points);
  }
  const side = [], cap = [];
  for (let k = 0; k < rings; k++) for (let j = 0; j < sides; j++) {
    const j2 = (j + 1) % sides, a = ring[k][j], b = ring[k][j2], c = ring[k + 1][j], d = ring[k + 1][j2];
    side.push(...a, ...c, ...b, ...b, ...c, ...d);
  }
  const top = ring[rings], crown = [0, height + dome * radius, 0];
  for (let j = 0; j < sides; j++) cap.push(...crown, ...top[(j + 1) % sides], ...top[j]);
  const geo = verts => { const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3)); return g; };
  return [geo(side), geo(cap)];
}

export function placeLump(root, at, radius, height, rng, stone = ROCK, top = LAWN, sides = 9, taper = 0.8, wobble = 0.2) {
  const [s, c] = lump(radius, height, rng, sides, 4, taper, wobble);
  Ink.part(root, s, stone, at, 1, { cell: 8, lift: 0.3 });
  Ink.part(root, c, top, at, 1, { cell: 8, lift: 0.2 });
}

export function palm(root, at, height, rng) {
  const trunk = Ink.part(root, Ink.cone(0.05, height, 0.035, 5), WOOD, [at[0], at[1] + height / 2, at[2]], 1);
  trunk.rotation.z = rng.uniform(-0.15, 0.15);
  for (let f = 0; f < 7; f++) {
    const holder = new THREE.Group();
    holder.position.set(at[0], at[1] + height, at[2]);
    holder.rotation.set(0, TAU * f / 7 + rng.uniform(-0.2, 0.2), -Math.PI / 2 - rng.uniform(0.25, 0.6), "YXZ");
    root.add(holder);
    Ink.part(holder, Ink.cone(0.0, 0.7, 0.11, 3), FROND, [0, -0.35, 0], [1, 1, 0.35]);
  }
}

// Builds the setting from the first snapshot. Returns the flags' two summits, the waterfall sheets and the reeds.
export function build(root, snap, viewer) {
  const rng = new Rng(11), size = snap.size, falls = [], reeds = [];
  let summit = null, far = null;
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  // the plateau, flat on top at the lawn, and the sea round it
  const [under, lawn] = lump(0.75 * size, 2.0, rng, 14, 3, 1.08, 0.04, 0.0);
  const mid = [size / 2, -2.002, -size / 2];
  Ink.part(root, under, ROCK, mid, 1, { cell: 8, lift: 0.22 });
  Ink.part(root, lawn, LAWN, mid, 1, { cell: 8, tone: 0.97 });
  Ink.part(root, Ink.cone(60.0, 0.01, 60.0, 24), SEA, [size / 2, -2.0, -size / 2], 1, { cell: 10, tone: 0.95 });
  for (const [x, z, r, h] of [[-7.0, 9.0, 1.6, 3.2], [13.0, 11.0, 2.2, 4.4], [16.0, -2.0, 1.4, 2.6]]) placeLump(root, [x, -2.0, -z], r, h + 2.0, rng);
  // the bowl: tall lumps behind the north and east edges, tallest in the corner
  for (let edge = 0; edge < 2; edge++) {
    for (let along = -0.6; along < size + 1.0;) {
      const r = rng.uniform(0.95, 1.5);
      const corner = Math.min(Math.max(1.0 - Math.abs(size - along) / (0.6 * size), 0), 1);
      const h = rng.uniform(1.7, 2.5) + 1.3 * corner;
      const out = r * 0.78 + 0.3 + rng.uniform(0.0, 0.25);  // 0.3 back: the rocks stand no further into the lawn than CLIFF_M
      const at = edge === 0 ? [along, -0.3, -(size + out)] : [size + out, -0.3, -along];
      placeLump(root, at, r, h, rng);
      const top = [at[0], at[1] + h, at[2]];
      if (summit === null || d(at, viewer) < d(summit, viewer)) summit = top;
      if (far === null || d(at, viewer) > d(far, viewer)) far = top;
      if (rng.random() < 0.3) palm(root, [at[0] + rng.uniform(-0.3, 0.3), at[1] + h - 0.25, at[2] + rng.uniform(-0.3, 0.3)], rng.uniform(0.5, 0.9), rng);
      along += r * 1.25;
    }
  }
  // the rail fence along the open south and west edges
  for (let edge = 0; edge < 2; edge++) {
    const posts = Math.trunc(size / 0.75);
    for (let i = 0; i <= posts; i++) {
      const t = size * i / posts, at = edge === 0 ? [t, 0.17, 0.06] : [-0.06, 0.17, -t];
      Ink.part(root, Ink.cone(0.035, 0.34, 0.03, 5), WOOD, at, 1);
    }
    for (const rail of [0.14, 0.27]) {
      Ink.part(root, Ink.box(), WOOD, edge === 0 ? [size / 2, rail, 0.06] : [-0.06, rail, -size / 2],
               edge === 0 ? [size, 0.035, 0.03] : [0.03, 0.035, size]);
    }
  }
  if (snap.pond) {
    const [px, py, pr] = snap.pond;
    Ink.part(root, Ink.cone(pr, 0.004, pr, 28), WATER, [px, 0.003, -py], 1, { cell: 8, tone: 0.9 });
    if (Math.hypot(px - size, py - size) < pr + 2.4) waterfall(root, snap, [px, py], size, rng, falls);
    for (let i = 0; i < Math.trunc(10 * pr / 0.35); i++) {  // reeds round the open shore
      const a = rng.uniform(0, TAU), at = [px + Math.cos(a) * (pr + 0.05), 0, -py - Math.sin(a) * (pr + 0.05)];
      if (at[0] > 0.1 && at[0] < size - 0.4 && -at[2] > 0.1 && -at[2] < size - 0.4) {
        const h = rng.uniform(0.2, 0.36), reed = new THREE.Group();
        reed.position.set(...at);
        root.add(reed);
        Ink.part(reed, Ink.cone(0.012, h, 0.008, 4), FROND, [0, h / 2, 0], 1);
        Ink.part(reed, Ink.cone(0.016, 0.07, 0.016, 5), WOOD, [0, h, 0], 1);
        reeds.push(reed);
      }
    }
  }
  for (let i = 0; i < Math.trunc(70 * size * size / 16); i++) {  // grass tufts
    const at = [rng.uniform(0.1, size - 0.1), 0.03, -rng.uniform(0.1, size - 0.1)];
    if (!snap.pond || Math.hypot(at[0] - snap.pond[0], -at[2] - snap.pond[1]) > snap.pond[2] + 0.1) Ink.part(root, Ink.cone(0.025, 0.07, 0.0, 4), FROND, at, 1);
  }
  const [tx, ty, tr] = snap.tree;  // the fruit tree, whose shade is in the world
  Ink.part(root, Ink.cone(tr, 0.001, tr, 18), LAWN, [tx, 0.004, -ty], 1, { cell: 8, tone: 0.55 });
  Ink.part(root, Ink.cone(0.1, 1.0, 0.07, 6), WOOD, [tx, 0.5, -ty], 1, { outlined: true });
  for (const [x, y, z, r] of [[0.0, 1.2, 0.0, 0.6], [0.32, 1.05, 0.14, 0.42], [-0.3, 1.1, -0.18, 0.45], [0.03, 1.6, 0.04, 0.38]]) {
    Ink.part(root, Ink.ball(r, 7), FROND, [tx + x, y, -ty + z], [1, 0.8, 1], { cell: 8, lift: 0.15, outlined: true });
  }
  return { summits: [summit, far], falls, reeds };
}

// A cave mouth's outline, flat on the ground with straight sides and a rounded top, facing +z.
function archGeometry(width, side, segments = 14) {
  const r = width / 2, shape = new THREE.Shape();
  shape.moveTo(-r, 0);
  shape.lineTo(r, 0);
  shape.lineTo(r, side);
  shape.absarc(0, side, r, 0, Math.PI, false);
  shape.lineTo(-r, 0);
  return new THREE.ShapeGeometry(shape, segments);
}

// The falls come out of the cliff: a wide mass of rock like the cliffs round it, standing on the highest step and
// leaning back into the corner cliffs, with a cave mouth in its face and the water running out over the lip.
function cave(root, [x, y, r, h], toward, rng) {
  const at = d => [x + toward[0] * d, -(y + toward[1] * d)];
  const big = 0.95 * r, back = 0.25 * r;
  const [cx, cz] = at(back);
  placeLump(root, [cx, h - 0.1, cz], big, 1.4, rng, ROCK, LAWN, 11, 0.75, 0.04);
  const face = back - 1.07 * big;  // just in front of the rock, so none of the mouth is hidden
  const out = Math.atan2(-toward[0], toward[1]);  // turns +z to face the pond
  const [rx, rz] = at(face);
  Ink.part(root, archGeometry(0.74, 0.3), ROCK, [rx, h - 0.02, rz], 1, { tone: 0.35 }).rotation.y = out;  // a darker rim
  const [mx, mz] = at(face - 0.01);
  Ink.part(root, archGeometry(0.6, 0.24), Ink.NAVY, [mx, h - 0.02, mz], 1, { tone: 0.02 }).rotation.y = out;
  const [wx, wz] = at((face - 0.93 * r) / 2);  // the water, from the mouth over the lip of the step
  Ink.part(root, Ink.box(), WATER, [wx, h + 0.01, wz], [Math.max(face + 0.93 * r, 0.05), 0.03, 0.5], { cell: 8, tone: 0.97 }).rotation.y = Math.atan2(toward[1], toward[0]);
}

// Pale columns stepping down from the cliff into the pond, with water sheets on them.
function waterfall(root, snap, centre, size, rng, falls) {
  const [, , pr] = snap.pond;
  const len = Math.hypot(size - centre[0], size - centre[1]);
  const toward = [(size - centre[0]) / len, (size - centre[1]) / len], side = [-toward[1], toward[0]];
  const steps = (snap.rocks || []).map(([x, y, r], i) => [x, y, r, 0.35 + 0.5 * i, true]);  // true: solid in the world
  if (!steps.length) steps.push([centre[0] + toward[0] * (pr + 0.1), centre[1] + toward[1] * (pr + 0.1), 0.55, 0.35]);
  while (steps.length < 3) {  // any higher and the corner cliffs hide the top, and its cave
    const last = steps[steps.length - 1];
    steps.push([last[0] + toward[0] * 0.6, last[1] + toward[1] * 0.6, last[2] + 0.12, last[3] + 0.65]);
  }
  const turn = Math.atan2(toward[1], toward[0]);
  steps.forEach(([x, y, r, h, solid], k) => {
    // a rock the ducks bump into is drawn nearly true to its size, or they would seem to walk into it
    placeLump(root, [x, -0.2, -y], r, h + 0.2, rng, PALE_ROCK, WATER, 10, 0.9, solid ? 0.05 : 0.2);
    const sheet = Ink.part(root, Ink.box(), WATER, [x - toward[0] * r * 0.93, h / 2, -y + toward[1] * r * 0.93], [0.05, h, 0.5], { cell: 8, tone: 0.97 });
    sheet.rotation.y = turn;
    falls.push(sheet);
    if (k === steps.length - 1) cave(root, [x, y, r, h], toward, rng);
  });
  for (const flank of [-1, 1]) {  // darker rock either side of the fall; a palm here would stand inside the cliffs
    const at = [centre[0] + toward[0] * (pr + 0.9) + side[0] * flank * 1.25, centre[1] + toward[1] * (pr + 0.9) + side[1] * flank * 1.25];
    placeLump(root, [at[0], -0.2, -at[1]], 0.75, 1.5, rng);
  }
}
