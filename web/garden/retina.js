// body/stub2d/retina.py: the garden rendered onto flyvis's 721-column hex lattice, one eye per side. Each thing is a
// vertical cylinder a column sees when the angle to it is under its angular radius, so approach grows it and a loom
// comes out of the geometry. 0 is dark, 0.5 the grey background, 1 bright.
import { BALL_R, DISH_R, DUCK_R } from "./fields.js";

export const EXTENT = 15;
export const N_HEX = 721;
export const OMMATIDIUM_DEG = 5.8;
export const EYE_AZ_DEG = 55.0;
export const BACKGROUND = 0.5, DISH_I = 1.0, DUCK_I = 0.15, TREE_I = 0.0;
export const ROCK_I = 0.5;
export const BALL_I = 0.95;
export const HAND_I = 0.9, HAND_R = 0.12;
export const POND_I = 0.85;
export const EYE_H = 0.10;
export const HORIZON_DEG = 1.5;

// flyvis's get_hex_coords and hex_to_pixel, so column k here is column k of its network
export const HEX_U = [], HEX_V = [];
for (let q = -EXTENT; q <= EXTENT; q++) {
  for (let r = Math.max(-EXTENT, -EXTENT - q); r <= Math.min(EXTENT, EXTENT - q); r++) { HEX_U.push(q); HEX_V.push(r); }
}
const RAD = Math.PI / 180;  // numpy.radians multiplies by this
const SCALE = (OMMATIDIUM_DEG * RAD) / Math.sqrt(3);
export const HEX_AZ = Float64Array.from(HEX_V, (v, k) => 3 / 2 * v * SCALE);
export const HEX_EL = Float64Array.from(HEX_U, (u, k) => -Math.sqrt(3) * (u + HEX_V[k] / 2) * SCALE);
const EYE_AZ = [EYE_AZ_DEG * RAD, -EYE_AZ_DEG * RAD];
const f32 = Math.fround;

// Visible things as [x, y, radius, intensity] rows, and which duck each row is (-1 if not a duck).
export function scene(world, duck_xy) {
  const rows = [], owner = [];
  for (const [x, y] of world.food) { rows.push([x, y, DISH_R, DISH_I]); owner.push(-1); }
  duck_xy.forEach(([x, y], i) => { rows.push([x, y, DUCK_R, DUCK_I]); owner.push(i); });
  rows.push([...world.tree, TREE_I]); owner.push(-1);
  for (const [x, y, r] of world.rocks) { rows.push([x, y, r, ROCK_I]); owner.push(-1); }
  for (const [x, y] of world.balls) { rows.push([x, y, BALL_R, BALL_I]); owner.push(-1); }
  if (world.hand !== null) { rows.push([world.hand[0], world.hand[1], HAND_R, HAND_I]); owner.push(-1); }
  return [rows, owner];
}

// Per duck, a Float32Array of 2 * 721 intensities, left eye first. A duck does not see itself. `light` dims the
// scene toward the background at night.
export function luminance(duck_xy, heading, world, light = 1.0) {
  const [rows, owner] = scene(world, duck_xy);
  const out = [];
  const below = HEX_EL.map(el => el < -HORIZON_DEG * RAD);
  for (let i = 0; i < duck_xy.length; i++) {
    const lum = new Float32Array(2 * N_HEX).fill(BACKGROUND);
    const nearest = new Float64Array(2 * N_HEX).fill(Infinity);
    for (let row = 0; row < rows.length; row++) {
      const [ox, oy, r, intensity] = rows[row];
      const dx = ox - duck_xy[i][0], dy = oy - duck_xy[i][1];
      const d = owner[row] === i ? Infinity : Math.hypot(dx, dy);
      const bearing = Math.atan2(dy, dx), reach = Math.atan2(r, Math.max(d, r));
      for (let eye = 0; eye < 2; eye++) {
        for (let c = 0; c < N_HEX; c++) {
          let off = heading[i] + EYE_AZ[eye] - HEX_AZ[c] - bearing;
          off = Math.abs(Math.atan2(Math.sin(off), Math.cos(off)));
          const k = eye * N_HEX + c;
          if (Math.hypot(off, HEX_EL[c]) < reach && d < nearest[k]) { lum[k] = intensity; nearest[k] = d; }
        }
      }
    }
    if (world.pond !== null) {
      const [px, py, pr] = world.pond;
      for (let eye = 0; eye < 2; eye++) {
        for (let c = 0; c < N_HEX; c++) {
          const ground = below[c] ? EYE_H / Math.tan(-HEX_EL[c]) : 1e6;
          const az = heading[i] + EYE_AZ[eye] - HEX_AZ[c];
          const gx = duck_xy[i][0] + ground * Math.cos(az), gy = duck_xy[i][1] + ground * Math.sin(az);
          const k = eye * N_HEX + c;
          if (Math.hypot(gx - px, gy - py) < pr && ground < nearest[k]) lum[k] = POND_I;
        }
      }
    }
    const lit = f32(light);
    for (let k = 0; k < lum.length; k++) lum[k] = f32(BACKGROUND + f32(f32(lum[k] - BACKGROUND) * lit));
    out.push(lum);
  }
  return out;
}
