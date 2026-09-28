// viewer/godot/hats.gd: a hat from its style number. The same number always makes the same hat, on the lawn or on a
// head: fifteen shapes, in a colour and a trim of their own that the number picks at random.
import * as THREE from "three";
import * as Ink from "./ink.js";
import { Rng } from "../garden/rng.js";

export const SHAPES = 15;
const TAU = 2 * Math.PI;

export function hsv(h, s, v) {
  const f = n => { const k = (n + h * 6) % 6; return v - v * s * Math.max(0, Math.min(k, 4 - k, 1)); };
  return new THREE.Color().setRGB(f(5), f(3), f(1), THREE.SRGBColorSpace);
}

export function make(style) {
  const hat = new THREE.Group(), dice = new Rng(style);
  const mainH = dice.random();
  const main = hsv(mainH, dice.uniform(0.45, 0.85), dice.uniform(0.7, 0.95));
  const band = hsv((mainH + dice.uniform(0.25, 0.75)) % 1, dice.uniform(0.35, 0.8), dice.uniform(0.75, 0.98));
  const part = (geo, colour, at, size = 1, lined = false) => Ink.part(hat, geo, colour, at, size, { cell: 5, lift: 0.3, outlined: lined });
  const { ball, cone, box } = Ink;
  switch (style % SHAPES) {
    case 0: part(cone(0.045, 0.1, 0, 8), main, [0, 0.05, 0], 1, true); part(ball(0.014, 6), band, [0, 0.105, 0]); break;
    case 1:
      part(cone(0.07, 0.008, 0.07, 10), main, [0, 0.004, 0], 1, true);
      part(cone(0.042, 0.085, 0.045, 10), main, [0, 0.05, 0], 1, true);
      part(cone(0.044, 0.016, 0.044, 10), band, [0, 0.02, 0]); break;
    case 2:
      part(ball(0.05, 8), main, [0, 0.012, 0], [1, 0.85, 1], true);
      part(cone(0.052, 0.018, 0.052, 10), band, [0, 0.008, 0]);
      part(ball(0.017, 6), band, [0, 0.062, 0]); break;
    case 3:
      part(cone(0.045, 0.04, 0.045, 10), main, [0, 0.02, 0], 1, true);
      for (let k = 0; k < 5; k++) {
        const a = TAU * k / 5;
        part(cone(0.013, 0.035, 0, 4), main, [Math.cos(a) * 0.036, 0.055, Math.sin(a) * 0.036]);
        part(ball(0.008, 5), band, [Math.cos(a) * 0.036, 0.076, Math.sin(a) * 0.036]);
      }
      break;
    case 4:
      part(cone(0.095, 0.008, 0.1, 12), main, [0, 0.004, 0], 1, true);
      part(ball(0.042, 8), main, [0, 0.01, 0], [1, 0.8, 1], true);
      part(cone(0.044, 0.014, 0.043, 10), band, [0, 0.014, 0]); break;
    case 5:
      part(cone(0.1, 0.008, 0.1, 12), main, [0, 0.006, 0], [1, 1, 0.75], true);
      for (const side of [-1, 1]) part(cone(0.03, 0.006, 0.03, 8), main, [side * 0.085, 0.02, 0], [1, 1, 2.2]).rotation.z = side * 0.9;
      part(cone(0.04, 0.06, 0.046, 10), main, [0, 0.038, 0], [1, 1, 0.85], true);
      part(cone(0.047, 0.012, 0.047, 10), band, [0, 0.016, 0], [1, 1, 0.86]); break;
    case 6:
      part(ball(0.045, 10), main, [0, 0.02, 0], [1, 0.95, 1], true);
      part(cone(0.062, 0.01, 0.062, 12), main, [0, 0.006, 0]);
      part(cone(0.047, 0.012, 0.047, 10), band, [0, 0.015, 0]); break;
    case 7:
      part(ball(0.06, 10), main, [0.01, 0.016, 0], [1, 0.35, 1], true).rotation.z = -0.25;
      part(cone(0.006, 0.02, 0.004, 5), band, [0.005, 0.042, 0]); break;
    case 8:
      part(cone(0.045, 0.035, 0.047, 10), Ink.CREAM, [0, 0.018, 0], 1, true);
      part(ball(0.058, 10), Ink.CREAM, [0, 0.07, 0], [1, 0.8, 1], true);
      part(cone(0.047, 0.01, 0.047, 10), main, [0, 0.008, 0]); break;
    case 9:
      part(ball(0.047, 10), main, [0, 0.01, 0], [1, 0.8, 1], true);
      part(box(), band, [0.055, 0.008, 0], [0.06, 0.006, 0.07]);
      part(ball(0.008, 5), band, [0, 0.048, 0]); break;
    case 10:
      part(cone(0.09, 0.008, 0.09, 12), main, [0, 0.004, 0], 1, true);
      part(cone(0.042, 0.075, 0.022, 8), main, [0, 0.042, 0], 1, true);
      part(cone(0.022, 0.05, 0, 6), main, [-0.012, 0.098, 0], 1, true).rotation.z = 0.5;
      part(cone(0.044, 0.014, 0.043, 10), band, [0, 0.014, 0]); break;
    case 11:
      part(ball(0.048, 10), main, [0, 0.01, 0], [1, 0.8, 1], true);
      part(cone(0.004, 0.03, 0.004, 5), Ink.NAVY, [0, 0.058, 0]);
      for (const side of [-1, 1]) part(box(), band, [side * 0.028, 0.074, 0], [0.05, 0.004, 0.014]).rotation.x = side * 0.3;
      break;
    case 12:
      part(cone(0.047, 0.01, 0.047, 12), new THREE.Color("#2fa36b"), [0, 0.008, 0]);
      for (let k = 0; k < 7; k++) {
        const a = TAU * k / 7;
        part(ball(0.014, 6), k % 2 === 0 ? main : band, [Math.cos(a) * 0.047, 0.016, Math.sin(a) * 0.047]);
        part(ball(0.005, 4), Ink.MUSTARD, [Math.cos(a) * 0.057, 0.02, Math.sin(a) * 0.057]);
      }
      break;
    case 13:
      part(ball(0.05, 10), Ink.STONE, [0, 0.008, 0], [1, 0.85, 1], true);
      part(cone(0.052, 0.012, 0.052, 10), main, [0, 0.006, 0]);
      for (const side of [-1, 1]) part(cone(0.012, 0.06, 0, 6), Ink.CREAM, [0, 0.04, side * 0.048], 1, true).rotation.x = -side * 0.9;
      break;
    case 14:
      part(ball(0.012, 6), band, [0, 0.02, 0]);
      for (const side of [-1, 1]) part(cone(0.028, 0.045, 0.004, 6), main, [0, 0.02, side * 0.025], [1, 1, 0.5], true).rotation.x = -side * Math.PI / 2;
      break;
  }
  return hat;
}
