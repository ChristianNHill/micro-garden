// The things in the garden, as viewer/godot/main.gd builds them: ten fruits, ten balls, the drum, ten instruments, the
// stink patch, the music box, the wind flags and the player's glove. Every one is made from the same few shapes.
import * as THREE from "three";
import * as Ink from "./ink.js";
import { WOOD } from "./scenery.js";

const TAU = 2 * Math.PI;
const C = hex => new THREE.Color(hex);
const lined = (n, geo, colour, at, size = 1, cell = 6) => Ink.part(n, geo, colour, at, size, { cell, lift: 0.3, outlined: true });
const plain = (n, geo, colour, at, size = 1, cell = 6) => Ink.part(n, geo, colour, at, size, { cell });

// Ten fruits (world/fields.py FRUITS), all the same food; each duck has a favourite.
export function fruit(n, [x, y, kind]) {
  const leaf = C("#2f8f4a"), skin = (geo, colour, at, size = 1) => lined(n, geo, colour, at, size);
  const stalk = (at, lean = 0) => { plain(n, Ink.cone(0.005, 0.04, 0.005, 4), C("#6b4326"), at).rotation.z = lean; };
  const { ball, cone } = Ink;
  switch (kind) {
    case 3: skin(ball(0.07, 8), C("#b8c94a"), [0, 0.066, 0]); skin(ball(0.045, 8), C("#b8c94a"), [0, 0.13, 0]); stalk([0, 0.18, 0], 0.2); break;
    case 4: for (const side of [-1, 1]) { skin(ball(0.038, 8), C("#b3142c"), [side * 0.04, 0.038, 0]); stalk([side * 0.022, 0.09, 0], side * 0.45); } break;
    case 5:
      for (const at of [[-0.04, 0.1, 0], [0, 0.1, 0.02], [0.04, 0.1, 0], [-0.02, 0.066, 0.01], [0.02, 0.066, -0.01], [0, 0.033, 0], [0, 0.1, -0.03]]) skin(ball(0.026, 6), C("#6b3f8f"), at);
      stalk([0, 0.14, 0]); break;
    case 6:
      skin(cone(0.0, 0.1, 0.055, 8), C("#e0293a"), [0, 0.05, 0]);
      for (let k = 0; k < 5; k++) { const a = TAU * k / 5; plain(n, ball(0.018, 4), leaf, [Math.cos(a) * 0.03, 0.102, Math.sin(a) * 0.03], [1.6, 0.4, 1.0]); }
      break;
    case 7:
      skin(ball(0.06, 8), C("#f5dc3a"), [0, 0.06, 0], [1.35, 0.9, 0.9]);
      for (const side of [-1, 1]) skin(ball(0.014, 5), C("#f5dc3a"), [side * 0.083, 0.06, 0]);
      break;
    case 8: skin(ball(0.062, 8), C("#5a2a6e"), [0, 0.062, 0], [1.0, 1.05, 0.9]); stalk([0, 0.13, 0]); break;
    case 9: skin(ball(0.07, 8), C("#f6a26b"), [0, 0.068, 0]); plain(n, ball(0.022, 5), leaf, [0.025, 0.14, 0], [1.6, 0.4, 0.9]); break;
    case 0: skin(ball(0.075, 8), C("#f39a2b"), [0, 0.072, 0]); plain(n, ball(0.018, 5), leaf, [0, 0.148, 0], [1.6, 0.5, 1.0]); break;
    case 1:
      skin(ball(0.075, 8), C("#d8342c"), [0, 0.068, 0], [1.0, 0.9, 1.0]);
      plain(n, cone(0.006, 0.045, 0.006, 4), C("#6b4326"), [0, 0.15, 0]);
      plain(n, ball(0.022, 5), leaf, [0.03, 0.155, 0], [1.6, 0.4, 0.9]); break;
    default:
      for (let k = 0; k < 5; k++) {  // a banana: a curve of short pieces, fat in the middle
        const u = (k - 2) / 2, r = 0.03 - 0.008 * Math.abs(u);
        skin(cone(r, 0.062, r, 6), C("#f4d23c"), [u * 0.075, 0.035 + 0.03 * u * u, 0]).rotation.z = Math.PI / 2 - u * 0.55;
      }
  }
  n.rotation.y = x * 7.0 + y * 3.0;
}

// Ten kinds of ball (body/stub2d/stub.py BALL_STYLES), all the same size. The markings roll with the ball.
export function ballProp(n, [, , style = 0]) {
  const r = 0.06 * 1.9, s = style % 10;
  const base = [C("#f6f1e6"), C("#f6f1e6"), C("#f6f1e6"), C("#e0782c"), C("#c8d84a"), Ink.NAVY, Ink.CORAL, Ink.TEAL, Ink.MUSTARD, Ink.STONE][s];
  const ball = Ink.part(n, Ink.ball(r, 10), base, [0, r, 0], 1, { cell: 5, lift: 0.3, outlined: true });
  const mark = (colour, size, turn = [0, 0, 0], at = [0, 0, 0]) => { Ink.part(ball, Ink.ball(r * 1.01, 10), colour, at, size, { cell: 5, lift: 0.3 }).rotation.set(...turn); };
  const spots = (colour, count, size) => {
    for (let k = 0; k < count; k++) {
      const y = 1 - 2 * (k + 0.5) / count, ring = Math.sqrt(1 - y * y);
      Ink.part(ball, Ink.ball(r * size, 6), colour, [Math.cos(k * 2.4) * ring * r * 0.93, y * r * 0.93, Math.sin(k * 2.4) * ring * r * 0.93], 1, { cell: 5 });
    }
  };
  switch (s) {
    case 0: mark(C("#e2483d"), [1.0, 0.34, 1.0]); break;
    case 1: [Ink.CORAL, Ink.MUSTARD, Ink.TEAL].forEach((c, k) => mark(c, [0.5, 1.0, 1.0], [0, TAU * k / 6, 0])); break;
    case 2: spots(Ink.NAVY, 12, 0.3); break;
    case 3: mark(Ink.NAVY, [1.0, 0.06, 1.0]); mark(Ink.NAVY, [0.06, 1.0, 1.0]); break;
    case 4: mark(C("#f6f1e6"), [1.0, 0.08, 1.0], [0.6, 0, 0.4]); break;
    case 5: Ink.part(ball, Ink.ball(r * 0.45, 8), C("#f6f1e6"), [0, 0, r * 0.8], [1, 1, 0.5], { cell: 5 }); break;
    case 7: for (const y of [-0.45, 0.45]) mark(C("#f6f1e6"), [0.9, 0.16, 0.9], [0, 0, 0], [0, y * r, 0]); break;
    case 8: spots(Ink.CORAL, 16, 0.24); break;
    case 9: spots(C("#8f8a80"), 9, 0.32); break;
  }
  n.userData.ball = ball;
}

export function drum(n) {
  Ink.part(n, Ink.cone(0.13, 0.16, 0.11, 10), Ink.CORAL, [0, 0.11, 0], 1, { cell: 6, lift: 0.3, outlined: true });
  Ink.part(n, Ink.cone(0.135, 0.02, 0.135, 10), C("#f6f1e6"), [0, 0.2, 0], 1, { cell: 6, lift: 0.4, outlined: true });
  for (let k = 0; k < 3; k++) plain(n, Ink.cone(0.012, 0.06, 0.012, 4), WOOD, [Math.cos(TAU * k / 3) * 0.09, 0.03, Math.sin(TAU * k / 3) * 0.09]);
}

export const STINK_PUFFS = 9;
// The garbage can at a stink patch: things put in it are gone, and the stink still rises off it.
export const CAN_TOP = 0.3;
export function stink(n) {
  const tin = C("#8fa3a8"), lid = C("#6c7f86");
  Ink.part(n, Ink.cone(0.22, 0.002, 0.22, 10), Ink.MUSTARD, [0, 0.003, 0], 1, { cell: 5, tone: 0.3 });
  Ink.part(n, Ink.cone(0.1, 0.26, 0.12, 12), tin, [0, 0.13, 0], 1, { cell: 6, lift: 0.3, outlined: true });
  for (const y of [0.07, 0.19]) Ink.part(n, Ink.cone(0.117, 0.012, 0.117, 12), lid, [0, y, 0], [1.02, 1, 1.02], { cell: 6 });
  Ink.part(n, Ink.cone(0.108, 0.004, 0.108, 12), Ink.NAVY, [0, 0.259, 0], 1, { tone: 0.02 });  // the dark inside, seen with the lid up
  const swivel = new THREE.Group();  // turned each frame so the hinge is on the far side and the lid opens to the viewer
  n.add(swivel);
  n.userData.swivel = swivel;
  const hinge = new THREE.Group();  // the lid swings up on a hinge at the back of the rim
  hinge.position.set(-0.13, 0.275, 0);
  swivel.add(hinge);
  Ink.part(hinge, Ink.cone(0.13, 0.03, 0.13, 12), lid, [0.13, 0, 0], 1, { cell: 6, lift: 0.3, outlined: true });
  Ink.part(hinge, Ink.box(), lid, [0.13, 0.025, 0], [0.07, 0.02, 0.02], { cell: 6, outlined: true });
  n.userData.lid = hinge;
  n.userData.puffs = Array.from({ length: STINK_PUFFS }, () => Ink.part(n, Ink.ball(0.035, 7), Ink.MUSTARD, [0, 0, 0], 1, { cell: 4, tone: 0.7 }));
}

export function musicBox(n) {
  Ink.part(n, Ink.box(), Ink.CORAL, [0, 0.05, 0], [0.14, 0.1, 0.14], { cell: 7, outlined: true });
  Ink.part(n, Ink.cone(0.02, 0.16, 0.1, 8), Ink.MUSTARD, [0.03, 0.18, 0], 1, { cell: 7, outlined: true }).rotation.z = -0.5;
}

// Ten instruments (body/stub2d/stub.py INSTRUMENTS).
export function instrument(n, [x, y, kind]) {
  const brass = C("#e6b54a"), cream = C("#f6f1e6");
  const part = (geo, colour, at, size = 1, outlined = false) => Ink.part(n, geo, colour, at, size, { cell: 6, lift: 0.3, outlined });
  const { ball, cone, box } = Ink;
  n.userData.kind = kind;
  switch (kind) {
    case 0:
      for (const side of [-1, 1]) part(box(), WOOD, [0, 0.03, side * 0.07], [0.34, 0.03, 0.02]);
      [Ink.CORAL, Ink.MUSTARD, C("#7ac74f"), Ink.TEAL, C("#3a7bd5"), C("#7d4fc2")].forEach((c, k) => part(box(), c, [-0.13 + 0.052 * k, 0.05, 0], [0.04, 0.015, 0.2 - 0.018 * k], true));
      break;
    case 1:
      for (const side of [-1, 1]) {
        part(ball(0.045, 8), side > 0 ? Ink.MUSTARD : Ink.CORAL, [side * 0.05, 0.045, -0.04], [1, 1, 1.2], true);
        part(cone(0.01, 0.12, 0.012, 5), WOOD, [side * 0.02, 0.02, 0.05]).rotation.x = Math.PI / 2 - side * 0.3;
      }
      break;
    case 2:
      part(cone(0.12, 0.035, 0.12, 14), cream, [0, 0.02, 0], 1, true);
      for (let k = 0; k < 6; k++) { const a = TAU * k / 6; part(cone(0.018, 0.008, 0.018, 8), brass, [Math.cos(a) * 0.12, 0.03, Math.sin(a) * 0.12]); }
      break;
    case 3:
      for (let k = 0; k < 3; k++) {
        const a0 = Math.PI / 2 + TAU * k / 3, a1 = Math.PI / 2 + TAU * (k + 1) / 3;
        const from = [Math.cos(a0) * 0.1, Math.sin(a0) * 0.1], to = [Math.cos(a1) * 0.1, Math.sin(a1) * 0.1];
        const side = part(box(), brass, [(from[0] + to[0]) / 2, 0.12 + (from[1] + to[1]) / 2, 0], [Math.hypot(to[0] - from[0], to[1] - from[1]), 0.02, 0.02], true);
        side.rotation.z = Math.atan2(to[1] - from[1], to[0] - from[0]);
      }
      part(cone(0.04, 0.02, 0.05, 8), WOOD, [0, 0.01, 0]);
      part(cone(0.004, 0.11, 0.004, 4), Ink.STONE, [0.1, 0.006, 0.04]).rotation.z = Math.PI / 2;
      break;
    case 4:
      part(box(), Ink.CORAL, [0, 0.06, 0], [0.3, 0.12, 0.16], true);
      part(box(), cream, [0, 0.1, 0.07], [0.26, 0.02, 0.05]);
      for (let k = 0; k < 5; k++) part(box(), Ink.NAVY, [-0.1 + 0.05 * k, 0.115, 0.06], [0.02, 0.012, 0.03]);
      break;
    case 5:
      part(ball(0.08, 10), C("#c46b2a"), [0, 0.03, 0], [1, 0.4, 1], true);
      part(ball(0.06, 10), C("#c46b2a"), [0.1, 0.03, 0], [1, 0.4, 1], true);
      part(ball(0.025, 8), Ink.NAVY, [0.02, 0.058, 0], [1, 0.2, 1]);
      part(box(), WOOD, [0.25, 0.04, 0], [0.22, 0.02, 0.035]);
      part(box(), WOOD, [0.38, 0.04, 0], [0.05, 0.025, 0.055]);
      break;
    case 6:
      part(cone(0.015, 0.08, 0.06, 10), brass, [0.14, 0.06, 0], 1, true).rotation.z = -Math.PI / 2;
      part(cone(0.013, 0.2, 0.013, 6), brass, [0, 0.06, 0], 1, true).rotation.z = Math.PI / 2;
      for (let k = 0; k < 3; k++) part(cone(0.01, 0.04, 0.01, 5), brass, [-0.02 + 0.025 * k, 0.09, 0]);
      break;
    case 7:
      part(cone(0.07, 0.1, 0.03, 10), brass, [0, 0.05, 0], 1, true);
      part(cone(0.012, 0.07, 0.012, 5), WOOD, [0, 0.13, 0]);
      part(ball(0.018, 6), Ink.NAVY, [0, 0.01, 0]);
      break;
    case 8:
      part(cone(0.018, 0.3, 0.014, 8), cream, [0, 0.02, 0], 1, true).rotation.z = Math.PI / 2;
      for (let k = 0; k < 5; k++) part(ball(0.005, 4), Ink.NAVY, [-0.08 + 0.035 * k, 0.037, 0]);
      break;
    case 9:
      part(box(), WOOD, [-0.08, 0.15, 0], [0.03, 0.3, 0.03], true);
      part(box(), WOOD, [0.0, 0.02, 0], [0.2, 0.03, 0.04], true);
      part(box(), WOOD, [0.0, 0.26, 0], [0.2, 0.03, 0.03], true).rotation.z = -0.35;
      for (let k = 0; k < 5; k++) part(box(), cream, [-0.05 + 0.03 * k, 0.13 + 0.01 * k, 0], [0.004, 0.2 - 0.025 * k, 0.004]);
      break;
  }
  n.rotation.y = x * 5.0 + y * 3.0;
}

// A pole on a cliff rock, and a white flag with a duck printed on both faces. The cloth turns with the wind.
export function flag(root, summit) {
  Ink.part(root, Ink.cone(0.035, 1.5, 0.028, 5), WOOD, [summit[0], summit[1] + 0.7, summit[2]], 1);
  const cloth = new THREE.Group();
  cloth.position.set(summit[0], summit[1] + 1.22, summit[2]);
  root.add(cloth);
  Ink.part(cloth, Ink.box(), Ink.WHITE, [0.42, 0, 0], [0.84, 0.46, 0.02], { tone: 1.0, outlined: true });
  for (const face of [0.016, -0.016]) {
    Ink.part(cloth, Ink.ball(0.1, 9), Ink.NAVY, [0.4, -0.03, face], [1.05, 0.8, 0.12], { cell: 6, tone: 0.75 }).rotation.z = 0.15;
    Ink.part(cloth, Ink.ball(0.062, 9), Ink.NAVY, [0.49, 0.09, face], [1, 1, 0.12], { cell: 6, tone: 0.75 });
    Ink.part(cloth, Ink.cone(0.03, 0.06, 0.0, 5), Ink.MUSTARD, [0.565, 0.085, face], [1, 1, 0.3], { cell: 6, tone: 0.9 }).rotation.z = -Math.PI / 2;
    Ink.part(cloth, Ink.cone(0.0, 0.1, 0.05, 3), Ink.NAVY, [0.33, 0.0, face], [1, 1, 0.12], { cell: 6, tone: 0.75 }).rotation.z = 2.0;
  }
  return cloth;
}

// A white glove with a coral cuff: a palm, four fingers and a thumb.
// The player's hand: a cartoon glove, palm down with the fingers pointing along +x. Its fingers bend at the knuckle,
// so `curl(glove, 0 to 1)` opens and closes it.
export function glove() {
  const g = new THREE.Group(), o = { cell: 5, outlined: true }, white = Ink.WHITE;
  const hand = new THREE.Group();  // palm up, so what it carries sits in it and the fingers close up round it
  hand.rotation.x = Math.PI;
  g.add(hand);
  const finger = (length, radius) => new THREE.CapsuleGeometry(radius, length, 4, 8).rotateZ(-Math.PI / 2);
  Ink.part(hand, Ink.ball(0.07, 14), white, [0, 0, 0], [1.0, 0.5, 0.95], { ...o, lift: 0.4 });  // the palm
  g.userData.knuckles = [0.055, 0.065, 0.062, 0.05].map((length, k) => {
    const knuckle = new THREE.Group();
    knuckle.position.set(0.05, 0.004, -0.046 + 0.031 * k);
    hand.add(knuckle);
    Ink.part(knuckle, finger(length, 0.017), white, [length / 2 + 0.012, 0, 0], 1, o);
    return knuckle;
  });
  const thumb = new THREE.Group();
  thumb.position.set(0.005, -0.004, 0.058);
  thumb.rotation.set(0.3, -0.9, 0);
  hand.add(thumb);
  Ink.part(thumb, finger(0.04, 0.019), white, [0.03, 0, 0], 1, o);
  g.userData.knuckles.push(thumb);
  for (let k = -1; k <= 1; k++) {  // three stitched lines down the back
    Ink.part(hand, Ink.box(), Ink.NAVY, [-0.01, 0.034, 0.02 * k], [0.05, 0.004, 0.005], { cell: 5, tone: 0.2 });
  }
  const cuff = Ink.part(hand, Ink.cone(0.058, 0.05, 0.052, 14), Ink.CORAL, [-0.085, 0, 0], 1, o);
  cuff.rotation.z = Math.PI / 2;
  const rim = Ink.part(hand, Ink.cone(0.066, 0.014, 0.066, 14), Ink.CORAL, [-0.11, 0, 0], 1, { ...o, lift: 0.3 });
  rim.rotation.z = Math.PI / 2;
  g.scale.setScalar(1.6);
  g.visible = false;
  curl(g, 0);
  return g;
}

export function curl(g, amount) {
  g.userData.knuckles.forEach((knuckle, k) => {
    if (k < 4) knuckle.rotation.z = -0.25 - 1.35 * amount;  // a finger bends in over the palm
    else knuckle.rotation.z = -0.9 * amount;  // the thumb tucks in
  });
}
