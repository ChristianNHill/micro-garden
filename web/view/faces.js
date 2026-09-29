// The faces over a duck's head, one for each mood that shows: joy, fear, anger and sorrow. In the garden each is a
// little ball in the mood's colour with the features drawn on its front; the controls card shows them flat.
import * as THREE from "three";
import * as Ink from "./ink.js";

const NAVY = "#1c2a4d";
export const FACE_COLOURS = { joy: "#e6b54a", fear: "#6cc4b8", anger: "#ee6f5c", sorrow: "#7fa6e8" };

// Draws the face for `mood` into a square canvas context of side `n`: the round head too, or with `head` false only
// the features, to lay on a ball.
export function drawFace(g, mood, n, head = true) {
  const u = n / 100;  // the face is laid out on a 100 by 100 grid
  const line = w => { g.lineWidth = w * u; g.lineCap = "round"; g.lineJoin = "round"; g.strokeStyle = NAVY; };
  const arc = (x, y, r, a, b) => { g.beginPath(); g.arc(x * u, y * u, r * u, a, b); g.stroke(); };
  const dot = (x, y, r) => { g.beginPath(); g.arc(x * u, y * u, r * u, 0, 2 * Math.PI); g.fillStyle = NAVY; g.fill(); };
  const path = pts => { g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo : g.moveTo).call(g, x * u, y * u)); g.stroke(); };
  g.clearRect(0, 0, n, n);
  if (head) {
    g.beginPath();
    g.arc(50 * u, 50 * u, 44 * u, 0, 2 * Math.PI);
    g.fillStyle = FACE_COLOURS[mood];
    g.fill();
    line(6);
    g.stroke();
  }
  if (mood === "joy") {  // eyes shut with smiling, and a wide grin
    line(5.5);
    arc(35, 44, 8, Math.PI * 1.1, Math.PI * 1.9);
    arc(65, 44, 8, Math.PI * 1.1, Math.PI * 1.9);
    arc(50, 54, 20, Math.PI * 0.15, Math.PI * 0.85);
  } else if (mood === "fear") {  // brows up, eyes wide, a wobbling mouth
    line(4.5);
    path([[27, 27], [40, 23]]);
    path([[60, 23], [73, 27]]);
    g.beginPath(); g.arc(36 * u, 42 * u, 9 * u, 0, 2 * Math.PI); g.fillStyle = "#fff"; g.fill(); g.stroke();
    g.beginPath(); g.arc(64 * u, 42 * u, 9 * u, 0, 2 * Math.PI); g.fill(); g.stroke();
    dot(36, 43, 3.5); dot(64, 43, 3.5);
    line(5);
    path([[33, 72], [40, 67], [46, 72], [52, 67], [58, 72], [65, 67]]);
  } else if (mood === "anger") {  // brows down to the middle, a clenched frown
    line(6);
    path([[25, 30], [44, 39]]);
    path([[75, 30], [56, 39]]);
    dot(37, 47, 5); dot(63, 47, 5);
    line(5.5);
    arc(50, 82, 16, Math.PI * 1.2, Math.PI * 1.8);
  } else if (mood === "sorrow") {  // brows raised in the middle, eyes down, a frown, and a tear
    line(4.5);
    path([[27, 34], [42, 28]]);
    path([[58, 28], [73, 34]]);
    dot(36, 45, 4.5); dot(64, 45, 4.5);
    line(5.5);
    arc(50, 80, 15, Math.PI * 1.2, Math.PI * 1.8);
    g.beginPath();
    g.moveTo(68 * u, 54 * u);
    g.bezierCurveTo(73 * u, 63 * u, 76 * u, 66 * u, 71 * u, 70 * u);
    g.bezierCurveTo(66 * u, 73 * u, 61 * u, 67 * u, 68 * u, 54 * u);
    g.fillStyle = "#4f9fe0";
    g.fill();
    line(2);
    g.stroke();
  }
}

const features = {};

// The face for `mood` as a ball `size` across, its features on the +z side: turn it to the camera to show them.
export function faceModel(mood, size) {
  if (!features[mood]) {
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 256;
    drawFace(canvas.getContext("2d"), mood, 256, false);
    features[mood] = new THREE.CanvasTexture(canvas);
    features[mood].colorSpace = THREE.SRGBColorSpace;
  }
  const g = new THREE.Group(), r = size / 2;
  Ink.part(g, Ink.ball(r, 18), new THREE.Color(FACE_COLOURS[mood]), [0, 0, 0], 1, { cell: 6, lift: 0.35, outlined: true });
  // the features on a card just in front of the ball, sized to the drawn head so they sit on its face
  const card = new THREE.Mesh(new THREE.PlaneGeometry(size * 100 / 88, size * 100 / 88),
                              new THREE.MeshBasicMaterial({ map: features[mood], transparent: true, toneMapped: false, fog: false, depthWrite: false }));
  card.position.z = r * 1.02;  // clear of the ball everywhere, or its front would hide the middle of the face
  g.add(card);
  return g;
}

// The face for `mood` as a picture the page can show, `n` pixels across.
export function faceImage(mood, n = 22) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = n * 2;
  drawFace(canvas.getContext("2d"), mood, n * 2);
  return canvas.toDataURL();
}

// A puffy heart `size` across, facing +z like the faces: pink for a duck giving comfort, red for one glad to get it.
export const HEART_COLOURS = { giving: "#f28ca6", glad: "#e0435b" };
export function heartModel(kind, size) {
  const s = new THREE.Shape();  // a heart 2 wide, point down, drawn around its middle
  s.moveTo(0, -1);
  s.bezierCurveTo(-0.35, -0.6, -1, -0.2, -1, 0.3);
  s.bezierCurveTo(-1, 0.8, -0.45, 1.05, 0, 0.6);
  s.bezierCurveTo(0.45, 1.05, 1, 0.8, 1, 0.3);
  s.bezierCurveTo(1, -0.2, 0.35, -0.6, 0, -1);
  const geo = new THREE.ExtrudeGeometry(s, { depth: 0.35, bevelEnabled: true, bevelThickness: 0.25, bevelSize: 0.2, bevelSegments: 4, curveSegments: 14 });
  geo.center();
  geo.scale(size / 2, size / 2, size / 2);
  const g = new THREE.Group();
  Ink.part(g, geo, new THREE.Color(HEART_COLOURS[kind]), [0, 0, 0], 1, { cell: 6, lift: 0.35, outlined: true });
  return g;
}

// The heart as a picture for the controls card.
export function heartImage(kind, n = 22) {
  const c = HEART_COLOURS[kind];
  return `<svg width="${n}" height="${n}" viewBox="0 0 22 22" aria-hidden="true"><path d="M11 19 C7 15 3 12 3 8 C3 5 6 3.5 8.5 4.5 C10 5 11 6.5 11 6.5 C11 6.5 12 5 13.5 4.5 C16 3.5 19 5 19 8 C19 12 15 15 11 19 Z" fill="${c}" stroke="#1c2a4d" stroke-width="1.4" stroke-linejoin="round"/></svg>`;
}
