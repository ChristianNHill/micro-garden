// viewer/godot/ink.gd, print.gdshader and outline.gdshader: the palette and the two materials everything in the garden
// is printed with. The look is flat faces shaded towards navy ink by the sun; night is a wash of blue over the
// finished picture, not more ink.
import * as THREE from "three";

export const NAVY = new THREE.Color("#1c2a4d");
export const CREAM = new THREE.Color("#f4ead5");
export const CORAL = new THREE.Color("#ee6f5c");
export const TEAL = new THREE.Color("#2f9c8f");
export const MUSTARD = new THREE.Color("#e6b54a");
export const STONE = new THREE.Color("#c9c2b2");
export const GRASS = new THREE.Color("#b9c46a");
export const WHITE = new THREE.Color("#ffffff");

// Shared by every printed material, as Godot's global shader uniforms are.
export const globals = {
  sun_dir: { value: new THREE.Vector3(0.4, 0.8, 0.3).normalize() },
  daylight: { value: 1.0 },
  screen: { value: 1.0 },
  pixel_ratio: { value: 1.0 },
};

const PRINT_VERTEX = /* glsl */ `
  varying vec3 vView;
  void main() {
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    vView = view.xyz;
    gl_Position = projectionMatrix * view;
  }`;

const PRINT_FRAGMENT = /* glsl */ `
  uniform vec3 sun_dir;
  uniform float daylight;
  uniform float screen;
  uniform float pixel_ratio;
  uniform vec3 base;
  uniform vec3 ink;
  uniform float cell;
  uniform float lift;
  uniform float tone_fixed;
  varying vec3 vView;
  void main() {
    float tone = tone_fixed;
    if (tone < 0.0) {
      vec3 n = normalize(cross(dFdx(vView), dFdy(vView)));  // the face's own normal: flat shading
      n *= sign(n.z + 1e-6);  // towards the camera, whichever way the derivatives wind
      vec3 l = normalize((viewMatrix * vec4(sun_dir, 0.0)).xyz);
      tone = clamp(dot(n, l) * 0.7 + 0.47, 0.0, 1.0);
    }
    tone = clamp(tone + lift, 0.0, 1.0);
    // The Godot garden prints this tone as a halftone screen of ink dots; here it is a smooth shade, which reads
    // more calmly in a browser: as much ink as the dots would cover on average.
    float shade = (1.0 - tone) * 0.66 * 0.66 * 3.14159 * 0.85;
    vec3 col = mix(mix(base, ink, clamp(shade, 0.0, 1.0) * screen), vec3(0.03, 0.05, 0.16), 0.5 * (1.0 - daylight));
    gl_FragColor = vec4(col, 1.0);
    #include <colorspace_fragment>
  }`;

const OUTLINE_VERTEX = /* glsl */ `
  uniform float grow;
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position + normal * grow, 1.0);
  }`;

const OUTLINE_FRAGMENT = /* glsl */ `
  uniform vec3 ink;
  uniform float daylight;
  void main() {
    gl_FragColor = vec4(mix(ink, vec3(0.03, 0.05, 0.16), 0.5 * (1.0 - daylight)), 1.0);
    #include <colorspace_fragment>
  }`;

const cache = new Map();

// The print material for one colour, screen size (px between dots), fixed tone (-1 to shade by the sun) and lift.
export function material(colour, cell = 7.0, tone = -1.0, lift = 0.0) {
  const c = colour instanceof THREE.Color ? colour : new THREE.Color(colour);
  const key = `${c.getHexString()}|${cell}|${tone}|${lift}`;
  if (!cache.has(key)) {
    cache.set(key, new THREE.ShaderMaterial({
      uniforms: { ...globals, base: { value: c.clone() }, ink: { value: NAVY }, cell: { value: cell }, tone_fixed: { value: tone }, lift: { value: lift } },
      vertexShader: PRINT_VERTEX, fragmentShader: PRINT_FRAGMENT, side: THREE.DoubleSide,
    }));
  }
  return cache.get(key);
}

const outlines = new Map();
export function outline(grow = 0.005) {
  if (!outlines.has(grow)) {
    outlines.set(grow, new THREE.ShaderMaterial({
      uniforms: { ink: { value: NAVY }, grow: { value: grow }, daylight: globals.daylight },
      vertexShader: OUTLINE_VERTEX, fragmentShader: OUTLINE_FRAGMENT, side: THREE.BackSide,
    }));
  }
  return outlines.get(grow);
}

// A printed piece under `parent`: a mesh, its colour, where it sits and its scale, and optionally an ink outline.
export function part(parent, geometry, colour, at = [0, 0, 0], size = [1, 1, 1], { cell = 7.0, tone = -1.0, lift = 0.0, outlined = false, grow = 0.005 } = {}) {
  const mesh = new THREE.Mesh(geometry, material(colour, cell, tone, lift));
  mesh.position.set(...at);
  mesh.scale.set(...(typeof size === "number" ? [size, size, size] : size));
  if (outlined) mesh.add(new THREE.Mesh(geometry, outline(grow)));
  parent.add(mesh);
  return mesh;
}

const geometries = new Map();
const shared = (key, make) => { if (!geometries.has(key)) geometries.set(key, make()); return geometries.get(key); };

// Godot's SphereMesh and CylinderMesh as ink.gd makes them.
export const ball = (radius, segments = 8) =>
  shared(`ball|${radius}|${segments}`, () => new THREE.SphereGeometry(radius, segments, Math.max(2, segments >> 1)));
export const cone = (bottom, height, top = 0.0, segments = 6) =>
  shared(`cone|${bottom}|${height}|${top}|${segments}`, () => new THREE.CylinderGeometry(top, bottom, height, segments, 1));
export const box = () => shared("box", () => new THREE.BoxGeometry(1, 1, 1));
