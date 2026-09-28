// brain/save.py: saving a garden and catching it up on the next visit. The Python writes a .npz; here the garden goes
// to the browser's local storage, which can refuse (a private window, cleared site data), so every use is guarded.
// Catching up integrates the drives in CATCH_UP_S steps with no body and no world; learned weights fade on their
// own clocks.
import { daylight } from "./fields.js";
import { IDS } from "./stub.js";

export const AMBIENT_C = 24.0;
export const CATCH_UP_S = 5.0;
export const MAX_GAP_S = 3 * 24 * 3600.0;
export const KEY = "micro-garden";
const VERSION = 1;

const BODY_SKIP = new Set(["n", "k"]);

function toB64(f32) {
  const bytes = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(s);
}

function fromB64(text) {
  const s = atob(text), bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

// weights: one {w, slow} a duck from the brains, or null
export function save(body, weights, stub, when = Date.now() / 1000) {
  const state = { version: VERSION, when, garden_t: stub.t, body: {}, knobs: body.k };
  for (const [k, v] of Object.entries(body)) if (!BODY_SKIP.has(k) && (Array.isArray(v) || typeof v === "number")) state.body[k] = v;
  if (weights) state.weights = weights.map(({ w, slow }) => ({ w: toB64(w), slow: toB64(slow) }));
  const w = stub.world;
  Object.assign(state, {
    pose: stub.pose, hats: stub.hats, duck_names: stub.duck_names, hat_style: stub.hat_style, hat_items: stub.hat_items,
    food: w.food, bites: w.bites, kinds: w.kinds, hand: w.hand, music: w.music,
  });
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
    return true;
  } catch {
    return false;
  }
}

export function saved() {
  try {
    const text = localStorage.getItem(KEY);
    const state = text && JSON.parse(text);
    return state && state.version === VERSION ? state : null;
  } catch {
    return null;
  }
}

export function forget() {
  try { localStorage.removeItem(KEY); } catch { /* nothing to forget */ }
}

// Put a saved garden back and run the drives forward over the gap. Returns [gap in seconds, the weights to load].
export function load(state, body, stub, now = Date.now() / 1000) {
  for (const [k, v] of Object.entries(state.body)) if (k in body) body[k] = v;
  for (const [k, v] of Object.entries(state.knobs)) body.k[k] = v;
  const gap = Math.min(Math.max(now - state.when, 0), MAX_GAP_S);
  const since = state.garden_t || 0;
  const w = stub.world;
  stub.pose = state.pose;
  stub.hats = state.hats;
  stub.duck_names = state.duck_names;
  stub.hat_style = state.hat_style;
  stub.hat_items = state.hat_items;
  stub.t = since + gap;
  w.set_food(state.food, state.bites, state.kinds);
  w.hand = state.hand;
  w.music = state.music;
  w.odor.fill(0);
  w.diffuse(2000);
  catch_up(body, gap, since);
  const weights = state.weights ? state.weights.map(({ w: ww, slow }) => ({ w: fromB64(ww), slow: fromB64(slow) })) : null;
  return [gap, weights];
}

// Age the drives over a gap with no garden to react to: hungrier, thirstier, and rested. Each step carries its own daylight.
export function catch_up(body, gap_s, since = 0.0) {
  const n = body.n;
  const zeros = () => new Array(n).fill(0);
  const quiet = { temp_left: new Array(n).fill(AMBIENT_C), temp_right: new Array(n).fill(AMBIENT_C) };
  for (const name of ["ate", "drank", "bumped", "swimming", "touch_left", "touch_right", "odor_left", "odor_right", "show", "kicked", "drummed", "light"]) quiet[name] = zeros();
  for (const name of IDS) quiet[name] = new Array(n).fill(-1);
  const still = new Array(n).fill(false), nothing = zeros();
  for (let step = 0; step < Math.floor(gap_s / CATCH_UP_S); step++) {
    quiet.light.fill(daylight(since + step * CATCH_UP_S));
    body.step(CATCH_UP_S, quiet, still, nothing);
  }
}
