// Holds the browser garden to the Python one. `runChecks(load)` fetches what web/pack.py wrote (load(path) gives an
// ArrayBuffer) and returns one row per check: {name, ok, detail}. web/check.html shows them.
//
// The brain must match spike for spike. The eyes are compared within a tolerance, since flyvis sums its inputs in
// another order. The body, world and readouts run the same inputs as web/parity.py and must agree to rounding.
import { gunzip, parseBundle } from "./bundle.js";
import { reference } from "./brain.js";
import { Eyes } from "./vision.js";
import { HEX_AZ, HEX_EL, N_HEX, luminance } from "./garden/retina.js";
import { World, contacts, daylight, temperature_at, wind_on, DT } from "./garden/fields.js";
import { Physiology, aggression_tone } from "./garden/physiology.js";
import { LABELS, label_of, preset, stack } from "./garden/personality.js";
import * as social from "./garden/social.js";
import * as reactions from "./garden/reactions.js";
import { feelings } from "./garden/emotes.js";
import { Decoder } from "./garden/decoder.js";
import { levels_for } from "./garden/server.js";
import { readout } from "./garden/snapshot.js";
import { catch_up } from "./garden/save.js";
import { Rng } from "./garden/rng.js";

// Largest difference between two nested structures of numbers (booleans count as 0 and 1), and where it is.
export function worst(a, b, path = "") {
  if (typeof a === "boolean") a = +a;
  if (typeof b === "boolean") b = +b;
  if (typeof a === "number" || typeof b === "number") {
    const d = a === b ? 0 : Math.abs(a - b);
    return [Number.isNaN(d) ? Infinity : d, path];
  }
  if (a === null || b === null || typeof a === "string") return [a === b ? 0 : Infinity, path];
  let best = [0, path];
  const keys = Array.isArray(b) || ArrayBuffer.isView(b) ? [...Array(Math.max(a.length, b.length)).keys()] : Object.keys(b);
  for (const k of keys) {
    if (a[k] === undefined) return [Infinity, `${path}.${k} missing`];
    const w = worst(a[k], b[k], `${path}.${k}`);
    if (w[0] > best[0]) best = w;
  }
  return best;
}

const row = (name, [d, where], tol) => ({ name, ok: d <= tol, detail: d === 0 ? "exact" : `worst ${d.toExponential(2)} at ${where}` });

function checkBrain(brain, ref) {
  const got = reference(brain, ref.meta.schedule);
  const want = ref.arrays;
  let first = -1, off = 0;
  for (let t = 0; t < want.per_tick.length; t++) if (got.perTick[t] !== want.per_tick[t] && first < 0) first = t;
  for (let i = 0; i < want.counts.length; i++) if (got.counts[i] !== want.counts[i]) off++;
  const total = a => a.reduce((s, x) => s + x, 0);
  return [
    { name: "brain: every tick fires as many neurons as in Python", ok: first < 0,
      detail: first < 0 ? `${want.per_tick.length} ticks, ${total(want.per_tick).toLocaleString()} spikes` : `first differs at tick ${first}` },
    { name: "brain: every neuron fires as often as in Python", ok: off === 0,
      detail: off === 0 ? `${want.counts.length.toLocaleString()} neurons` : `${off} neurons differ` },
    row("brain: learning changes the same synapses by the same amount", worst(got.w, want["plastic.w"]), 1e-6),
  ];
}

function checkEyes(brain, eyesBundle, ref) {
  const eyes = new Eyes(eyesBundle.arrays, brain.arrays);
  const rows = [];
  let level;
  for (let step = 0; step < ref.meta.eye_steps; step++) {
    const lum = new Float32Array(2 * N_HEX).fill(0.5);
    for (let c = 0; c < N_HEX; c++) if (Math.hypot(HEX_AZ[c], HEX_EL[c]) < 0.02 + 0.012 * step) lum[c] = 0.05;
    level = eyes.step(lum, 1.0);
    if (ref.meta.eye_kept.includes(step)) {
      const want = ref.arrays[`eyes.${step}`], n = eyes.n;
      let d = 0;
      for (let e = 0; e < 2; e++) for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(eyes.state[e][i] - want[e * n + i]));
      rows.push({ name: `eyes: flyvis state after ${step + 1} steps of a loom`, ok: d < 1e-4, detail: `worst ${d.toExponential(2)}` });
    }
  }
  const [d] = worst(Array.from(level), Array.from(ref.arrays["eyes.level"]));
  rows.push({ name: "eyes: what they send the brain", ok: d < 1e-3, detail: `worst ${d.toExponential(2)} of a 0 to 1 release` });
  return rows;
}

function five(labels, start) {
  const b = new Physiology(5, stack(labels.map(x => preset(x))), start.hunger, start.thirst);
  social.init(b);
  return b;
}

function state(b) {
  const keep = ["hunger", "thirst", "fatigue", "sleep_pressure", "boredom", "body_temp", "asleep", "alone_s", "joy", "fear",
                "sorrow", "anger", "scent", "scent_was", "swim_skill", "walk_skill", "eat_skill", "dance_skill", "fight_skill",
                "fashion_skill", "music_skill", "bond", "hand_trust"];
  return Object.fromEntries(keep.filter(k => b[k] !== undefined).map(k => [k, b[k]]));
}

function checkParity(c) {
  const rows = [];
  // the world
  const w = c.world, g = w.garden;
  const world = new World({ food_xy: g.food_xy, danger_xy: g.danger_xy, pond: g.pond, bites: g.bites, wind: w.wind,
                            size: g.size, tree: g.tree, rocks: g.rocks });
  for (let t = 0; t < w.steps; t++) world.step(t * DT);
  rows.push(row("world: smells after a minute of wind", worst({
    odor: w.at.map(p => world.odor_at(p)), danger: w.at.map(p => world.odor_at(p, world.danger_odor)),
    humidity: w.at.map(p => world.humidity_at(p)), temp: w.at.map(p => temperature_at(p, 0.7, world.tree)),
    grid_sum: [world.odor.reduce((a, x) => a + x, 0), world.danger_odor.reduce((a, x) => a + x, 0), world.damp.reduce((a, x) => a + x, 0)],
  }, { odor: w.odor, danger: w.danger, humidity: w.humidity, temp: w.temp, grid_sum: w.grid_sum }), 1e-9));
  const hs = [-3, -2, -1, 0, 1, 2, 3];
  rows.push(row("world: daylight, wind on the antennae, touch", worst({
    daylight: w.daylight.map((_, k) => daylight(1300 * k / 36)), wind_on: hs.map(h => wind_on(h, [0.3, -0.9])),
  }, { daylight: w.daylight, wind_on: w.wind_on }), 1e-12));
  // what the ducks see
  const r = c.retina;
  const lum = luminance(r.xy, r.heading, { food: r.food, pond: r.pond, rocks: r.rocks, balls: r.balls, hand: r.hand, tree: r.tree }, r.light);
  const flat = lum.flatMap(x => Array.from(x));
  let off = 0;
  flat.forEach((v, k) => { if (Math.abs(v - r.lum[k]) > 1e-6) off++; });
  rows.push({ name: "retina: what three ducks see", ok: off <= 3, detail: off ? `${off} of ${flat.length} columns differ` : "exact" });
  // bodies, social lives, the senses and every readout, twenty steps
  const bc = c.body, b = five(bc.labels, bc.start);
  b.body_temp = [...bc.start.body_temp];
  let wb = [0, ""];
  bc.steps.forEach((s, k) => {
    const f = s.frame;
    const [falls, wakes] = b.step(0.5, f, s.escaped, s.speed);
    social.update(b, f, 0.5, s.speed, new Rng(0));
    const got = {
      falls, wakes, state: state(b), gains: b.sense_gains(), motor: b.motor(s.wants, s.damp), cooling: b.cooling(), play: b.play(),
      steering: social.steering(b, f, s.scent), feelings: [0, 1, 2, 3, 4].map(i => feelings(b, i)),
      readout: [0, 1, 2, 3, 4].map(i => readout(b, i).map(x => x[2])),
      aggression: [0, 1, 2, 3, 4].map(i => aggression_tone(b.k.aggressiveness[i], b.hunger[i], 0.4, b.anger[i], b.k.kindness[i])),
    };
    const lv = levels_for(b, f, new Array(5).fill(0));
    got.levels = lv.levels;
    got.following = lv.following;
    const want = { ...s, readout: s.readout.map(rows => rows.map(x => x[2])) };
    const d = worst(got, Object.fromEntries(Object.keys(got).map(key => [key, want[key]])), `step ${k}`);
    if (d[0] > wb[0]) wb = d;
  });
  rows.push(row("body: drives, moods, skills, bonds, senses and readouts over 20 steps", wb, 1e-9));
  // the decoder
  const dc = c.decoder;
  let wd = [0, ""];
  dc.cases.forEach((cs, k) => {
    const dec = new Decoder(dc.sizes, 1, new Rng(k));
    dec.wander = [cs.wander];
    dec.ticks = 1;
    dec.rates[0].set(cs.rates);
    dec.body = cs.body;
    const it = dec.update([new Array(14).fill(0)])[0];
    const d = worst({ rates: Array.from(dec.rates[0]), intent: it }, { rates: cs.rates_after, intent: cs.intent }, `case ${k}`);
    if (d[0] > wd[0]) wd = d;
  });
  rows.push(row("decoder: rates and intents", wd, 1e-5));
  // who reacts how, skills, friends, labels
  const sc = c.social, s5 = five(bc.labels, { hunger: 0.2, thirst: 0.2 });
  s5.bond = sc.bond.map(r => [...r]);
  const kinds = ["shove", "fall", "laugh", "comfort", "angry", "sad"];
  rows.push(row("social: reaction weights, practice, friends and labels", worst({
    weights: [[0, 1, 2], [2, 0, 2], [3, 4, 1], [1, 3, -1]].map(([wd2, a, t]) => Object.fromEntries(kinds.map(kind => [kind, reactions.weights(kind, s5, wd2, a, t)]))),
    practised: sc.practised.map(([s, d, st, idle]) => [s, d, st, idle, social.practised(s, d, st, idle)]),
    friends: [0, 1, 2, 3, 4].map(i => social.friends(s5, i, bc.labels)),
    label_of: Object.keys(LABELS).map((_, i) => label_of(stack(Object.keys(LABELS).map(x => preset(x))), i)),
  }, { weights: sc.weights, practised: sc.practised, friends: sc.friends, label_of: sc.label_of }), 1e-12));
  // a night away
  const sv = c.save, away = five(bc.labels, { hunger: 0.3, thirst: 0.4 });
  catch_up(away, sv.gap, sv.since);
  rows.push(row("save: a garden caught up after seven hours away", worst(state(away), sv.state), 1e-9));
  return rows;
}

export async function runChecks(load) {
  const [brain, eyes, ref, parity] = await Promise.all([
    load("data/brain.bin.gz").then(parseBundle), load("data/eyes.bin.gz").then(parseBundle),
    load("data/reference.bin.gz").then(parseBundle),
    load("data/parity.json.gz").then(gunzip).then(buf => JSON.parse(new TextDecoder().decode(buf))),
  ]);
  const rows = [];
  for (const [part, run] of [["parity", () => checkParity(parity)], ["eyes", () => checkEyes(brain, eyes, ref)], ["brain", () => checkBrain(brain, ref)]]) {
    try { rows.push(...run()); } catch (e) { rows.push({ name: part, ok: false, detail: String((e && e.stack) || e) }); }
  }
  return rows;
}
