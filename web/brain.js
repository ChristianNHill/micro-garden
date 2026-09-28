// One duck's brain: the connectome, its mushroom body learning, and its eyes. web/worker.js runs one per thread;
// web/check.js runs them directly.
import { bySource, parseBundle } from "./bundle.js";
import { DT_MS, LIF, Plasticity } from "./lif.js";
import { Eyes, VIS_TONIC } from "./vision.js";
import { Rng } from "./garden/rng.js";

export const MAX_HZ = 50.0;  // brain/encoder.py: level 1.0 fires each neuron of a set at this rate
export const SENSE_NOISE = 0.3;
export const TICKS = 2;  // brain ticks per 20 ms body step
export const EYE_QUIET = 1e-3;  // of a spike's release: 5 synapses of it move a neuron 5e-5 of the way to firing
const f32 = Math.fround;

export class DuckBrain {
  // brain, eyes: parsed bundles (eyes may be null for a blind duck)
  constructor(brain, eyes, { seed = 0, smarts = 0.5, learns = true } = {}) {
    const a = this.a = brain.arrays, meta = this.meta = brain.meta;
    this.lif = new LIF(bySource(a, "W"));
    for (const i of a["set.kenyon_cells"]) this.lif.threshOffset[i] = meta.kc_threshold;
    // The Kenyon cell -> MBON block lives with learning, so a duck that does not learn still gets its current from it
    this.plastic = new Plasticity(a, a["set.kenyon_cells"], smarts);
    this.learns = learns;
    // The eyes' cells release around a resting level all the time, and the Python subtracts what that resting
    // release delivers every tick (LIF.calibrate). The same sum comes from sending only each cell's departure from
    // rest, and a departure too small to matter (EYE_QUIET) is not sent at all.
    this.eyes = eyes ? new Eyes(eyes.arrays, a) : null;
    this.departure = this.eyes ? new Float32Array(this.eyes.neurons.length) : null;
    this.rng = new Rng(seed);
    this.groups = new Int32Array(meta.neurons).fill(-1);
    const idx = a["decoder.idx"], group = a["decoder.group"];
    for (let k = 0; k < idx.length; k++) this.groups[idx[k]] = group[k];
    this.nGroups = meta.decoder_groups;
    this.viewAt = new Int32Array(meta.neurons).fill(-1);
    a["view.idx"].forEach((neuron, k) => { this.viewAt[neuron] = k; });
    this.viewFired = new Uint8Array(a["view.idx"].length);
  }

  static async create(brainGz, eyesGz, opts) {
    return new DuckBrain(await parseBundle(brainGz), eyesGz ? await parseBundle(eyesGz) : null, opts);
  }

  // brain/encoder.py's graded(): {set: level} -> the sets' neurons and their release, with `noise` of a spiking
  // receptor's shot noise, plus `extra` graded cells (the eyes'). `exact` gives releases instead of levels. The
  // neurons come sorted, laid out once for each combination of sets, so the brain can merge them with its spikes.
  drive(levels, noise = SENSE_NOISE, extra = null, exact = false) {
    const a = this.a;
    const names = Object.keys(levels).filter(name => a[`set.${name}`]);
    const key = names.join(",") + (extra ? `+${extra.idx.length}` : "");
    if (!this.plans) this.plans = new Map();
    if (!this.plans.has(key)) {
      const sources = names.map(name => a[`set.${name}`]).concat(extra ? [extra.idx] : []);
      const at = new Map();  // neuron -> slot; a later source wins, as the Python's assignment does
      sources.forEach((idx, s) => idx.forEach((neuron, k) => at.set(neuron, [s, k])));
      const idx = Int32Array.from([...at.keys()].sort((x, y) => x - y));
      const slots = sources.map(src => new Int32Array(src.length).fill(-1));
      idx.forEach((neuron, pos) => { const [s, k] = at.get(neuron); slots[s][k] = pos; });
      this.plans.set(key, { idx, rel: new Float32Array(idx.length), slots, sorted: true });
    }
    const plan = this.plans.get(key), rel = plan.rel;
    names.forEach((name, s) => {
      const r = exact ? levels[name] : Math.min(Math.max(levels[name] * MAX_HZ * DT_MS / 1000, 0), 1);
      for (const pos of plan.slots[s]) {
        if (pos < 0) continue;
        rel[pos] = noise ? f32(r + noise * ((this.rng.random() < r ? 1 : 0) - r)) : f32(r);
      }
    });
    if (extra) { const slot = plan.slots[names.length]; for (let k = 0; k < slot.length; k++) if (slot[k] >= 0) rel[slot[k]] = extra.rel[k]; }
    return plan;
  }

  tick(g, reward, punish) {
    this.lif.step(g, this.plastic);
    if (this.learns) this.plastic.step(this.lif, reward, punish);
  }

  // One body step: what the duck senses in, the decoder's spike counts per tick out.
  step({ levels, lum = null, gain = 1.0, reward = 0, punish = 0, watch = false }) {
    let vision = null;
    if (this.eyes && lum) {
      const level = this.eyes.step(lum, gain), d = this.departure;
      for (let k = 0; k < d.length; k++) { const x = level[k] - VIS_TONIC; d[k] = Math.abs(x) < EYE_QUIET ? 0 : x; }
      vision = { idx: this.eyes.neurons, rel: d };
    }
    const g = this.drive(levels, SENSE_NOISE, vision);
    const counts = [];
    const { lif, groups, viewAt, viewFired } = this;
    for (let t = 0; t < TICKS; t++) {
      this.tick(g, reward, punish);
      const c = new Array(this.nGroups).fill(0);
      for (let k = 0; k < lif.nFired; k++) {
        const neuron = lif.fired[k], grp = groups[neuron];
        if (grp >= 0) c[grp]++;
        if (watch && viewAt[neuron] >= 0) viewFired[viewAt[neuron]] = 1;
      }
      counts.push(c);
    }
    let view = null;
    if (watch) {
      view = [];
      for (let k = 0; k < viewFired.length; k++) if (viewFired[k]) { view.push(k); viewFired[k] = 0; }
    }
    return { counts, fondness: this.learns ? this.plastic.fondness() : 0, view };
  }

  save() { return { w: this.plastic.w.slice(), slow: this.plastic.slow.slice() }; }
  load({ w, slow }) { this.plastic.w.set(w); this.plastic.slow.set(slow); }
  rest(s) { this.plastic.rest(s); }
}

// web/pack.py's reference schedule through a fresh brain, with learning and no noise.
export function reference(brain, schedule) {
  const duck = new DuckBrain(brain, null, { learns: true });
  const perTick = [];
  for (const seg of schedule) {
    const g = duck.drive(seg.release, 0, null, true);
    for (let t = seg.from; t < seg.to; t++) { duck.tick(g, seg.reward, seg.punish); perTick.push(duck.lif.nFired); }
  }
  return { perTick: Uint32Array.from(perTick), counts: duck.lif.nSpikes, w: duck.plastic.w, slow: duck.plastic.slow };
}

// Milliseconds a body step takes, and spikes a tick, with food smelled on one side and the eyes on a grey scene.
export function bench(duck, steps) {
  const levels = { orn_food_left: 0.4, orn_food_right: 0.1, jo_push_left: 0.25 };
  const lum = new Float32Array(1442).fill(0.5);
  let spikes = 0;
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) { duck.step({ levels, lum }); spikes += duck.lif.nFired; }
  return { msPerStep: (performance.now() - t0) / steps, spikesPerTick: spikes / steps };
}
