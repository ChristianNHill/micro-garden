// One brain of brain/lif.py's leaky integrate-and-fire model, with brain/plasticity.py's mushroom body learning.
//
// The rule and the constants are the Python's. The difference is how input reaches each neuron: torch multiplies
// the whole weight matrix by every neuron's release each tick, while this walks only the neurons that released
// something, since few fire in any one tick. Neurons are visited in ascending order, so each one sums its inputs
// in the same order torch does, and every value is rounded to float32 where torch rounds it. The two agree spike
// for spike (web/check.js).

export const DT_MS = 10;
const LEAK = Math.fround(0.95), THRESH = 1, REFRAC = 3;
const ADAPT_INC = 1, ADAPT_DECAY = Math.fround(0.95);
const f32 = Math.fround;

export class LIF {
  // brain: synapses by source (bundle.js bySource), without the Kenyon cell -> MBON block when learning owns it
  constructor(brain) {
    this.b = brain;
    const n = brain.n;
    this.V = new Float32Array(n);
    this.syn = new Float32Array(n);
    this.adapt = new Float32Array(n);
    this.ref = new Int8Array(n);
    this.nSpikes = new Uint32Array(n);
    this.threshOffset = new Float32Array(n);  // above THRESH; the Kenyon cells' keeps odor codes sparse
    this.rest = null;  // the current graded cells deliver at rest, subtracted every tick (`calibrate`)
    this.fired = new Int32Array(n);  // this tick's spiking neurons, the first nFired of them
    this.nFired = 0;
    this.spiked = new Uint8Array(n);  // this tick's spikes as a mask
    this.release = new Float32Array(n);
    this.active = new Int32Array(n);
  }

  // Advance one tick. graded is {idx, rel} for non-spiking cells, whose release replaces their spike this tick,
  // or null; plastic a Plasticity that owns the Kenyon cell -> MBON block. Returns how many neurons fired.
  step(graded, plastic = null) {
    const { V, syn, adapt, ref, nSpikes, threshOffset, fired, spiked } = this;
    const n = V.length;
    let nf = 0;
    for (let i = 0; i < n; i++) {
      let v = f32(f32(V[i] * LEAK) + syn[i]);
      const a = adapt[i];
      if (ref[i] > 0) v = 0;
      const spk = v >= f32(f32(THRESH + a) + threshOffset[i]);
      if (spk) { v = 0; ref[i] = REFRAC; fired[nf++] = i; nSpikes[i]++; spiked[i] = 1; }
      else { if (ref[i] > 0) ref[i]--; spiked[i] = 0; }
      V[i] = v;
      adapt[i] = f32(f32(a * ADAPT_DECAY) + (spk ? ADAPT_INC : 0));
    }
    this.nFired = nf;
    this.propagate(graded);
    if (this.rest) for (let i = 0; i < n; i++) syn[i] = f32(syn[i] - this.rest[i]);
    if (plastic) plastic.addCurrent(this);
    return nf;
  }

  // syn = W . release, visiting only the neurons that release something, in ascending order. When the graded cells
  // come sorted (`graded.sorted`, as DuckBrain gives them) they are merged with this tick's spikes, which are in
  // order already; otherwise everything that releases is sorted first.
  propagate(graded, out = this.syn) {
    const { release, active, fired } = this;
    const { values, start, post, which } = this.b;
    out.fill(0);
    const send = (pre, r) => {
      const end = start[pre + 1];
      if (r === 1) for (let s = start[pre]; s < end; s++) out[post[s]] = f32(out[post[s]] + values[which[s]]);
      else for (let s = start[pre]; s < end; s++) out[post[s]] = f32(out[post[s]] + f32(values[which[s]] * r));
    };
    if (graded && graded.sorted) {
      const { idx, rel } = graded, nf = this.nFired, ng = idx.length;
      let i = 0, j = 0;
      while (i < nf || j < ng) {
        const a = i < nf ? fired[i] : Infinity, g = j < ng ? idx[j] : Infinity;
        if (g <= a) { if (rel[j] !== 0) send(g, rel[j]); j++; if (g === a) i++; }
        else { send(a, 1); i++; }
      }
      return;
    }
    let na = 0;
    for (let k = 0; k < this.nFired; k++) { release[fired[k]] = 1; active[na++] = fired[k]; }
    if (graded) {
      const { idx, rel } = graded;
      for (let k = 0; k < idx.length; k++) {
        if (release[idx[k]] === 0 && rel[k] !== 0) active[na++] = idx[k];
        release[idx[k]] = rel[k];
      }
    }
    const act = active.subarray(0, na).sort();
    for (let k = 0; k < na; k++) {
      const pre = act[k], r = release[pre];
      release[pre] = 0;
      if (r !== 0) send(pre, r);
    }
  }

  // Take the current a set of graded cells delivers at rest as the zero point (LIF.calibrate).
  calibrate(indices, restRelease) {
    this.rest = new Float32Array(this.V.length);
    const saved = this.nFired;
    this.nFired = 0;
    this.propagate({ idx: indices, rel: new Float32Array(indices.length).fill(restRelease) }, this.rest);
    this.nFired = saved;
  }
}

// brain/plasticity.py: dopamine-gated depression at Kenyon cell -> MBON synapses, one duck's worth.
export const MB_GAIN = 10.0;
export const KC_TRACE_MS = 1000.0;
export const DEPRESS_PER_S = 3.0;
export const DOPAMINE_S = 1.0;
export const RECOVER_S = 300.0;
export const CONSOLIDATE = 0.2;
export const SLOW_RECOVER_S = 3 * 24 * 3600.0;

export class Plasticity {
  // a: the brain bundle's arrays; kc: the Kenyon cells; smarts: the personality knob
  constructor(a, kc, smarts = 0.5) {
    this.post = a["plastic.post"];
    this.pre = a["plastic.pre"];
    this.preLocal = a["plastic.pre_local"];
    this.base = a["plastic.base"];
    this.rewardGate = a["plastic.reward_gate"];
    this.punishGate = a["plastic.punish_gate"];
    this.kc = kc;
    this.w = Float32Array.from(this.base);
    this.slow = Float32Array.from(this.base);
    this.trace = new Float32Array(kc.length);
    this.dopamine = [0, 0];  // reward, punishment: what is left of each burst
    this.rate = f32(0.5 + f32(smarts));
    this.out = new Float32Array(0);
  }

  // Add this tick's current from Kenyon cell spikes (Plasticity.current), after the rest of the brain's.
  addCurrent(lif) {
    const { post, pre, w } = this, { spiked, syn } = lif;
    const touched = [];
    if (this.out.length !== syn.length) this.out = new Float32Array(syn.length);
    const out = this.out;
    for (let e = 0; e < post.length; e++) {
      if (!spiked[pre[e]]) continue;
      if (out[post[e]] === 0) touched.push(post[e]);
      out[post[e]] = f32(out[post[e]] + w[e]);
    }
    for (const m of touched) { syn[m] = f32(syn[m] + out[m]); out[m] = 0; }
  }

  // Advance one tick. reward and punish are dopamine levels in [0, 1].
  step(lif, reward, punish) {
    const decay = f32(Math.exp(-DT_MS / KC_TRACE_MS));
    const { trace, kc } = this, spiked = lif.spiked;
    for (let k = 0; k < kc.length; k++) trace[k] = Math.min(f32(f32(trace[k] * decay) + spiked[kc[k]]), 1.0);
    const fade = f32(Math.exp(-DT_MS / 1000 / DOPAMINE_S));
    this.dopamine = [Math.max(f32(this.dopamine[0] * fade), f32(reward)), Math.max(f32(this.dopamine[1] * fade), f32(punish))];
    const [dr, dp] = this.dopamine;
    const k = f32(DEPRESS_PER_S * DT_MS / 1000);
    const { w, slow, rewardGate, punishGate, preLocal } = this;
    for (let e = 0; e < w.length; e++) {
      const dope = Math.min(Math.max(f32(f32(dr * rewardGate[e]) + f32(dp * punishGate[e])), 0), 1);
      if (dope === 0) continue;
      const eligible = trace[preLocal[e]];
      if (eligible === 0) continue;
      const lost = f32(f32(f32(f32(w[e] * eligible) * dope) * this.rate) * k);
      w[e] = f32(w[e] - lost);
      slow[e] = f32(slow[e] - f32(f32(CONSOLIDATE) * lost));
    }
    this.rest(DT_MS / 1000);
  }

  // Forget for s seconds with nothing happening: one tick of it, or a night away.
  rest(s) {
    const { w, slow, base } = this;
    const slowBack = f32(1 - Math.exp(-s / SLOW_RECOVER_S)), fastBack = f32(1 - Math.exp(-s / RECOVER_S));
    for (let e = 0; e < w.length; e++) {
      const we = Math.max(w[e], 0), before = Math.max(slow[e], 0);
      const sl = f32(before + f32(f32(base[e] - before) * slowBack));
      const back = f32(f32(f32(sl - we) * fastBack) + f32(sl - before));
      w[e] = Math.min(f32(we + back), sl);
      slow[e] = sl;
    }
  }

  // What this duck has learned about the smell in its nose right now, -1 to 1 (Plasticity.fondness).
  fondness() {
    let rGone = 0, rNose = 0, pGone = 0, pNose = 0;
    const { w, base, trace, preLocal, rewardGate, punishGate } = this;
    for (let e = 0; e < w.length; e++) {
      const nose = trace[preLocal[e]];
      if (nose === 0) continue;
      const gone = Math.max(1 - w[e] / base[e], 0) * nose;
      if (rewardGate[e]) { rGone += gone; rNose += nose; }
      if (punishGate[e]) { pGone += gone; pNose += nose; }
    }
    return rGone / (rNose + 1e-6) - pGone / (pNose + 1e-6);
  }
}
