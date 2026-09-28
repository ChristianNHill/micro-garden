// One brain of brain/lif.py's leaky integrate-and-fire model, in the browser.
//
// The rule and the constants are brain/lif.py's. The difference is how input reaches each neuron:
// torch multiplies the whole weight matrix by every neuron's release each tick, while this walks only
// the neurons that released something, since few fire in any one tick. Neurons are visited in
// ascending order, so each one sums its inputs in the same order torch does, and every value is
// rounded to float32 where torch rounds it. The two agree spike for spike (web/check.py).

export const DT_MS = 10;
const LEAK = Math.fround(0.95), THRESH = 1, REFRAC = 3;
const ADAPT_INC = 1, ADAPT_DECAY = Math.fround(0.95);
const f32 = Math.fround;

// brain.bin as web/pack.py writes it.
export function parseBrain(buf) {
  const head = new DataView(buf, 0, 16);
  if (String.fromCharCode(...new Uint8Array(buf, 0, 4)) !== "MGB1") throw new Error("not a brain.bin");
  const n = head.getUint32(4, true), nnz = head.getUint32(8, true), nv = head.getUint32(12, true);
  let at = 16;
  const take = (Type, count) => { const a = new Type(buf, at, count); at += count * Type.BYTES_PER_ELEMENT; return a; };
  return { n, nnz, values: take(Float32Array, nv), start: take(Int32Array, n + 1),
           post: take(Int32Array, nnz), which: take(Uint16Array, nnz) };
}

export class LIF {
  constructor(brain) {
    this.b = brain;
    const n = brain.n;
    this.V = new Float32Array(n);
    this.syn = new Float32Array(n);
    this.adapt = new Float32Array(n);
    this.ref = new Int8Array(n);
    this.nSpikes = new Uint32Array(n);
    this.threshOffset = new Float32Array(n);  // per neuron, above THRESH; the Kenyon cells' keeps odor codes sparse
    this.fired = new Int32Array(n);           // this tick's spiking neurons, the first nFired of them
    this.nFired = 0;
    this.release = new Float32Array(n);       // scratch: what each neuron releases this tick
    this.active = new Int32Array(n);          // scratch: which neurons release anything
  }

  // Advance one tick. graded is {idx: Int32Array, rel: Float32Array} for non-spiking cells, whose
  // release replaces their spike this tick, or null. Returns the number of neurons that fired.
  step(graded) {
    const { V, syn, adapt, ref, nSpikes, threshOffset, fired, release } = this;
    const n = V.length;
    let nf = 0;
    for (let i = 0; i < n; i++) {
      let v = f32(f32(V[i] * LEAK) + syn[i]);
      const a = adapt[i];
      if (ref[i] > 0) v = 0;
      const spk = v >= f32(f32(THRESH + a) + threshOffset[i]);
      if (spk) { v = 0; ref[i] = REFRAC; fired[nf++] = i; nSpikes[i]++; }
      else if (ref[i] > 0) ref[i]--;
      V[i] = v;
      adapt[i] = f32(f32(a * ADAPT_DECAY) + (spk ? ADAPT_INC : 0));
    }
    this.nFired = nf;
    this.propagate(graded);
    return nf;
  }

  // syn = W . release, visiting only the neurons that release something.
  propagate(graded) {
    const { syn, release, active, fired } = this;
    const { values, start, post, which } = this.b;
    syn.fill(0);
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
      if (r === 0) continue;  // a graded cell at rest that happened to spike
      const end = start[pre + 1];
      if (r === 1) for (let s = start[pre]; s < end; s++) syn[post[s]] = f32(syn[post[s]] + values[which[s]]);
      else for (let s = start[pre]; s < end; s++) syn[post[s]] = f32(syn[post[s]] + f32(values[which[s]] * r));
    }
  }
}
