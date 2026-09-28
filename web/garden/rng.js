// A seedable random generator with the few draws the garden uses, in place of numpy's default_rng. It does not
// reproduce numpy's numbers: the browser garden is the same model, not the same dice.

export class Rng {
  constructor(seed = 0) {
    // splitmix32 to spread the seed, then sfc32
    let s = (seed >>> 0) ^ 0x9e3779b9;
    const mix = () => {
      s = (s + 0x9e3779b9) >>> 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.a = mix(); this.b = mix(); this.c = mix(); this.d = 1;
    for (let i = 0; i < 12; i++) this.next();
  }

  next() {  // a uint32
    const t = (((this.a + this.b) >>> 0) + this.d) >>> 0;
    this.d = (this.d + 1) >>> 0;
    this.a = this.b ^ (this.b >>> 9);
    this.b = (this.c + (this.c << 3)) >>> 0;
    this.c = ((this.c << 21) | (this.c >>> 11)) >>> 0;
    this.c = (this.c + t) >>> 0;
    return t;
  }

  random() { return this.next() / 4294967296; }
  uniform(lo = 0, hi = 1) { return lo + (hi - lo) * this.random(); }
  integers(lo, hi) {  // one in [lo, hi), or [0, lo) with one argument
    if (hi === undefined) { hi = lo; lo = 0; }
    return lo + Math.floor(this.random() * (hi - lo));
  }
  randoms(n) { return Array.from({ length: n }, () => this.random()); }
  uniforms(n, lo, hi) { return Array.from({ length: n }, () => this.uniform(lo, hi)); }
  // one of `items`, with probabilities `p` (summing to 1) or evenly
  choice(items, p) {
    if (!p) return items[this.integers(items.length)];
    let u = this.random();
    for (let i = 0; i < items.length; i++) { if (u < p[i]) return items[i]; u -= p[i]; }
    return items[items.length - 1];
  }
  permutation(n) {
    const a = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = this.integers(i + 1); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }
  sample(items, k) { return this.permutation(items.length).slice(0, k).map(i => items[i]); }
}
