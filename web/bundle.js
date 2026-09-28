// Reads the bundles web/pack.py writes: gzipped, "MGB2", a JSON header naming each array, then the arrays.

const TYPES = { f4: Float32Array, f8: Float64Array, i4: Int32Array, u4: Uint32Array, i2: Int16Array, u2: Uint16Array,
                i1: Int8Array, u1: Uint8Array };

// Some web servers send a .gz file marked as compressed, and the browser unzips it on the way in; then there is
// nothing left to unzip.
export async function gunzip(buf) {
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] !== 0x1f || head[1] !== 0x8b) return buf;
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

// {arrays: {name: typed array}, meta} from a bundle's gzipped bytes.
export async function parseBundle(gz) {
  const buf = await gunzip(gz);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
  if (magic !== "MGB2") throw new Error("not a Micro Garden bundle");
  const len = new DataView(buf).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, len)));
  const base = 8 + len, arrays = {};
  for (const [name, { dtype, shape, offset }] of Object.entries(header.arrays)) {
    const Type = TYPES[dtype];
    arrays[name] = new Type(buf, base + offset, shape.reduce((a, b) => a * b, 1));
  }
  return { arrays, meta: header.meta };
}

export async function loadBundle(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return parseBundle(await res.arrayBuffer());
}

// Synapses grouped by source, as web/pack.py's by_source writes them under `prefix`: each target back from its gap.
export function bySource(a, prefix) {
  const start = a[`${prefix}.start`], gap = a[`${prefix}.gap`];
  const post = new Int32Array(gap.length);
  for (let s = 0; s + 1 < start.length; s++) {
    let at = 0;
    for (let k = start[s]; k < start[s + 1]; k++) { at = k === start[s] ? gap[k] : at + gap[k]; post[k] = at; }
  }
  return { n: start.length - 1, start, post, which: a[`${prefix}.which`], values: a[`${prefix}.values`] };
}
