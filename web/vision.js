// brain/vision.py: flyvis (trained on optic flow) stands in for the optic lobe, and its columns drive the matching
// FlyWire cells. Its network is passive point neurons with graded synapses, stepped by Euler:
//   v += dt / max(tau, dt) * (-v + bias + sum(w * relu(v_source)) + input)
// with the image added to the photoreceptors. Both eyes of one duck run here, left eye first.
import { bySource } from "./bundle.js";

export const DT_S = 0.02;
export const VIS_GAIN = 0.05;
export const VIS_ADAPT_S = 3.0;
export const VIS_FLOOR = 0.005;
export const VIS_TONIC = 0.5;
const N_HEX = 721;
const f32 = Math.fround;

export class Eyes {
  // eyes: the eyes bundle's arrays; mapping: the brain bundle's arrays (which FlyWire cell each flyvis cell drives)
  constructor(eyes, mapping) {
    this.net = bySource(eyes, "net");
    const n = this.net.n;
    this.n = n;
    const dt = f32(DT_S);
    this.inv = Float32Array.from(eyes["net.tau"], tau => f32(1 / Math.max(tau, dt)));
    this.bias = eyes["net.bias"];
    this.input = eyes["net.input"];  // (8 input types) x 721 columns
    this.rest = eyes["rest"];
    this.state = [Float32Array.from(eyes["settled"]), Float32Array.from(eyes["settled"])];
    this.current = new Float64Array(n);  // summed in double precision: flyvis sums in another order anyway
    this.weight = Float32Array.from(this.net.which, w => this.net.values[w]);
    this.x = new Float32Array(n);
    this.neurons = mapping["vision.neurons"];
    this.src = mapping["vision.src"];
    this.eye = mapping["vision.eye"];
    this.level = new Float32Array(this.neurons.length);
    this.scale = null;
  }

  // One flyvis step of one eye from its image (721 intensities).
  _stepEye(v, lum) {
    const { start, post } = this.net, weight = this.weight, cur = this.current, x = this.x, n = this.n;
    cur.fill(0);
    for (let s = 0; s < n; s++) {
      const a = v[s];
      if (a <= 0) continue;
      const end = start[s + 1];
      for (let k = start[s]; k < end; k++) cur[post[k]] += weight[k] * a;
    }
    x.fill(0);
    const rows = this.input.length / N_HEX;
    for (let t = 0; t < rows; t++) for (let c = 0; c < N_HEX; c++) x[this.input[t * N_HEX + c]] = f32(x[this.input[t * N_HEX + c]] + lum[c]);
    const dt = f32(DT_S);
    for (let i = 0; i < n; i++) {
      const vel = f32(this.inv[i] * f32(f32(f32(-v[i] + this.bias[i]) + f32(cur[i])) + x[i]));
      v[i] = f32(v[i] + f32(vel * dt));
    }
  }

  // One body step: lum is both eyes' images (2 x 721), gain the duck's vision gain. Returns the release of each
  // driven FlyWire cell (`this.neurons`), for LIF.step's graded input.
  step(lum, gain = 1.0) {
    this._stepEye(this.state[0], lum.subarray(0, N_HEX));
    this._stepEye(this.state[1], lum.subarray(N_HEX, 2 * N_HEX));
    const { neurons, src, eye, level, rest } = this;
    let sq = 0;
    for (let k = 0; k < neurons.length; k++) {
      const raw = f32(this.state[eye[k]][src[k]] - rest[src[k]]);
      level[k] = raw;
      sq += raw * raw;
    }
    const rms = Math.sqrt(sq / neurons.length);
    const a = DT_S / VIS_ADAPT_S;
    this.scale = this.scale === null ? rms : (1 - a) * this.scale + a * rms;
    const g = VIS_GAIN * gain / Math.max(this.scale, VIS_FLOOR);
    for (let k = 0; k < level.length; k++) level[k] = Math.min(Math.max(level[k] * g + VIS_TONIC, 0), 1);
    return level;
  }
}
