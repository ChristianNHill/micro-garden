// The garden's sounds, synthesised as viewer/godot/main.gd synthesises them: each duck's quack (a pitch-swept buzz
// shaped by the tag, its voice seeded from its number), notes on the instruments, the drum, and the pond. The music
// box plays files the visitor picks, or a tune of its own; no music ships with the garden.
import { Rng } from "../garden/rng.js";

const RATE = 22050;
const TAU = 2 * Math.PI;
// Each instrument's lowest note in Hz, how long a note rings, and which voice plays it (stub.js INSTRUMENTS order).
const INSTRUMENT_TONES = [[880, 0.4, "struck"], [0, 0.14, "shaken"], [0, 0.3, "jingle"], [2400, 1.0, "ring"], [523, 0.6, "struck"],
                          [196, 0.9, "plucked"], [349, 0.4, "brass"], [660, 1.0, "ring"], [784, 0.45, "blown"], [392, 0.9, "harp"]];
const QUACKS = { alarm: [900, 0.5, 0.12, 3], greet: [520, 0.8, 0.16, 2], inquire: [480, 1.3, 0.22, 1], peck: [700, 0.9, 0.05, 2],
                 chirp: [1100, 1.1, 0.07, 2], coo: [330, 0.9, 0.4, 1], wheee: [600, 1.8, 0.45, 1] };
const PENTATONIC = [0, 2, 4, 7, 9, 12];

export class Sound {
  constructor() {
    this.ctx = null;
    this.on = true;
    this.voices = new Map();
    this.tracks = [];  // files the visitor picked for the music box
    this.title = "";
  }

  // Browsers only let a page make sound after the visitor does something; call this from a tap or a key.
  start() {
    if (this.ctx) { this.ctx.resume(); return; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = this.on ? 1 : 0;
    this.master.connect(this.ctx.destination);
    this._pond();
    this.musicBus = this.ctx.createGain();
    this.musicBus.gain.value = 0;
    this.musicBus.connect(this.master);
  }

  setOn(on) {
    this.on = on;
    if (this.ctx) this.master.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.05);
  }

  _play(samples, gain = 1) {
    if (!this.ctx || !this.on) return;
    const buf = this.ctx.createBuffer(1, samples.length, RATE);
    buf.copyToChannel(samples, 0);
    const src = this.ctx.createBufferSource(), g = this.ctx.createGain();
    src.buffer = buf;
    g.gain.value = gain;
    src.connect(g).connect(this.master);
    src.start();
  }

  quack(duck, tag, skill = 1) {
    if (tag === "drum") return this._thump();
    if (tag.startsWith("instrument:")) { const [, kind, sour] = tag.split(":"); return this._note(+kind, !!sour, skill); }
    const [f0, sweep, dur, reps] = QUACKS[tag] ?? [500, 0.8, 0.15, 1];
    const v = new Rng(9001 + duck * 7919);
    const high = [0.62, 0.8, 1.0, 1.22, 1.5][duck % 5] * v.uniform(0.94, 1.06);
    const reedy = v.uniform(0, 1), wobble = v.uniform(0, 0.06), wobbleHz = v.uniform(9, 22), pace = v.uniform(0.8, 1.3);
    const n = Math.trunc(dur * pace * RATE), gap = Math.trunc(900 * pace);
    const out = new Float32Array(reps * (n + gap));
    let phase = 0, at = 0;
    for (let r = 0; r < reps; r++) {
      for (let k = 0; k < n + gap; k++, at++) {
        const u = k / n;
        const f = f0 * high * (1 + (sweep - 1) * u) * (1 + wobble * Math.sin(TAU * wobbleHz * k / RATE));
        phase += f / RATE;
        const saw = (phase % 1) * 2 - 1;
        out[at] = k < n ? (Math.sin(TAU * phase) * (1 - reedy) + saw * reedy) * Math.sin(Math.min(u, 1) * Math.PI) * 0.25 : 0;
      }
    }
    this._play(out);
  }

  // A note from a pentatonic scale, so any run of them sounds like a tune; a less practised duck drifts off pitch.
  _note(kind, sour, skill) {
    const [base, secs, voice] = INSTRUMENT_TONES[kind % INSTRUMENT_TONES.length];
    const off = (Math.random() * 2 - 1) * ((1 - skill) * 0.6 + (sour ? 1.8 : 0));
    const hz = base * 2 ** ((PENTATONIC[Math.trunc(Math.random() * 6)] + off) / 12);
    const n = Math.trunc(secs * RATE), out = new Float32Array(n);
    let phase = 0;
    for (let k = 0; k < n; k++) {
      const u = k / n, t = k / RATE;
      phase += hz / RATE;
      let v = 0;
      switch (voice) {
        case "struck": v = (Math.sin(TAU * phase) + 0.3 * Math.sin(TAU * phase * 4)) * Math.exp(-7 * u); break;
        case "shaken": v = (Math.random() * 2 - 1) * Math.exp(-9 * u); break;
        case "jingle": v = (Math.random() * 2 - 1) * Math.sin(TAU * t * 5200) * Math.exp(-6 * u); break;
        case "ring": v = (Math.sin(TAU * phase) + 0.4 * Math.sin(TAU * phase * 2.76) + 0.2 * Math.sin(TAU * phase * 5.4)) * Math.exp(-3 * u); break;
        case "plucked": v = ((phase % 1) * 2 - 1) * Math.exp(-5 * u) * 0.6; break;
        case "harp": v = Math.sin(TAU * phase) * Math.exp(-4 * u); break;
        case "brass": v = Math.min(Math.max(Math.sin(TAU * phase) * 3, -1), 1) * Math.min(u * 12, 1) * (1 - u) * 0.7; break;
        case "blown": v = Math.sin(TAU * phase + 0.3 * Math.sin(TAU * t * 5.5)) * Math.min(u * 10, 1) * (1 - u); break;
      }
      out[k] = v * 0.09;
    }
    this._play(out);
  }

  _thump() {
    const n = Math.trunc(0.14 * RATE), out = new Float32Array(n);
    let turn = 0;
    for (let k = 0; k < n; k++) {
      const u = k / n;
      turn += (150 + (80 - 150) * u) / RATE;
      out[k] = Math.sin(TAU * turn) * Math.exp(-5 * u) * 0.09;
    }
    this._play(out);
  }

  // Brown noise, a leaky sum of white, rising and falling slowly: the pond. Quieter at night.
  _pond() {
    const secs = 8, n = secs * RATE, out = new Float32Array(n);
    let level = 0;
    for (let k = 0; k < n; k++) {
      const t = k / RATE;
      level = Math.min(Math.max(level * 0.995 + (Math.random() * 0.08 - 0.04), -1), 1);
      out[k] = level * (0.6 + 0.4 * Math.sin(t * TAU / secs) * Math.sin(t * TAU * 2 / secs));
    }
    const buf = this.ctx.createBuffer(1, n, RATE);
    buf.copyToChannel(out, 0);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    this.pondGain = this.ctx.createGain();
    this.pondGain.gain.value = 0;
    src.connect(this.pondGain).connect(this.master);
    src.start();
  }

  pond(present, light) {
    if (this.pondGain) this.pondGain.gain.setTargetAtTime(present ? 0.05 * (0.4 + 0.6 * light) : 0, this.ctx.currentTime, 0.5);
  }

  // The music box: the visitor's own files, shuffled, or a slow music-box tune while the box is in the garden.
  pickTracks(files) {
    this.tracks = [...files].filter(f => f.type.startsWith("audio/"));
    this.queue = [];
    if (this.audioEl) { this.audioEl.pause(); this.audioEl = null; }
  }

  music(down, volume) {
    if (!this.ctx) return;
    const want = down ? volume : 0;
    this.musicBus.gain.setTargetAtTime(want * 0.6, this.ctx.currentTime, 0.4);
    if (!down) { this.title = ""; if (this.audioEl) this.audioEl.pause(); return; }
    if (this.tracks.length) this._file();
    else this._tune();
  }

  _file() {
    if (this.audioEl && !this.audioEl.ended) { if (this.audioEl.paused) this.audioEl.play(); return; }
    if (!this.queue || !this.queue.length) this.queue = [...this.tracks].sort(() => Math.random() - 0.5);
    const file = this.queue.pop();
    this.audioEl = new Audio(URL.createObjectURL(file));
    this.ctx.createMediaElementSource(this.audioEl).connect(this.musicBus);
    this.audioEl.play();
    this.title = file.name.replace(/\.[^.]+$/, "");
  }

  _tune() {
    const now = this.ctx.currentTime;
    if (this.tuneUntil && now < this.tuneUntil - 0.5) return;
    const start = Math.max(now, this.tuneUntil || now), step = 0.32;
    const scale = [523.25, 587.33, 659.25, 783.99, 880.0, 1046.5];
    let note = 2;
    for (let k = 0; k < 16; k++) {
      note = Math.max(0, Math.min(scale.length - 1, note + [-2, -1, 1, 2, 0][Math.trunc(Math.random() * 5)]));
      if (k % 8 === 7 && Math.random() < 0.5) continue;
      const osc = this.ctx.createOscillator(), g = this.ctx.createGain(), at = start + k * step;
      osc.type = "sine";
      osc.frequency.value = scale[note];
      g.gain.setValueAtTime(0, at);
      g.gain.linearRampToValueAtTime(0.25, at + 0.005);
      g.gain.exponentialRampToValueAtTime(0.001, at + 1.1);
      osc.connect(g).connect(this.musicBus);
      osc.start(at);
      osc.stop(at + 1.2);
    }
    this.tuneUntil = start + 16 * step;
    this.title = "a music-box tune";
  }
}
