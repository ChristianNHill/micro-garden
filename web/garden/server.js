// brain/server.py: senses in, microduck intents out, one brain per body. Each 20 ms body step: the frames -> levels ->
// two brain ticks in each duck's worker -> decode -> robot calls on the body. The brains run on their own threads, so
// `step` is async; everything else is the Python's step, in its order.
import { CELL_M, DECAY, DIFFUSION } from "./fields.js";
import { Physiology, aggression_tone, clip, pressing } from "./physiology.js";
import { Decoder } from "./decoder.js";
import * as social from "./social.js";
import * as reactions from "./reactions.js";
import * as emotes from "./emotes.js";
import { Rng } from "./rng.js";

export const BODY_DT_MS = 20.0;
export const ODOR_HALF = 0.5;
export const DANGER_HALF = 1.5;
export const ODOR_CONTRAST = 8.0;
export const PLUME_CONTRAST = ODOR_CONTRAST * (Math.sqrt(DIFFUSION / DECAY) * CELL_M) / 0.625;
export const DUCK_HALF = 0.3;
export const HUMID_HALF = 0.1;
export const DRY_LEVEL = 0.2;
export const TEMP_COMFORT_C = 25.0, TEMP_SPAN_C = 5.0, TEMP_LEVEL = 0.5;
export const TOUCH_LEVEL = 0.2;
export const PET_LEVEL = 0.6;
export const VOICE_COOLDOWN_S = 3.0;
export const SCARE_LEVEL = 0.8;
export const WEAR_P = [0.1, 0.95];
export const HAT_SHY_S = 30.0;
export const DANCE_EVERY_S = 3.0;
export const DANCE_REST_S = [10.0, 25.0];
export const PERFORM_BORED = 0.5;
export const DANCE_LOUD = 0.1;
export const DRUM_EVERY_S = 0.6;
export const KICK_EVERY_S = 1.0;
export const CLAP_S = 0.2;
export const ANTENNA_OUT = Math.PI / 4;
export const MUSIC_HALF = 0.4;
export const HAT_LEVEL = 0.5;
const DT = BODY_DT_MS / 1000;
const b = x => (x ? 1 : 0);

// Per-side input levels for one duck: saturating overall level, left/right difference amplified by contrast.
export function bilateral(left, right, half, contrast = ODOR_CONTRAST) {
  const mean = (left + right) / 2;
  const d = contrast * (left - right) / Math.max(left + right, 1e-9);
  const level = mean / (mean + half);
  return [level * clip(1 + d, 0, 2), level * clip(1 - d, 0, 2)];
}

// Frames -> encoder levels, one array a key.
export function sense_levels(f) {
  const n = f.odor_left.length, lv = {};
  const each = fn => Array.from({ length: n }, (_, i) => fn(i));
  for (const [name, key, half, contrast] of [["orn_food", "odor", ODOR_HALF, PLUME_CONTRAST], ["orn_danger", "danger", DANGER_HALF, PLUME_CONTRAST],
                                             ["moist_air", "humidity", HUMID_HALF, ODOR_CONTRAST], ["orn_pheromone", "duck", DUCK_HALF, ODOR_CONTRAST]]) {
    const pairs = each(i => bilateral(f[`${key}_left`][i], f[`${key}_right`][i], half, contrast));
    lv[`${name}_left`] = pairs.map(p => p[0]);
    lv[`${name}_right`] = pairs.map(p => p[1]);
  }
  for (const side of ["left", "right"]) {
    const temp = f[`temp_${side}`];
    lv[`heat_${side}`] = each(i => TEMP_LEVEL * clip((temp[i] - TEMP_COMFORT_C) / TEMP_SPAN_C, 0, 1));
    lv[`cold_${side}`] = each(i => TEMP_LEVEL * clip((TEMP_COMFORT_C - temp[i]) / TEMP_SPAN_C, 0, 1));
    lv[`dry_air_${side}`] = each(i => DRY_LEVEL * (1 - f[`humidity_${side}`][i]));
    lv[`bristle_${side}`] = each(i => Math.max(Math.max(TOUCH_LEVEL * Math.min(f[`touch_${side}`][i], 1), PET_LEVEL * f.petted[i]), HAT_LEVEL * f.hat[i]));
  }
  for (const [side, out] of [["left", ANTENNA_OUT], ["right", -ANTENNA_OUT]]) {
    const along = each(i => f.wind[i] * Math.cos(f.wind_from[i] - out));
    lv[`jo_push_${side}`] = along.map(a => Math.max(a, 0));
    lv[`jo_pull_${side}`] = along.map(a => Math.max(-a, 0));
  }
  lv.johnstons_organ = each(i => { const m = (f.music_left[i] + f.music_right[i]) / 2; return m / (m + MUSIC_HALF); });
  lv.LPLC2 = each(i => SCARE_LEVEL * f.scared[i]);
  lv.sugar = [...f.sugar];
  lv.water_taste = [...f.water];
  return lv;
}

// BrainServer._levels: the levels shaped by the body. clap_left is updated in place; returns {levels, following}.
export function levels_for(body, f, clap_left) {
  const n = body.n, levels = sense_levels(f);
  for (let i = 0; i < n; i++) clap_left[i] = f.scared[i] > 0 ? CLAP_S : Math.max(clap_left[i] - DT, 0);
  levels.LPLC2 = clap_left.map(c => SCARE_LEVEL * b(c > 0));
  const wet = f.swimming.map(s => s > 0);
  for (const s of ["left", "right"]) {
    levels[`moist_air_${s}`] = levels[`moist_air_${s}`].map((v, i) => wet[i] ? 1.0 : v);
    levels[`bristle_${s}`] = levels[`bristle_${s}`].map((v, i) => wet[i] ? TOUCH_LEVEL : v);
  }
  const gains = body.sense_gains();
  for (const key of Object.keys(levels)) {
    const base = key.replace(/_left$/, "").replace(/_right$/, "");
    if (gains[base]) levels[key] = levels[key].map((v, i) => v * gains[base][i]);
  }
  const following = [];
  const push = [];
  for (let i = 0; i < n; i++) {
    const scent = clip((levels.orn_food_left[i] + levels.orn_food_right[i]) / 2, 0, 1) * pressing(body.hunger[i]);
    const damp = (clip((levels.moist_air_left[i] + levels.moist_air_right[i]) / 2, 0, 1)
                  * (1 - body.at_water((f.humidity_left[i] + f.humidity_right[i]) / 2)) * b(!wet[i]));
    following.push(Math.max(scent, damp) * b(f.wind[i] > 0));
    push.push(Math.max(scent, damp));
  }
  for (const key of ["jo_push_left", "jo_push_right", "jo_pull_left", "jo_pull_right"]) levels[key] = levels[key].map((v, i) => v * push[i]);
  levels.sugar_grn = levels.sugar.map((v, i) => Math.max(v, levels.water_taste[i]));
  delete levels.sugar;
  delete levels.water_taste;
  levels.pC1_aggr = body.hunger.map((h, i) => aggression_tone(body.k.aggressiveness[i], h, (f.odor_left[i] + f.odor_right[i]) / 2,
                                                               body.anger[i], body.k.kindness[i]));
  return { levels, following };
}

export class BrainServer {
  // brains: one per duck, each with an async step(message) (web/worker.js through a proxy); sizes: decoder group sizes
  constructor({ stub, brains, sizes, seed = 0, personality = {}, hunger = 0.5, thirst = 0.5, learns = true, eyes = true }) {
    const n = this.n = stub.n;
    this.stub = stub;
    this.brains = brains;
    this.learns = learns;
    this.eyes = eyes;
    this.body = new Physiology(n, personality, hunger, thirst);
    this.decoder = new Decoder(sizes, n, new Rng(seed), this.body.k.stink_affinity);
    this.voice_rng = new Rng(seed + 1);
    this.last_vx = new Array(n).fill(0);
    this.escaped = new Array(n).fill(false);
    this.quiet_until = new Array(n).fill(0);
    this.clap_left = new Array(n).fill(0);
    this.possessed = new Array(n).fill(false);
    this.zooming = new Array(n).fill(false);
    this.emote_rng = new Rng(seed + 7);
    this.slip_rng = new Rng(seed + 13);
    this.mishap = new Array(n).fill(null);
    this.laughs = new Array(n).fill(false);
    this.squabble = new Array(n).fill(false);
    this.reactions = new reactions.Reactions(n, new Rng(seed + 17));
    this.shown = [];
    this.react_acts = Array.from({ length: n }, () => []);
    social.init(this.body, seed);
    this.hat_was_near = new Array(n).fill(false);
    this.kick_at = new Array(n).fill(0);
    this.drum_at = new Array(n).fill(0);
    this.dance_at = new Array(n).fill(0);
    this.dancing = new Array(n).fill(0);
    this.performing = new Array(n).fill(0);
    this.playing = new Array(n).fill(0);
    this.following = new Array(n).fill(0);
    this.fondness = new Array(n).fill(0);
    this.hat_shy_until = new Array(n).fill(0);
    this.emote_at = this.emote_rng.uniforms(n, 0, emotes.EVERY_S);
    this.watched = -1;  // the duck whose brain the viewer shows
    this.view = new Set();  // shown neurons that fired since the viewer last took them
    this.t = 0.0;
  }

  // One body step on the frames the body just sent.
  async step(f) {
    const body = this.body, n = this.n;
    this.t += DT;
    const [falls_asleep, wakes] = body.step(DT, f, this.escaped, this.last_vx);
    this.laughs = social.update(body, f, DT, this.last_vx, this.slip_rng);
    this.squabble = social.squabbles(body, f, DT, this.slip_rng);
    this.mishap = social.mishaps(body, f, DT, this.last_vx, this.slip_rng);
    const left = f.scent_left, right = f.scent_right;
    const among = social.steering(body, f, [left, right]);
    this._react(body, f, left, right, among);

    const { levels, following } = levels_for(body, f, this.clap_left);
    this.following = following;
    this.decoder.body = this._decoder_input(f, levels, among);
    const gains = body.sense_gains();
    const reward = f.ate.map((a, i) => this.learns ? Math.max(a, f.petted[i]) : 0);
    const punish = f.scared.map((s, i) => this.learns ? Math.max(s, +this.escaped[i]) : 0);
    this.escaped.fill(false);
    const results = await Promise.all(this.brains.map((brain, i) => brain.step({
      levels: Object.fromEntries(Object.entries(levels).map(([k, v]) => [k, v[i]])),
      lum: this.eyes && f.lum ? f.lum[i] : null, gain: gains.vision[i], reward: reward[i], punish: punish[i], watch: i === this.watched,
    })));
    let intents;
    for (let tick = 0; tick < results[0].counts.length; tick++) {
      intents = this.decoder.update(results.map(r => r.counts[tick]));
      intents.forEach((it, i) => { this.escaped[i] = this.escaped[i] || it.escape; });
    }
    this.fondness = results.map(r => r.fondness);
    if (this.watched >= 0 && results[this.watched].view) for (const k of results[this.watched].view) this.view.add(k);
    this.last_vx = intents.map(it => it.vx);
    for (let i = 0; i < n; i++) {
      if (this.possessed[i]) continue;
      this._send(i, intents[i], f, i, falls_asleep[i], wakes[i]);
    }
    return intents;
  }

  // The shown neurons of the watched duck that fired since the last take, as positions in the brain view.
  take_view() {
    const out = [...this.view];
    this.view.clear();
    return out;
  }

  _react(body, f, left, right, among) {
    const R = reactions.stimulus;
    const stimuli = this.shown;
    this.shown = [];
    for (let j = 0; j < this.n; j++) {
      let by = Math.trunc(f.bumped_by[j]);
      if (by >= 0) {
        const key = `${by},${j}`;
        stimuli.push(R("shove", by, j, this.reactions.heat_of.has(key) ? this.reactions.heat_of.get(key) : 1.0));
        this.reactions.heat_of.delete(key);
      }
      by = Math.trunc(f.comforted_by[j]);
      if (by >= 0) stimuli.push(R("comfort", by, j));
    }
    this.mishap.forEach((m, i) => { if (m === "trip" || m === "flounder" || m === "whiff") stimuli.push(R("fall", i)); });
    this.laughs.forEach((l, i) => { if (l) stimuli.push(R("laugh", i, Math.trunc(f.saw_fall_by[i]))); });
    const near = left.map((row, w) => row.map((l, x) => { const t = l + right[w][x]; return t / (t + reactions.SCENT_HALF); }));
    const acts = this.reactions.react(body, stimuli, near, f.light, this.t);
    const [turn, want, arrived] = this.reactions.steer(body, f, left, right, this.t);
    among.intent_turn = turn;
    among.intent = want;
    among.social_want = among.social_want.map((v, i) => Math.max(v, want[i]));
    this.react_acts = acts.map((a, i) => [...a, ...arrived[i]]);
  }

  _decoder_input(f, levels, among) {
    const body = this.body, k = body.k, n = this.n;
    const at_ease = body.hunger.map((h, i) => 1 - pressing(Math.max(h, body.thirst[i])));
    const tune = at_ease.map((e, i) => Math.abs(2 * k.music_affinity[i] - 1) * levels.johnstons_organ[i] * e);
    this.dancing = at_ease.map((e, i) => clip(2 * k.music_affinity[i] - 1, 0, 1) * clip(levels.johnstons_organ[i] / DANCE_LOUD, 0, 1) * e);
    const bored = at_ease.map((e, i) => clip((body.boredom[i] - PERFORM_BORED) / (1 - PERFORM_BORED), 0, 1) * e);
    this.performing = this.dancing.map((d, i) => Math.max(d, 0.6 * bored[i] * Math.max(k.chattiness[i], k.playfulness[i])));
    const play = body.play();
    const toy_left = [], toy_right = [], ball = [];
    this.playing = [];
    for (let i = 0; i < n; i++) {
      const instrument = clip(3 * Math.max(f.drum_left[i], f.drum_right[i]), 0, 1);
      const musical = clip(2 * k.music_affinity[i] - 1, 0, 1) * instrument * (0.4 + 0.6 * body.boredom[i]);
      this.playing.push(Math.max(play[i], musical) * at_ease[i]);
      toy_left.push(Math.max(f.ball_left[i], f.drum_left[i]));
      toy_right.push(Math.max(f.ball_right[i], f.drum_right[i]));
      ball.push(this.playing[i] * Math.max(toy_left[i], toy_right[i]));
    }
    const [to_pond, to_shade] = body.cooling();
    const cool_left = to_pond.map((p, i) => p * f.pond_left[i] + to_shade[i] * f.shade_left[i]);
    const cool_right = to_pond.map((p, i) => p * f.pond_right[i] + to_shade[i] * f.shade_right[i]);
    const social_want = among.social_want;
    delete among.social_want;
    const wants = tune.map((t, i) => Math.max(t, ball[i], social_want[i] * at_ease[i], to_pond[i] + to_shade[i]));
    const music = f.music_left.map((m, i) => bilateral(m, f.music_right[i], MUSIC_HALF));
    const ducks = f.duck_left.map((d, i) => bilateral(d, f.duck_right[i], DUCK_HALF));
    return {
      ...body.motor(wants, f.humidity_left.map((h, i) => (h + f.humidity_right[i]) / 2)), ...among,
      cool_left, cool_right, play: this.playing, ball_left: toy_left, ball_right: toy_right,
      surge: this.following, swimming: f.swimming.map(s => s > 0), at_shore: f.water.map(w => w > 0),
      thirst: [...body.thirst], hatted: f.hat.map(h => h > 0), fear: [...body.fear],
      hunger: body.hunger.map(h => pressing(h)),
      tasting: f.sugar.map((s, i) => s > 0 || f.water[i] > 0),
      music_left: music.map((m, i) => at_ease[i] * m[0]), music_right: music.map((m, i) => at_ease[i] * m[1]),
      duck_left: ducks.map(d => d[0]), duck_right: ducks.map(d => d[1]),
      at_ease, sociability: k.sociability, fondness: this.learns ? this.fondness : 0.0,
      music_affinity: k.music_affinity, vanity: k.vanity,
    };
  }

  _send(i, it, f, _, falls_asleep, wakes) {
    const send = (method, params) => this.stub.robot(i, method, params);
    const fi = name => f[name][i];
    if (falls_asleep) send("robot.relax");
    if (wakes) send("robot.init");
    send("robot.move", { vx: it.vx, vy: it.vy, vyaw: it.vyaw });
    if (it.feed) send("robot.do", { skill: fi("water") > 0 ? "drink" : "ground_pick" });
    if (it.attack || this.squabble[i]) send("robot.do", { skill: social.strike(this.body, i, this.slip_rng) });
    if (this.mishap[i]) send("robot.do", { skill: this.mishap[i] });
    if (this.laughs[i]) send("robot.do", { skill: "emote_laugh" });
    for (const skill of this.react_acts[i]) send("robot.do", { skill: skill === "strike" ? social.strike(this.body, i, this.slip_rng) : skill });
    if (it.preen) { send("robot.do", { skill: "preen" }); this.hat_shy_until[i] = this.t + HAT_SHY_S; }
    if (!this.body.asleep[i]) {
      this._toy(send, i, fi("ball_near") > 0, this.kick_at, KICK_EVERY_S, "kick");
      this._perform(send, i);
      this._toy(send, i, fi("drum_near") > 0, this.drum_at, DRUM_EVERY_S, "drum");
      this._consider_hat(send, i, f);
    }
    this.hat_was_near[i] = fi("hat_near") > 0;
    if (it.zoomies && !this.zooming[i]) send("robot.do", { skill: "zoomies" });
    this.zooming[i] = it.zoomies;
    this._emote(send, i);
    const tag = this._voice(i, f, falls_asleep, wakes);
    if (tag) send("robot.sound", { tag });
  }

  _toy(send, i, near, at, every_s, skill) {
    if (near && this.t >= at[i]) {
      at[i] = this.t + every_s;
      if (this.emote_rng.random() < this.playing[i]) send("robot.do", { skill: skill === "drum" ? social.plays(this.body, i, this.slip_rng) : skill });
    }
  }

  _perform(send, i) {
    if (this.t < this.dance_at[i]) return;
    this.dance_at[i] = this.t + DANCE_EVERY_S;
    if (this.emote_rng.random() >= this.performing[i] * (0.6 + 0.8 * this.body.dance_skill[i])) return;
    const k = this.body.k;
    const sings = this.emote_rng.random() < 0.3 + 0.6 * k.chattiness[i];
    const dances = this.emote_rng.random() < 0.3 + 0.6 * Math.max(k.playfulness[i], this.dancing[i]) || !sings;
    send("robot.do", { skill: "emote_" + (sings && dances ? "singdance" : sings ? "sing" : "dance") });
    if (dances && social.falls_dancing(this.body, i, this.slip_rng)) send("robot.do", { skill: "trip" });
    this.body.amuse(i);
    social.performed(this.body, i);
    this.dance_at[i] += this.emote_rng.uniform(...DANCE_REST_S);
  }

  _consider_hat(send, i, f) {
    const comes_upon = f.hat_near[i] > 0 && !this.hat_was_near[i];
    if (comes_upon && f.hat[i] === 0 && this.t >= this.hat_shy_until[i]) {
      if (this.emote_rng.random() < WEAR_P[0] + (WEAR_P[1] - WEAR_P[0]) * this.body.k.vanity[i]) send("robot.do", { skill: "wear" });
    }
  }

  _emote(send, i) {
    if (this.t < this.emote_at[i]) return;
    this.emote_at[i] = this.t + emotes.EVERY_S;
    const emote = emotes.pick(this.body, i, this.emote_rng);
    if (emote) {
      send("robot.do", { skill: `emote_${emote}` });
      const shown = { angry: "angry", stomp: "angry", sad: "sad", cry: "sad", lonely: "sad" }[emote];
      if (shown) this.shown.push(reactions.stimulus(shown, i));
      if (emotes.AMUSING.includes(emote)) this.body.amuse(i);
    }
  }

  _voice(i, f, falls_asleep, wakes) {
    if (this.t < this.quiet_until[i]) return null;
    const tag = this._pick_voice(i, f, falls_asleep, wakes);
    if (tag) this.quiet_until[i] = this.t + VOICE_COOLDOWN_S;
    return tag;
  }

  _pick_voice(i, f, falls_asleep, wakes) {
    const bd = this.body;
    const events = [[this.escaped[i], "alarm"], [wakes, "greet"], [falls_asleep, "coo"],
                    [f.ate[i] > 0 || f.drank[i] > 0, "chirp"], [f.bumped[i] > 0, "alarm"]];
    let tag = (events.find(([happened]) => happened) || [null, null])[1];
    if (tag === null && !bd.asleep[i]) {
      if (this.voice_rng.random() < bd.k.chattiness[i] * BODY_DT_MS / 1000 / 60 * (1 + 3 * bd.boredom[i])) {
        tag = bd.sorrow[i] > 0.5 ? "peck" : bd.boredom[i] > 0.5 ? "inquire" : "chirp";
      }
      return tag;
    }
    return this.voice_rng.random() < 0.2 + 0.8 * bd.k.chattiness[i] ? tag : null;
  }
}
