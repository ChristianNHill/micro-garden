// viewer/snapshot.py: what the garden looks like this step, for a viewer to draw. The Python sends it to Godot as JSON
// over UDP; here the page's renderer reads the same object.
import { HEX_AZ, HEX_EL, N_HEX } from "./retina.js";
import * as social from "./social.js";
import { phrase } from "./emotes.js";
import { label_of } from "./personality.js";
import { clip, pressing } from "./physiology.js";
import { DAY_S, daylight } from "./fields.js";

export const MOODS = ["fear", "anger", "joy", "sorrow"];
export const MOOD_AT = 0.25;
export const SHAPE_KNOBS = ["appetite", "aggressiveness", "timidity", "vanity", "chattiness", "energy", "sleepiness"];
export const EMOTE_S = 3.0;
export const EATING_S = 1.0;
export const KICKING_S = 0.4;
export const HEARD_S = 1.0;
export const TOASTS = 6;
export const DN_NAMES = ["forward", "back", "steer L", "steer R", "giant fiber", "feed"];
export const FRUIT_NAMES = ["oranges", "apples", "bananas", "pears", "cherries", "grapes", "strawberries", "lemons", "plums", "peaches"];
const TOAST_FORMATS = [["eaten", "{who} ate"], ["headbutts", "{who} shoved {other}"], ["pets", "{who} was petted"], ["emotes", "{who} {other}"],
                       ["kicks", "{who} kicked the ball"], ["drums", "{who} played the {other}"], ["given", "{who} was handed a fruit"],
                       ["throws", "{who} was thrown"], ["donned", "{who} put a hat on"], ["preened", "{who} shook its hat off"], ["fails", "{who} {other}"],
                       ["binned", "{who} went in the bin"]];
// where each of an eye's 721 columns looks, from -1 to 1 across the eye's field
const WIDE = Math.max(...HEX_AZ.map(Math.abs));
export const HEX = { az: Array.from(HEX_AZ, a => a / WIDE), el: Array.from(HEX_EL, e => e / WIDE) };
// Python's round: correctly rounded from the float's exact value, which toFixed also works from, and a value exactly
// half way goes to the even neighbour. Only a multiple of 1 / 2^(places + 1) can be exactly half way.
const pyRound = (x, places) => {
  if (Number.isInteger(x * 2 ** (places + 1)) && x.toFixed(places + 1).endsWith("5")) {
    const f = Math.floor(x * 10 ** places);
    return (f % 2 === 0 ? f : f + 1) / 10 ** places;
  }
  return Number(x.toFixed(places));
};
const r2 = x => pyRound(x, 2), r3 = x => pyRound(x, 3);

// What duck i is to the others and to the player, in words, for its card.
export function among(body, i, names) {
  const [friend, grudge] = social.friends(body, i, names);
  const trust = body.hand_trust[i];
  return { friend, grudge, favourite: FRUIT_NAMES[body.favourite[i]],
           hand: trust > 0.3 ? "trusts you" : trust < -0.2 ? "is wary of you" : "is still making up its mind about you",
           skills: [body.swim_skill, body.walk_skill, body.dance_skill, body.eat_skill, body.fight_skill, body.fashion_skill, body.music_skill].map(v => r2(v[i])) };
}

// Duck i's needs, moods and wants as [section, name, 0 to 1] rows, for a viewer's bars.
export function readout(body, i) {
  const [hot, cold] = body.discomfort().map(v => v[i]);
  const k = name => body.k[name][i];
  const free = 1.0 - pressing(Math.max(body.hunger[i], body.thirst[i]));
  const rows = [["needs", "hungry", body.hunger[i]], ["needs", "thirsty", body.thirst[i]], ["needs", "sleepy", body.sleep_pressure[i]],
                ["needs", "tired", body.fatigue[i]], ["needs", "bored", body.boredom[i]], ["needs", "too hot", hot], ["needs", "too cold", cold],
                ["moods", "joy", body.joy[i]], ["moods", "fear", body.fear[i]], ["moods", "anger", body.anger[i]], ["moods", "sorrow", body.sorrow[i]],
                ["wants", "a swim", k("water_love") * (1 + hot) / 2 * free], ["wants", "company", k("sociability") * free],
                ["wants", "music", k("music_affinity") * free], ["wants", "a hat", k("vanity")], ["wants", "to play", k("playfulness") * free],
                ["wants", "the stink", k("stink_affinity")]];
  return rows.map(([section, name, v]) => [section, name, r2(clip(v, 0, 1))]);
}

export class Snapshot {
  constructor() {
    this.seen = Object.fromEntries(TOAST_FORMATS.map(([key]) => [key, 0]));
    this.toasts = [];
    this.told = 0;  // news lines ever added, so the view can tell a new line from a repeat of the same words
  }

  _gather_toasts(stub) {
    const short = who => typeof who === "number" ? stub.duck_names[who] : who;
    for (const [key, fmt] of TOAST_FORMATS) {
      const events = stub[key];
      for (const e of events.slice(this.seen[key])) {
        if (key === "emotes" && (e[2] === "dance" || e[2] === "singdance")) continue;
        const other = key === "emotes" ? phrase(e[2]) : e.length > 2 ? short(e[2]) : "";
        const line = fmt.replace("{who}", short(e[1])).replace("{other}", other);
        if (!this.toasts.slice(-3).includes(line)) { this.toasts.push(line); this.told++; }
      }
      this.seen[key] = events.length;
      if (events.length > 400) { events.splice(0, events.length - 200); this.seen[key] = events.length; }  // the garden never needs more
    }
    this.toasts = this.toasts.slice(-TOASTS);
  }

  // `server` is the BrainServer; `watched` the duck whose readout and brain go along.
  build(stub, server, watched = -1) {
    this._gather_toasts(stub);
    const n = stub.n;
    const last = events => { const m = new Map(); for (const e of events.slice(-4 * n)) m.set(e[1], e); return m; };
    const recent = { emotes: last(stub.emotes), bites: last(stub.eaten), kicks: last(stub.kicks), taps: last(stub.drums),
                     posture: stub.posture(), down_left: stub.down_left(), swimming: stub._swimming() };
    const w = stub.world, body = server ? server.body : null;
    const ducks = Array.from({ length: n }, (_, i) => this._duck(stub, body, i, watched, recent));
    return {
      t: r2(stub.t), size: w.size, light: r3(daylight(stub.t)), day: (stub.t % DAY_S) / DAY_S, ducks,
      food: w.food.map(([x, y], k) => [r3(x), r3(y), w.kinds[k]]), danger: w.danger.map(d => [...d]),
      pond: w.pond, tree: w.tree, rocks: w.rocks, wind: w.wind,
      hats: stub.hat_items.map(([x, y, k]) => [r3(x), r3(y), k]),
      balls: w.balls.map(([x, y], k) => [r3(x), r3(y), stub.ball_styles[k]]),
      drum: w.drum, instruments: w.instruments.map(([x, y, k]) => [r3(x), r3(y), k]),
      held: stub.held, music: w.music, music_volume: r2(w.music_volume), hand: w.hand, toasts: this.toasts, told: this.told,
      sounds: stub.sounds.slice(-2 * n).filter(([t]) => stub.t - t < HEARD_S),
    };
  }

  _duck(stub, body, i, watched, { emotes, bites, kicks, taps, posture, down_left, swimming }) {
    const emote = emotes.get(i);
    const showing = emote !== undefined && stub.t - emote[0] < EMOTE_S;
    const recent = (m, s) => m.has(i) && stub.t - m.get(i)[0] < s;
    const duck = {
      name: stub.duck_names[i], x: stub.pose[i][0], y: stub.pose[i][1], h: stub.pose[i][2],
      sat: posture[i] === "sat", down: posture[i] === "down", down_left: down_left[i], swimming: swimming[i],
      hat: stub.hats[i], hat_style: Math.max(stub.hat_style[i], 0), head: stub.head[i],
      eating: recent(bites, EATING_S), kicking: recent(kicks, KICKING_S), drumming: recent(taps, 0.5),
      emote: showing ? emote[2] : "", emote_t: showing ? emote[0] : -1.0, crying: stub.crying_until[i] > stub.t,
      label: "", asleep: false, mood: "", strength: 0.0, hunger: 0, thirst: 0, sleepy: 0, knobs: {}, readout: [], among: {},
    };
    if (body === null) return duck;
    let mood = MOODS[0], strength = body[MOODS[0]][i];
    for (const m of MOODS) if (body[m][i] > strength) { mood = m; strength = body[m][i]; }
    if (strength < MOOD_AT) mood = "content";
    Object.assign(duck, {
      asleep: body.asleep[i], mood, strength: r2(strength), tune: r2(body.music_skill[i]),
      hunger: r2(body.hunger[i]), thirst: r2(body.thirst[i]), sleepy: r2(body.sleep_pressure[i]),
      label: label_of(body.k, i), knobs: Object.fromEntries(SHAPE_KNOBS.map(k => [k, r2(body.k[k][i])])),
    });
    if (i === watched) Object.assign(duck, { readout: readout(body, i), among: among(body, i, stub.duck_names) });
    return duck;
  }

  // The ride view's overlay: both retinas (left eye first) and the six descending readouts in Hz.
  ride(stub, server, duck) {
    return { duck, lum: stub.seen ? stub.seen[duck] : new Float32Array(2 * N_HEX).fill(0.5),
             dn: DN_NAMES.map((name, g) => [name, r2(server.decoder.rates[duck][g])]) };
  }
}
