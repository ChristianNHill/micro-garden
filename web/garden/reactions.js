// brain/reactions.py: chains of reactions, after Bailey and Katchabaw's Actor/Reactor system. A stimulus is a kind,
// the duck that did it, the duck it was done to, and a heat; every duck near enough may respond, and a response is
// itself a stimulus one link further down the chain, cooler for it.
import { clip, pressing } from "./physiology.js";

export const CHAIN_KEEP = 0.5;
export const CROWD = 0.3;
export const IGNORE = 1.5;
export const NOTICE = 0.5;
export const SCENT_HALF = 0.05;
export const SCENT_CONTRAST = 8.0;
export const INTENT_S = 20.0;
export const ARRIVE = { comfort: 0.6, shove: 0.7 };
export const MOOD_STEP = 0.25;

export const stimulus = (kind, actor, target = -1, heat = 1.0) => ({ kind, actor, target, heat });

const HURT = { shove: "target", laugh: "target", fall: "actor", sad: "actor" };

// Each response duck w could make to a stimulus, before heat, busyness and the crowd.
export function weights(kind, body, w, actor, target) {
  const k = name => body.k[name][w];
  const tie = x => (x >= 0 ? body.bond[w][x] : 0.0);
  const friend = x => Math.max(tie(x), 0.0), grudge = x => Math.max(-tie(x), 0.0);
  const dares = x => 1.0 - Math.max(body.k.aggressiveness[x] - k("aggressiveness"), 0.0);
  if (kind === "shove" && w === target)
    return { retaliate: k("aggressiveness") * dares(actor) * (0.3 + grudge(actor)), flee: k("timidity") * 0.6 };
  if (kind === "shove")
    return { comfort: k("kindness") * (0.3 + friend(target)),
             defend: k("aggressiveness") * (0.2 + friend(target)) * (1 + grudge(actor)) * dares(actor),
             join: k("aggressiveness") * (1 - k("kindness")) * (0.1 + friend(actor)) * (0.2 + grudge(target)),
             flee: k("timidity") * (0.3 + grudge(actor)) };
  if (kind === "fall") return { comfort: k("kindness") * (0.2 + friend(actor)) };
  if (kind === "laugh")
    return { laugh: (1 - k("kindness")) * k("playfulness") * (0.2 + grudge(target)),
             defend: k("aggressiveness") * friend(target) * dares(actor),
             comfort: k("kindness") * (0.2 + friend(target)) };
  if (kind === "comfort") return { cheer: 0.5 * k("sociability") * (0.5 + friend(actor)) };
  if (kind === "angry") {
    const mine = k("aggressiveness"), theirs = body.k.aggressiveness[actor];
    return { anger: mine * (mine >= theirs - 0.1 ? 1.0 : 0.2), sadden: k("kindness") * (1 - mine),
             fear: (1 - k("kindness")) * (1 - mine) * (0.5 + k("timidity")) };
  }
  if (kind === "sad") return { comfort: k("kindness") * (0.2 + friend(actor)) * (0.5 + k("sociability")) };
  return {};
}

export class Reactions {
  constructor(n, rng) {
    this.rng = rng;
    this.n = n;
    this.intent = new Array(n).fill(null);  // [act, target, until, heat]
    this.heat_of = new Map();  // "actor,target" of a shove sent as a response -> its heat
    this.later = [];
  }

  busy(body, w, light) {
    if (body.asleep[w] || this.intent[w] !== null) return 0.0;
    const need = pressing(Math.max(body.hunger[w], body.thirst[w]));
    return (1 - 0.7 * need) * (1 - 0.5 * body.fatigue[w]) * (0.4 + 0.6 * light);
  }

  caught_up(duck) { return this.intent.filter(it => it !== null && it[1] === duck).length; }

  // `near[w][x]` is how plainly w smells x, 0 to 1. Returns per duck what to act out now.
  react(body, stimuli, near, light, t) {
    const acts = Array.from({ length: this.n }, () => []);
    const pending = this.later;
    this.later = [];
    for (const s of [...stimuli, ...pending]) {
      const hurt = s.kind in HURT ? s[HURT[s.kind]] : -1;
      for (let w = 0; w < this.n; w++) {
        if (w === s.actor) continue;
        const noticed = w === s.target || near[w][s.actor] > NOTICE || (s.target >= 0 && near[w][s.target] > NOTICE);
        const free = noticed ? this.busy(body, w, light[w]) : 0.0;
        if (free === 0.0) continue;
        const crowd = CROWD ** this.caught_up(hurt >= 0 ? hurt : s.actor);
        const ws = Object.entries(weights(s.kind, body, w, s.actor, s.target)).filter(([, v]) => v > 0)
          .map(([r, v]) => [r, v * s.heat * free * crowd]);
        let roll = this.rng.random() * (IGNORE + ws.reduce((a, [, v]) => a + v, 0));
        for (const [response, v] of ws) {
          if (roll < v) { this._respond(body, response, w, s, hurt, t, acts); break; }
          roll -= v;
        }
      }
    }
    return acts;
  }

  _respond(body, response, w, s, hurt, t, acts) {
    const heat = s.heat * CHAIN_KEEP;
    if (response === "retaliate" || response === "defend") this.intent[w] = ["shove", s.actor, t + INTENT_S, heat];
    else if (response === "join") this.intent[w] = ["shove", s.target, t + INTENT_S, heat];
    else if (response === "comfort" && hurt >= 0) this.intent[w] = ["comfort", hurt, t + INTENT_S, heat];
    else if (response === "flee") body.fear[w] = Math.min(body.fear[w] + MOOD_STEP, 1.0);
    else if (response === "laugh") {
      acts[w].push("emote_laugh");
      body.joy[w] = Math.min(body.joy[w] + 0.15, 1.0);
      body.sorrow[s.target] = Math.min(body.sorrow[s.target] + 0.15, 1.0);
      body.bond[s.target][w] -= 0.1;
      this.later.push(stimulus("laugh", w, s.target, heat));
    } else if (response === "cheer") {
      body.joy[w] = Math.min(body.joy[w] + MOOD_STEP, 1.0);
      body.bond[w][s.actor] += 0.05;
    } else if (response === "anger" || response === "sadden" || response === "fear") {
      const mood = body[{ anger: "anger", sadden: "sorrow", fear: "fear" }[response]];
      mood[w] = Math.min(mood[w] + MOOD_STEP, 1.0);
    }
  }

  // Walk each duck with an intent towards its duck and act on arrival: [turn, want, acts].
  steer(body, f, left, right, t) {
    const turn = new Array(this.n).fill(0), want = new Array(this.n).fill(0);
    const acts = Array.from({ length: this.n }, () => []);
    for (let w = 0; w < this.n; w++) {
      const it = this.intent[w];
      if (it === null) continue;
      const [act, x, until, heat] = it;
      if (t > until || body.asleep[w] || body.asleep[x] || pressing(Math.max(body.hunger[w], body.thirst[w])) > 0.9) {
        this.intent[w] = null;
        continue;
      }
      const beside = f.near_left[w] + f.near_right[w];
      if (Math.trunc(f.near_id[w]) === x && beside >= ARRIVE[act]) {
        this.intent[w] = null;
        if (act === "comfort") {
          acts[w].push("emote_comfort");
          body.sorrow[x] = Math.max(body.sorrow[x] - 0.3, 0.0);
          body.fear[x] = Math.max(body.fear[x] - 0.2, 0.0);
          body.bond[x][w] += 0.2;
          body.bond[w][x] += 0.1;
          this.later.push(stimulus("comfort", w, x, heat));
        } else {
          acts[w].push("strike");
          this.heat_of.set(`${w},${x}`, heat);
        }
        continue;
      }
      const total = left[w][x] + right[w][x];
      const side = (left[w][x] - right[w][x]) / Math.max(total, 1e-9);
      turn[w] = total > 0 ? clip(side * SCENT_CONTRAST * (total / (total + SCENT_HALF)), -1, 1) : 0.0;
      want[w] = 1.0;
    }
    return [turn, want, acts];
  }
}
