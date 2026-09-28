// brain/emotes.py: explicit code, not the fly brain, that lets a duck show what it is feeling now and then.
import { clip } from "./physiology.js";

export const EVERY_S = 5.0;
export const CHANCE = 0.35;
export const FELT_AT = 0.4;
export const SIGNATURE = 0.6;
export const DOES = { laugh: "laughs", comfort: "comforts a friend", cry: "cries", dance: "dances", singdance: "sings and dances",
                      stomp: "stomps", yawn: "yawns", splash: "splashes", sing: "sings", cower: "cowers" };
export const AMUSING = ["sing", "dance", "singdance", "playful"];

export const phrase = emote => DOES[emote] ?? `looks ${emote}`;

// How strongly duck i feels each thing worth showing, 0 to 1.
export function feelings(body, i) {
  const k = name => body.k[name][i];
  const v = name => +body[name][i];
  const joy = v("joy");
  const trait = name => clip((k(name) - SIGNATURE) / (1 - SIGNATURE), 0, 1);
  return {
    happy: joy * (1 - 0.5 * k("playfulness")),
    playful: joy * k("playfulness") + 0.5 * v("boredom") * k("playfulness") * k("energy"),
    scared: v("fear") * (0.5 + k("timidity")),
    angry: v("anger"),
    sad: v("sorrow") * (1 - k("sociability")),
    lonely: v("sorrow") * k("sociability"),
    bored: v("boredom") * (0.5 + k("boredom_rate")) * (1 - k("playfulness") * k("energy")),
    hungry: Math.max(v("hunger") - 0.5, 0) * 2 * (0.5 + k("appetite")),
    thirsty: Math.max(v("thirst") - 0.5, 0) * 2,
    sleepy: Math.max(v("sleep_pressure") - 0.5, 0) * 2,
    curious: 0.6 * k("curiosity") * (1 - Math.max(v("hunger"), v("thirst"))),
    proud: 0.7 * k("vanity") * (1 - Math.max(v("hunger"), v("thirst"))),
    stomp: trait("aggressiveness") * Math.max(v("anger"), 0.6 * Math.max(v("hunger") - 0.4, 0) / 0.6),
    yawn: trait("sleepiness") * Math.max(v("sleep_pressure"), 0.45),
    splash: trait("water_love") * +body.swimming[i],
    sing: trait("chattiness") * (1 - Math.max(v("hunger"), v("thirst"))) * (1 - v("fear")),
    cower: trait("timidity") * v("fear") * 1.5,
    cry: Math.max(v("sorrow"), Math.max(v("hunger") - 0.85, 0) / 0.15) * 0.9,
    dance: 0.0, singdance: 0.0,
    laugh: 0.0, comfort: 0.0,
  };
}

// The emote duck i shows now, if any. Called every EVERY_S, not every tick.
export function pick(body, i, rng) {
  if (body.asleep[i]) return null;
  const felt = Object.entries(feelings(body, i)).filter(([, s]) => s >= FELT_AT);
  if (!felt.length) return null;
  const expressive = 0.4 + 0.6 * body.k.chattiness[i];
  const strengths = felt.map(([, s]) => clip(s, 0, 1));
  if (rng.random() > CHANCE * expressive * Math.max(...strengths)) return null;
  const total = strengths.reduce((a, s) => a + s, 0);
  return rng.choice(felt.map(([name]) => name), strengths.map(s => s / total));
}
