// brain/personality.py: personality knobs and label presets. Every knob is a 0-1 scale; unlisted knobs sit at 0.5.

export const KNOBS = [
  "aggressiveness", "stink_affinity", "timidity", "curiosity", "sociability", "kindness", "appetite",
  "energy", "sleepiness", "heat_tolerance", "water_love", "boredom_rate", "chattiness", "carelessness",
  "smarts", "playfulness", "music_affinity", "hoarding", "vanity",
];
export const KNOB_DEFAULT = 0.5;
export const JITTER = 0.1;

export const LABELS = {
  "Gentle": { aggressiveness: 0.05, kindness: 0.9, sociability: 0.7 },
  "Naughty": { aggressiveness: 0.6, kindness: 0.2, curiosity: 0.7, playfulness: 0.8, music_affinity: 0.7 },
  "Energetic": { energy: 0.9, sleepiness: 0.2, playfulness: 0.7, music_affinity: 0.8 },
  "Quiet": { energy: 0.3, chattiness: 0.1, sociability: 0.3, music_affinity: 0.15 },
  "Big eater": { appetite: 0.95, aggressiveness: 0.5 },
  "Chatty": { chattiness: 0.95, sociability: 0.8, music_affinity: 0.9 },
  "Easily bored": { boredom_rate: 0.9, curiosity: 0.6 },
  "Curious": { curiosity: 0.95, timidity: 0.2 },
  "Carefree": { timidity: 0.1, water_love: 0.8, stink_affinity: 0.5, music_affinity: 0.7 },
  "Careless": { carelessness: 0.9, timidity: 0.1 },
  "Smart": { smarts: 0.95 },
  "Cry baby": { timidity: 0.9, aggressiveness: 0.05, curiosity: 0.2, music_affinity: 0.2 },
  "Lonely": { sociability: 0.95 },
  "Naive": { smarts: 0.2, timidity: 0.1, curiosity: 0.7 },
  "No personality": {},
  "Bully": { aggressiveness: 0.95, kindness: 0.1, appetite: 0.9, sociability: 0.8 },
  "Zoomer": { energy: 0.95, boredom_rate: 0.8, playfulness: 0.8, music_affinity: 0.8 },
  "Napper": { sleepiness: 0.95, energy: 0.3, heat_tolerance: 0.3, music_affinity: 0.25 },
  "Show-off": { vanity: 0.9, playfulness: 0.8, sociability: 0.8, music_affinity: 0.9 },
  "Scaredy": { timidity: 0.95, sociability: 0.4, stink_affinity: 0.0, music_affinity: 0.1 },
  "Loner": { sociability: 0.05, curiosity: 0.6, timidity: 0.3, music_affinity: 0.3 },
  "Musician": { music_affinity: 0.95, chattiness: 0.7, playfulness: 0.6, sociability: 0.7 },
  "Stink lover": { stink_affinity: 0.95, curiosity: 0.7, timidity: 0.2 },
};

export const HATCHABLE = Object.keys(LABELS).filter(l => !["Smart", "Careless", "Naive", "No personality"].includes(l));

// Knob values for a label, jittered when rng is given.
export function preset(label, rng = null) {
  const k = {};
  for (const name of KNOBS) {
    const v = LABELS[label][name] ?? KNOB_DEFAULT;
    k[name] = rng ? Math.min(Math.max(v + rng.uniform(-JITTER, JITTER), 0), 1) : v;
  }
  return k;
}

// Per-knob arrays, one value per duck.
export function stack(ducks) {
  return Object.fromEntries(KNOBS.map(name => [name, ducks.map(d => d[name])]));
}

// The label duck i's knobs sit nearest to.
export function label_of(k, i) {
  let best = null, bd = Infinity;
  for (const label of Object.keys(LABELS)) {
    let d = 0;
    for (const name of KNOBS) d += (k[name][i] - (LABELS[label][name] ?? KNOB_DEFAULT)) ** 2;
    if (d < bd) { bd = d; best = label; }
  }
  return best;
}
