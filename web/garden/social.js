// brain/social.py: what the ducks are to each other and to the player's hand. Explicit code, none of it the fly's.
// Everything lives on the Physiology so a save keeps it: bond[i][j] (-1 a grudge to 1 a friend), hand_trust,
// favourite, and the skills.
import { FOOD_NEAR_HALF, clip, pressing } from "./physiology.js";
import { FRUITS } from "./fields.js";
import { Rng } from "./rng.js";

export const BOND_FADES_S = 1800.0;
export const TOGETHER_S = 120.0;
export const SCENT_HALF = 0.05;
export const SCENT_CONTRAST = 8.0;
export const KINDLY = 0.5;
export const PRACTICE_S = 1800.0;
export const PRACTICE_SEED = 0.15;
export const WALKING_MS = 0.03;
export const BITE_PRACTICE = 0.01;
export const WATCHED_DANCE = 0.04, DANCED = 0.02;
export const RUST_S = 14400.0;
export const KEPT = 0.3;
export const STARVING = 0.8;
export const TRIP_S = 20.0, RUN_MS = 0.3;
export const FLOUNDER_S = 40.0;
export const FUMBLE_P = 0.08;
export const DANCE_FALL_P = 0.3;
export const WHIFF_P = 0.35;
export const SHOVE_PRACTICE = 0.03;
export const HARD_P = 0.8;
export const HAT_SLIP_S = 90.0;
export const DAPPER = 0.02;
export const NOTE_PRACTICE = 0.02;
export const SOUR_P = 0.3;
export const WELL_PLAYED = 0.15;
export const LAUGHED_AT = 0.3;
export const LAUGH_P = [0.05, 0.8];
export const SQUABBLE_S = 20.0;
export const SQUABBLE_FLOOR = 0.1;
export const CARE_FLOOR = 0.15;

const b = x => (x ? 1 : 0);

export function init(body, seed = 0) {
  const n = body.n, rng = new Rng(seed + 11);
  body.bond = Array.from({ length: n }, () => new Array(n).fill(0));
  body.hand_trust = new Array(n).fill(0);
  body.favourite = Array.from({ length: n }, () => rng.integers(FRUITS));
}

// Up the S-curve while doing it, slowly down while not; never below KEPT, or below where it is if under that.
export function practised(skill, doing, step, idle_dt) {
  const faded = Math.max(skill - idle_dt / RUST_S, Math.min(skill, KEPT));
  return clip(doing ? skill + step * (PRACTICE_SEED + skill) * (1 - skill) : faded, 0, 1);
}

// Whether a go at something goes wrong for duck i: `chance` for a duck with none of the skill.
export function fails(body, skill, chance, rng, i) {
  const careless = 0.5 + body.k.carelessness[i];
  return rng.random() < chance * (1 - skill) ** 2 * careless;
}

// What goes wrong for each duck this step, as a skill for the body to act out, or null.
export function mishaps(body, f, dt, speed, rng) {
  const out = [];
  for (let i = 0; i < body.n; i++) {
    const awake = !body.asleep[i];
    const pace = clip(Math.abs(speed[i]) / RUN_MS, 0, 1.5) ** 2;
    // every roll is made, as numpy draws them all
    const trip = fails(body, body.walk_skill[i], dt / TRIP_S * pace, rng, i);
    const flounder = fails(body, body.swim_skill[i], dt / FLOUNDER_S, rng, i);
    const fumble = fails(body, body.eat_skill[i], FUMBLE_P, rng, i);
    const slip = fails(body, body.fashion_skill[i], dt / HAT_SLIP_S, rng, i);
    const tripped = awake && f.swimming[i] === 0 && trip;
    const under = awake && f.swimming[i] > 0 && flounder;
    const fumbled = f.ate[i] > 0 && fumble;
    const slipped = awake && f.hat[i] > 0 && slip;
    body.sorrow[i] = clip(body.sorrow[i] + 0.15 * body.k.timidity[i] * b(tripped || under), 0, 1);
    out.push(tripped ? "trip" : under ? "flounder" : fumbled ? "fumble" : slipped ? "lose_hat" : null);
  }
  return out;
}

// Whether each duck shoves the duck it is pressed against this step.
export function squabbles(body, f, dt, rng) {
  return body.hunger.map((_, i) => {
    const touching = (f.touch_left[i] + f.touch_right[i]) > 0;
    const food = (f.odor_left[i] + f.odor_right[i]) / 2;
    const cause = Math.max(pressing(body.hunger[i]) * food / (food + FOOD_NEAR_HALF), body.anger[i]);
    const temper = SQUABBLE_FLOOR + (1 - SQUABBLE_FLOOR) * body.k.aggressiveness[i];
    return touching && !body.asleep[i] && rng.random() < cause * temper * dt / SQUABBLE_S;
  });
}

export function strike(body, i, rng) {
  if (fails(body, body.fight_skill[i], WHIFF_P, rng, i)) return "whiff";
  return rng.random() < HARD_P * body.fight_skill[i] ? "headbutt_hard" : "headbutt";
}

export function plays(body, i, rng) {
  return fails(body, body.music_skill[i], SOUR_P, rng, i) ? "sour" : "drum";
}

export function falls_dancing(body, i, rng) {
  return fails(body, body.dance_skill[i], DANCE_FALL_P, rng, i);
}

// How far a knob is past `past`, 0 to 1.
export function trait(body, name, past = 0.5) {
  return body.k[name].map(v => clip((v - past) / (1 - past), 0, 1));
}

// One body step of all of it. Returns which ducks laugh at a fall this step.
export function update(body, f, dt, speed, rng) {
  const n = body.n, bond = body.bond;
  const fan = trait(body, "sociability"), put_off = body.k.sociability.map(s => clip(1 - 2 * s, 0, 1));
  const cross = trait(body, "aggressiveness", 0.6), kind = trait(body, "kindness", KINDLY), vain = trait(body, "vanity");
  const who = name => [f[name].map(x => Math.trunc(x)), f[name].map(x => x >= 0)];

  let [by, hit] = who("bumped_by");
  const landed = new Array(n).fill(false);
  for (let i = 0; i < n; i++) if (hit[i]) { bond[i][by[i]] -= 0.3; landed[by[i]] = true; }
  let saw;
  [by, saw] = who("saw_shove_by");
  const [victim] = who("saw_shove_of");
  for (let i = 0; i < n; i++) if (saw[i]) { bond[i][by[i]] -= 0.15 * kind[i]; bond[i][victim[i]] += 0.1 * kind[i]; }
  for (let i = 0; i < n; i++) {
    body.sorrow[i] = clip(body.sorrow[i] + 0.1 * kind[i] * b(saw[i]), 0, 1);
    body.fear[i] = clip(body.fear[i] + 0.25 * body.k.timidity[i] * b(saw[i]), 0, 1);
  }
  let fell;
  [by, fell] = who("saw_fall_by");
  const laughs = new Array(n).fill(false);
  const dice = rng.randoms(n);
  for (let i = 0; i < n; i++) {
    const toward = bond[i][fell[i] ? by[i] : 0];
    let unkind = (1 - body.k.kindness[i]) * Math.max(body.k.playfulness[i], body.k.aggressiveness[i]);
    unkind = clip(unkind * (1 - clip(toward, 0, 1)) * (1 + clip(-toward, 0, 1)), 0, 1);
    laughs[i] = fell[i] && dice[i] < LAUGH_P[0] + (LAUGH_P[1] - LAUGH_P[0]) * unkind;
    body.joy[i] = clip(body.joy[i] + 0.15 * b(laughs[i]), 0, 1);
  }
  for (let laugher = 0; laugher < n; laugher++) {
    if (!laughs[laugher]) continue;
    const v = by[laugher];
    if (rng.random() < 0.2 + 0.6 * body.k.aggressiveness[v]) body.anger[v] = Math.min(body.anger[v] + LAUGHED_AT, 1);
    else body.sorrow[v] = Math.min(body.sorrow[v] + LAUGHED_AT * (0.5 + body.k.timidity[v]), 1);
    bond[v][laugher] -= 0.15;
  }
  let show;
  [by, show] = who("show_by");
  for (let i = 0; i < n; i++) {
    if (show[i]) bond[i][by[i]] += 0.08 * fan[i] * (1 - cross[i]) - 0.05 * put_off[i] - 0.08 * cross[i];
    body.dance_skill[i] = practised(body.dance_skill[i], show[i], WATCHED_DANCE * fan[i] * (1 - cross[i]), dt);
    body.joy[i] = clip(body.joy[i] + (WELL_PLAYED * fan[i] * body.music_skill[show[i] ? by[i] : 0] * b(show[i])), 0, 1);
  }
  let took;
  [by, took] = who("hat_taken_by");
  for (let i = 0; i < n; i++) {
    if (took[i]) bond[i][by[i]] -= 0.2 * vain[i];
    body.anger[i] = clip(body.anger[i] + 0.3 * vain[i] * b(took[i]), 0, 1);
  }
  let held;
  [by, held] = who("comforted_by");
  for (let i = 0; i < n; i++) if (held[i]) { bond[i][by[i]] += 0.25; bond[by[i]][i] += 0.1; }
  for (let i = 0; i < n; i++) body.sorrow[i] = clip(body.sorrow[i] - 0.4 * b(held[i]), 0, 1);
  for (let i = 0; i < n; i++) if (held[i]) body.joy[by[i]] += 0.1;

  const [near, with_one] = who("near_id");
  for (let i = 0; i < n; i++) {
    const at_ease = Math.max(body.hunger[i], body.thirst[i]) < 0.6;
    const both_asleep = body.asleep[i] && body.asleep[with_one[i] ? near[i] : 0];
    const grows = dt / TOGETHER_S * (fan[i] * b(at_ease) + 2.0 * b(both_asleep)) * b(with_one[i]);
    if (with_one[i]) bond[i][near[i]] += grows;
  }
  const keep = Math.exp(-dt / BOND_FADES_S);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) bond[i][j] = i === j ? 0 : clip(bond[i][j] * keep, -1, 1);

  for (let i = 0; i < n; i++) {
    const k = body.k;
    body.fear[i] = clip(body.fear[i] + 0.25 * (0.3 + k.timidity[i]) * b(f.heard_alarm[i] > 0), 0, 1);
    body.joy[i] = clip(body.joy[i] + 0.1 * fan[i] * b(f.heard_joy[i] > 0), 0, 1);
    body.hand_trust[i] = clip(body.hand_trust[i] + 0.08 * b(f.petted[i] > 0) + 0.15 * b(f.hand_fed[i] > 0) - 0.05 * b(f.scared[i] > 0), -1, 1);
    body.joy[i] = clip(body.joy[i] + 0.25 * b(f.ate_kind[i] === body.favourite[i]), 0, 1);
    const hld = f.held[i] > 0, thrown = f.thrown[i] > 0, trusting = body.hand_trust[i] > 0.2;
    body.joy[i] = clip(body.joy[i] + dt * 0.3 * b(hld) * b(trusting), 0, 1);
    body.fear[i] = clip(body.fear[i] + dt * 0.5 * (0.3 + k.timidity[i]) * b(hld) * b(!trusting) + 0.5 * b(thrown), 0, 1);
    body.anger[i] = clip(body.anger[i] + 0.4 * cross[i] * b(thrown), 0, 1);
    body.hand_trust[i] = clip(body.hand_trust[i] + dt * 0.02 * b(hld) * b(trusting) - 0.35 * b(thrown), -1, 1);

    const ate = f.ate[i] > 0;
    body.swim_skill[i] = practised(body.swim_skill[i], f.swimming[i] > 0, dt / PRACTICE_S, dt);
    const walking = Math.abs(speed[i]) > WALKING_MS && f.swimming[i] === 0;
    body.walk_skill[i] = practised(body.walk_skill[i], walking, dt / PRACTICE_S, dt);
    body.eat_skill[i] = practised(body.eat_skill[i], ate, BITE_PRACTICE, dt * b(body.hunger[i] >= STARVING && !ate));
    body.fight_skill[i] = practised(body.fight_skill[i], landed[i], SHOVE_PRACTICE, dt);
    body.music_skill[i] = practised(body.music_skill[i], f.drummed[i] > 0, NOTE_PRACTICE, dt);
    const hatted = f.hat[i] > 0;
    body.fashion_skill[i] = practised(body.fashion_skill[i], hatted, dt / PRACTICE_S, dt);
    body.joy[i] = clip(body.joy[i] + dt * DAPPER * k.vanity[i] * body.fashion_skill[i] * b(hatted), 0, 1);
  }
  return laughs;
}

export function performed(body, i) {
  body.dance_skill[i] = practised(body.dance_skill[i], true, DANCED, 0.0);
}

// What the decoder needs, one value per duck. `scent` is [left, right], each n by n.
export function steering(body, f, scent) {
  const n = body.n;
  const kind = trait(body, "kindness", KINDLY);
  const [left, right] = scent;
  const out = { bond_turn: [], bond_near: [], near_left: [...f.near_left], near_right: [...f.near_right],
                hand_trust: [...body.hand_trust], hand_left: [...f.hand_left], hand_right: [...f.hand_right],
                comfort: [], cry_left: [...f.cry_left], cry_right: [...f.cry_right], sharing: [], social_want: [],
                sleepy_together: [], swim_skill: [...body.swim_skill], walk_skill: [...body.walk_skill] };
  const soc = trait(body, "sociability");
  for (let i = 0; i < n; i++) {
    const carer = CARE_FLOOR + (1 - CARE_FLOOR) * kind[i];
    const crying = Math.max(f.cry_left[i], f.cry_right[i]);
    const hand = Math.max(f.hand_left[i], f.hand_right[i]);
    let turn = 0;
    for (let j = 0; j < n; j++) {
      const total = left[i][j] + right[i][j];
      const side = (left[i][j] - right[i][j]) / Math.max(total, 1e-9);
      turn += body.bond[i][j] * side * (total / (total + SCENT_HALF));
    }
    out.bond_turn.push(clip(turn * SCENT_CONTRAST, -1, 1));
    const with_one = f.near_id[i] >= 0;
    out.bond_near.push(with_one ? body.bond[i][Math.trunc(f.near_id[i])] : 0.0);
    out.comfort.push(carer);
    out.sharing.push(kind[i] > 0.3 && crying > 0.3 && body.hunger[i] < 0.6);
    out.social_want.push(Math.max(carer * crying, clip(body.hand_trust[i], 0, 1) * hand));
    out.sleepy_together.push(soc[i] * clip((body.sleep_pressure[i] - 0.6) / 0.4, 0, 1));
  }
  return out;
}

// Duck i's best friend and worst grudge, by name, or "" where it has neither to speak of.
export function friends(body, i, names) {
  const row = body.bond[i];
  let best = 0, worst = 0;
  row.forEach((v, j) => { if (v > row[best]) best = j; if (v < row[worst]) worst = j; });
  return [row[best] > 0.15 ? names[best] : "", row[worst] < -0.15 ? names[worst] : ""];
}
