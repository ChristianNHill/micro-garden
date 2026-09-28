// brain/physiology.py: bodily state per duck, drives and Chao-style emotions, shaped by personality knobs.
// Every per-duck value is an array, one entry a duck, and each line keeps the Python's order of operations.
import { KNOB_DEFAULT, KNOBS } from "./personality.js";

export const HUNGER_RISE_S = 600.0;
export const BITE_FULLNESS = 0.1;
export const EAT_GROWS = 0.5;
export const THIRST_RISE_S = 800.0;
export const SIP_QUENCH = 0.1;
export const FATIGUE_M = 30.0;
export const REST_S = 60.0;
export const SCENT_MEMORY_S = 30.0;
export const SCENT_FLOOR = 0.05;
export const SEARCH_TURN = 3.0, SEARCH_SLOW = 0.5;
export const CONTENT_BELOW = 0.4, PRESSING_AT = 0.8;
export const BORED_WANDER = 0.4;
export const AMUSED = 0.3;
export const PLAY_BORED = 0.25, PLAY_TIRED = 0.7;
export const REST_BELOW = 0.15;
export const NIGHT_SLEEPINESS = 2.0;
export const DAY_WAKING = 1.0;
export const AWAKE_S = 600.0;
export const SLEEP_S = 120.0;
export const BODY_TEMP_TAU_S = 30.0, WATER_C = 18.0;
export const COMFORT_C = 24.0, COMFORT_BAND_C = 2.0, COMFORT_SPAN_C = 4.0;
export const BORED_S = 120.0;
export const LONELY_S = 60.0;
export const AGGR_TONE_MAX = 0.8;
export const FOOD_NEAR_HALF = 0.5;

export const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const b = x => (x ? 1 : 0);

// How far a need has a duck's attention, 0 to 1.
export const pressing = need => clip((need - CONTENT_BELOW) / (PRESSING_AT - CONTENT_BELOW), 0, 1);

export class Physiology {
  constructor(n, personality = {}, hunger = 0.5, thirst = 0.5, provoked = 0.0, body_temp = COMFORT_C) {
    this.n = n;
    const each = v => Array.isArray(v) ? [...v] : new Array(n).fill(v);
    this.k = Object.fromEntries(KNOBS.map(name => [name, each(personality[name] ?? KNOB_DEFAULT)]));
    this.hunger = each(hunger); this.thirst = each(thirst);
    this.fatigue = each(0); this.sleep_pressure = each(0); this.boredom = each(0);
    this.body_temp = each(body_temp);
    this.asleep = each(false);
    this.alone_s = each(0);
    this.swimming = each(false);
    this.joy = each(0); this.fear = each(0); this.sorrow = each(0);
    this.anger = each(provoked);
    this.scent = each(0); this.scent_was = each(0);
    this.swim_skill = each(0); this.walk_skill = each(0);
    this.eat_skill = each(0); this.dance_skill = each(0); this.fight_skill = each(0);
    this.fashion_skill = each(0); this.music_skill = each(0);
  }

  // f: the frames as arrays by field; escaped and speed: per duck, from the last step. Returns [falls_asleep, wakes].
  step(dt, f, escaped, speed) {
    const k = this.k, n = this.n;
    const falls_asleep = new Array(n), wakes = new Array(n);
    for (let i = 0; i < n; i++) {
      const ate = f.ate[i] > 0, drank = f.drank[i] > 0, bumped = f.bumped[i] > 0, swimming = f.swimming[i] > 0;
      this.swimming[i] = swimming;
      const touching = (f.touch_left[i] + f.touch_right[i]) > 0;
      const ambient = (f.temp_left[i] + f.temp_right[i]) / 2;
      const esc = !!escaped[i], spd = Math.abs(speed[i]);

      this.scent[i] = (f.odor_left[i] + f.odor_right[i]) / 2.0;
      this.scent_was[i] = this.scent_was[i] + (this.scent[i] - this.scent_was[i]) * Math.min(dt / SCENT_MEMORY_S, 1.0);
      this.hunger[i] = clip(this.hunger[i] + dt / HUNGER_RISE_S * (0.5 + k.appetite[i])
                            - BITE_FULLNESS * (1 + EAT_GROWS * this.eat_skill[i]) * b(ate), 0, 1);
      this.thirst[i] = clip(this.thirst[i] + dt / THIRST_RISE_S - SIP_QUENCH * b(drank), 0, 1);
      const tire = spd * dt / FATIGUE_M * (1.5 - k.energy[i]);
      const rest = (spd < 0.01 ? dt / REST_S : 0.0) * (1 + b(this.asleep[i]));
      this.fatigue[i] = clip(this.fatigue[i] + tire - rest, 0, 1);
      const light = f.light[i], dark = 1.0 - light;
      const build = dt / AWAKE_S * (0.5 + k.sleepiness[i]) * (1 + NIGHT_SLEEPINESS * dark);
      this.sleep_pressure[i] = clip(this.sleep_pressure[i] + (this.asleep[i] ? -dt / SLEEP_S : build)
                                    - dt / AWAKE_S * DAY_WAKING * light, 0, 1);
      const target = swimming ? WATER_C : ambient;
      this.body_temp[i] += (target - this.body_temp[i]) * dt / BODY_TEMP_TAU_S;

      const show = b(f.show[i] > 0);
      const fan = clip(2 * k.sociability[i] - 1, 0, 1);
      const put_off = clip(1 - 2 * k.sociability[i], 0, 1);
      const cross = clip((k.aggressiveness[i] - 0.6) / 0.4, 0, 1);
      this.boredom[i] = clip(this.boredom[i] - 0.7 * AMUSED * fan * (1 - cross) * show, 0, 1);
      const kicked = (f.kicked[i] > 0) || (f.drummed[i] > 0);
      const eventful = ate || drank || bumped || esc || touching || kicked;
      this.boredom[i] = clip(eventful ? this.boredom[i] - 0.2 : this.boredom[i] + dt / BORED_S * (0.5 + k.boredom_rate[i]), 0, 1);
      this.alone_s[i] = touching ? 0.0 : this.alone_s[i] + dt;

      const fade = (x, tau) => x * Math.exp(-dt / tau);
      this.joy[i] = clip(fade(this.joy[i], 10.0) + 0.3 * b(ate || drank) + 0.3 * k.playfulness[i] * b(kicked)
                         + 0.2 * fan * (1 - cross) * show, 0, 1);
      this.fear[i] = clip(fade(this.fear[i], 3 + 15 * k.timidity[i] + 5 * (1 - k.aggressiveness[i]))
                          + 0.8 * b(esc) + b(bumped) * k.timidity[i], 0, 1);
      this.anger[i] = clip((bumped ? this.anger[i] + k.aggressiveness[i] : fade(this.anger[i], 3 + 15 * k.aggressiveness[i]))
                           + 0.25 * Math.max(cross, 0.5 * put_off) * show, 0, 1);
      const lonely = this.alone_s[i] > LONELY_S && k.sociability[i] > 0.5;
      const mope = dt / 30.0 * b(lonely) * (k.sociability[i] - 0.5) * 2 + 0.3 * b(bumped) * k.timidity[i];
      this.sorrow[i] = clip(fade(this.sorrow[i], 5 + 20 * (1 - k.curiosity[i])) + mope, 0, 1);

      falls_asleep[i] = !this.asleep[i] && this.sleep_pressure[i] >= 1 && this.fear[i] < 0.2;
      wakes[i] = this.asleep[i] && (this.sleep_pressure[i] <= 0.1 || bumped || esc);
      this.asleep[i] = (this.asleep[i] || falls_asleep[i]) && !wakes[i];
    }
    return [falls_asleep, wakes];
  }

  at_water(humidity) { return clip((humidity - 0.5) / 0.5, 0, 1); }

  amuse(i) {
    this.boredom[i] = Math.max(this.boredom[i] - AMUSED, 0.0);
    this.joy[i] = Math.min(this.joy[i] + 0.1, 1.0);
  }

  play() {
    return this.hunger.map((_, i) => {
      const bored = clip((this.boredom[i] - PLAY_BORED) / (1 - PLAY_BORED), 0, 1);
      const rested = clip(1 - this.fatigue[i] / PLAY_TIRED, 0, 1);
      return this.k.playfulness[i] * bored * rested * b(!this.asleep[i]);
    });
  }

  discomfort() {  // [too hot, too cold], each an array
    const hot = [], cold = [];
    for (let i = 0; i < this.n; i++) {
      const hot_edge = COMFORT_C + COMFORT_BAND_C + 4 * (this.k.heat_tolerance[i] - 0.5);
      hot.push(clip((this.body_temp[i] - hot_edge) / COMFORT_SPAN_C, 0, 1));
      cold.push(clip((COMFORT_C - COMFORT_BAND_C - this.body_temp[i]) / COMFORT_SPAN_C, 0, 1));
    }
    return [hot, cold];
  }

  cooling() {  // [to the pond, to the shade]
    const [hot] = this.discomfort();
    const h = hot.map((x, i) => x * (1 - 0.8 * pressing(this.hunger[i])) * b(!this.asleep[i]));
    return [h.map((x, i) => x * this.k.water_love[i]), h.map((x, i) => x * (1 - this.k.water_love[i]))];
  }

  sense_gains() {
    const k = this.k, [hot, cold] = this.discomfort(), g = {};
    for (const name of ["orn_food", "sugar", "water_taste", "moist_air", "cold", "heat", "orn_danger", "orn_pheromone", "vision"]) g[name] = [];
    for (let i = 0; i < this.n; i++) {
      const need = this.hunger[i] + this.thirst[i] + 1e-9;
      const hungry = 2 * this.hunger[i] / need, thirsty = 2 * this.thirst[i] / need;
      g.orn_food.push((0.5 + this.hunger[i]) * (0.75 + 0.5 * k.appetite[i]) * (1 - 0.6 * this.fear[i]) * hungry);
      const hungry_now = pressing(this.hunger[i]);
      const water = (this.thirst[i] * thirsty + hot[i] * k.water_love[i] * (1 - 0.8 * hungry_now)
                     + 0.5 * k.water_love[i] * (1 - hungry_now));
      const companionship = 2 * k.sociability[i] * (1 - pressing(Math.max(this.hunger[i], this.thirst[i])));
      const care = 1 - 0.6 * k.carelessness[i];
      g.sugar.push(this.hunger[i] * (0.75 + 0.5 * k.appetite[i]));
      g.water_taste.push(this.thirst[i]);
      g.moist_air.push(water);
      g.cold.push(1 + 2 * hot[i] * (1 - k.water_love[i]));
      g.heat.push(1 + 2 * cold[i]);
      g.orn_danger.push(care);
      g.orn_pheromone.push(companionship);
      g.vision.push((0.5 + k.timidity[i]) * care * (1 + this.fear[i]));
    }
    return g;
  }

  // wants and damp: one number or one per duck
  motor(wants = 0.0, damp = 0.0) {
    const k = this.k, [hot] = this.discomfort(), at = (v, i) => Array.isArray(v) ? v[i] : v;
    const m = { speed: [], wander: [], zoomies: [], social: [], asleep: [], swim_urge: [], swim_thirst_weight: [], restlessness: [] };
    for (let i = 0; i < this.n; i++) {
      let lost = clip(1 - this.scent[i] / Math.max(this.scent_was[i], 1e-6), 0, 1) * b(this.scent_was[i] > SCENT_FLOOR);
      lost = lost * pressing(this.hunger[i]);
      const swim_urge = clip(k.water_love[i] * (1 + hot[i]) * (1 - 0.8 * pressing(this.hunger[i])), 0, 1);
      m.speed.push((0.4 + 1.2 * k.energy[i]) * (1 - 0.6 * this.fatigue[i]) * (1 - 0.5 * this.sorrow[i])
                   * (1 - SEARCH_SLOW * lost));
      m.wander.push((0.5 + k.curiosity[i]) * (1 + this.boredom[i]) * (1 + (SEARCH_TURN - 1) * lost));
      m.zoomies.push(this.boredom[i] > 0.8 && k.energy[i] > 0.6 && this.fatigue[i] < 0.3);
      m.social.push(clip((k.sociability[i] - 0.5) * 2, 0, 1));
      m.asleep.push(this.asleep[i]);
      m.swim_urge.push(swim_urge);
      m.swim_thirst_weight.push(1 - 0.5 * k.water_love[i]);
      m.restlessness.push(Math.max(
        pressing(Math.max(this.hunger[i], this.thirst[i])),
        clip((Math.max(BORED_WANDER * this.boredom[i], at(wants, i), swim_urge * at(damp, i)) - REST_BELOW) / 0.5, 0, 1)));
    }
    return m;
  }
}

// pC1d/e input level for one duck.
export function aggression_tone(aggressiveness, hunger, food_odor, provoked, kindness = KNOB_DEFAULT) {
  const near_food = food_odor / (food_odor + FOOD_NEAR_HALF);
  const calm = 1 - Math.max(kindness - KNOB_DEFAULT, 0);
  return AGGR_TONE_MAX * aggressiveness * calm * clip(hunger * near_food + provoked, 0, 1);
}
