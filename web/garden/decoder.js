// brain/decoder.py: descending-neuron spikes -> smoothed rates -> microduck intent. The groups are the Python
// Decoder's own, in its order (web/pack.py saves them); the brain worker counts each group's spikes per tick.
import { clip } from "./physiology.js";

export const DT_MS = 10;
export const TAU_MS = 100.0;
export const BASE_VX = 0.11, RUN_VX = 0.3;
export const VX_PER_HZ = 0.02;
export const VYAW_PER_HZ = 1.0;
export const ESCAPE_TICKS = 50;
export const ESCAPE_SPIKES = 5, ESCAPE_WINDOW = 30;
export const WANDER_VYAW = 0.5, WANDER_TICKS = 50;
export const FEED_HZ = 1.0;
export const STINK_FLOOR_HZ = 0.3, STINK_FULL_HZ = 0.8, STINK_TAU_MS = 2000.0;
export const STINK_VYAW_PER_HZ = 0.5;
export const COMPANIONSHIP_VYAW = 1.5;
export const FLEE = 1.0, CHASE = 1.0;
export const SWIM_GROWS = 0.6, WALK_GROWS = 0.4;
export const BOND_VYAW = 1.5, HAND_VYAW = 1.5, COMFORT_VYAW = 2.0;
export const ROOM_FROM = 0.5, ROOM_GONE = 0.65;
export const CROWDED_VYAW = 1.5;
export const INTENT_VYAW = 2.0;
export const COOL_VYAW = 2.0;
export const PLAY_VYAW = 2.0;
export const PURSUE_VX = 0.2;
export const MUSIC_VYAW = 1.5;
export const GROOM_HZ = 0.4;
export const PREEN_REROLL_TICKS = 500;
export const PREEN_P = 0.3;
export const FEAR_STOPS_FEEDING = 0.5;
export const STINK_LINGER = 0.5;
export const AGGR_FLOOR_HZ = 0.05, AGGR_FULL_HZ = 4.4, AGGR_TAU_MS = 1000.0;
export const WADE_REROLL_TICKS = 500;
export const ATTACK_P = 0.01;
export const TOUCH_HZ = 1.0;

export const [FWD, BACK, STEER_L, STEER_R, GF, FEED, STINK_L, STINK_R, TOUCH_L, TOUCH_R, AIPG, GROOM, WIND_L, WIND_R] =
  Array.from({ length: 14 }, (_, i) => i);
export const GROUPS = 14;
const f32 = Math.fround;
const b = x => (x ? 1 : 0);

export class Decoder {
  // size: members per group; batch: ducks
  constructor(size, batch, rng, stink_affinity = 0.0) {
    this.size = Float32Array.from(size);
    this.tau = new Array(GROUPS).fill(TAU_MS);
    this.tau[STINK_L] = this.tau[STINK_R] = STINK_TAU_MS;
    this.tau[AIPG] = AGGR_TAU_MS;
    this.touch_share = f32(this.size[TOUCH_L] / (this.size[STEER_L] + this.size[TOUCH_L]));
    this.n = batch;
    this.rates = Array.from({ length: batch }, () => new Float32Array(GROUPS));
    this.stink_affinity = Array.isArray(stink_affinity) ? [...stink_affinity] : new Array(batch).fill(stink_affinity);
    this.rng = rng;
    this.wander = new Array(batch).fill(0);
    this.escape_left = new Array(batch).fill(0);
    this.in_stink = new Array(batch).fill(false);
    this.at_water = new Array(batch).fill(false);
    this.wades = new Array(batch).fill(false);
    this.body = {};
    this.lingers = new Array(batch).fill(false);
    this.gf_recent = Array.from({ length: ESCAPE_WINDOW }, () => new Array(batch).fill(0));
    this.ticks = 0;
  }

  // counts[d][g]: how many of group g's neurons spiked this tick in duck d. Returns one intent per duck.
  update(counts) {
    const n = this.n;
    for (let d = 0; d < n; d++) {
      const r = this.rates[d];
      for (let g = 0; g < GROUPS; g++) {
        const per = f32(counts[d][g] / this.size[g]);
        const diff = f32(f32(f32(per * 1000) / DT_MS) - r[g]);
        r[g] = f32(r[g] + DT_MS / this.tau[g] * diff);
      }
    }
    if (this.ticks % WANDER_TICKS === 0) this.wander = this.rng.uniforms(n, -WANDER_VYAW, WANDER_VYAW);
    this.ticks += 1;
    this.gf_recent.pop();
    this.gf_recent.unshift(counts.map(c => c[GF]));
    const at = (key, dflt, d) => { const v = this.body[key] ?? dflt; return Array.isArray(v) ? v[d] : v; };
    const out = [];
    const dice = { wade: this.rng.randoms(n), linger: this.rng.randoms(n), attack: this.rng.randoms(n), preen: this.rng.randoms(n) };
    for (let d = 0; d < n; d++) {
      const r = this.rates[d];
      let recent = 0;
      for (const row of this.gf_recent) recent += row[d];
      const onset = recent >= ESCAPE_SPIKES && this.escape_left[d] === 0;
      this.escape_left[d] = onset ? ESCAPE_TICKS : Math.max(this.escape_left[d] - 1, 0);

      const asleep = !!at("asleep", false, d), swimming = !!at("swimming", false, d), shore = !!at("at_shore", false, d);
      const water = shore || swimming;
      const visit = water && (!this.at_water[d] || this.ticks % WADE_REROLL_TICKS === 0);
      const urge = at("swim_urge", 0.5, d) * (1 - at("thirst", 0.5, d) * at("swim_thirst_weight", 1.0, d));
      this.wades[d] = visit ? dice.wade[d] < urge : this.wades[d] && water;
      this.at_water[d] = water;

      const stink = clip((Math.max(r[STINK_L], r[STINK_R]) - STINK_FLOOR_HZ) / (STINK_FULL_HZ - STINK_FLOOR_HZ), 0, 1);
      const meeting = stink > 0 && !this.in_stink[d];
      if (meeting) this.lingers[d] = dice.linger[d] < this.stink_affinity[d];
      this.in_stink[d] = stink > 0;
      const avoid = stink * b(!this.lingers[d]), like = stink * b(this.lingers[d]) * (1 - at("hunger", 0.0, d));
      const aggression = clip((r[AIPG] - AGGR_FLOOR_HZ) / (AGGR_FULL_HZ - AGGR_FLOOR_HZ), 0, 1);
      const feeding = r[FEED] > FEED_HZ && !this.wades[d] && !swimming && !asleep && !!at("tasting", true, d)
                      && at("fear", 0.0, d) < FEAR_STOPS_FEEDING && !at("sharing", false, d);
      const touched = r[TOUCH_L] + r[TOUCH_R] > TOUCH_HZ;
      const attack = touched && dice.attack[d] < ATTACK_P * aggression && !swimming && !asleep;
      const zoomies = !!at("zoomies", false, d);
      const preen = r[GROOM] > GROOM_HZ && !!at("hatted", false, d) && !asleep && this.ticks % PREEN_REROLL_TICKS === 0
                    && dice.preen[d] < PREEN_P * (1 - at("vanity", 0.5, d));

      let vx = clip(BASE_VX * at("restlessness", 1.0, d) + VX_PER_HZ * (r[FWD] - r[BACK]), -RUN_VX, RUN_VX) * at("speed", 1.0, d);
      if (zoomies) vx = RUN_VX;
      if (swimming && this.wades[d]) vx = vx * (1 - 0.9 * at("swim_urge", 0.5, d));
      if (this.escape_left[d] > 0) vx = -RUN_VX;
      vx = vx + (RUN_VX - vx) * avoid;
      vx = vx + (RUN_VX - Math.max(vx, 0)) * at("surge", 0.0, d) * b(vx > 0);
      vx = vx * (1 - (1 - STINK_LINGER) * like);
      if (feeding) vx = 0.0;
      const ducks_near = clip((at("duck_left", 0.0, d) + at("duck_right", 0.0, d)) / 2, 0, 1);
      const fled = FLEE * clip(at("fear", 0.0, d), 0, 1), chased = CHASE * aggression;
      vx = vx + (RUN_VX - Math.max(vx, 0)) * fled * ducks_near * b(vx >= 0);
      const apart = !touched;
      vx = vx + (PURSUE_VX - Math.min(Math.max(vx, 0), PURSUE_VX)) * chased * ducks_near * b(apart) * b(vx >= 0);
      if (attack) vx = RUN_VX;
      vx = vx * (swimming ? 1 + SWIM_GROWS * at("swim_skill", 0.0, d) : 1 + WALK_GROWS * at("walk_skill", 0.0, d));
      if (asleep) vx = 0.0;

      const share = this.touch_share;
      const toward_touch = Math.max(aggression, at("social", 0.0, d));
      const steer = (1 - share) * (r[STEER_L] - r[STEER_R]) + share * (r[TOUCH_L] - r[TOUCH_R]) * (1 - 2 * toward_touch);
      const wander = this.wander[d] * at("wander", 1.0, d) * (zoomies ? 2.0 : 1.0);
      const taste = 2 * at("music_affinity", 0.5, d) - 1;
      let liking = clip(2 * at("sociability", 0.5, d) - 1 + at("fondness", 0.0, d) + at("sleepy_together", 0.0, d), -1, 1) * at("at_ease", 1.0, d);
      const beside = at("near_left", 0.0, d) + at("near_right", 0.0, d);
      const room = 1 - clip((beside - ROOM_FROM) / (ROOM_GONE - ROOM_FROM), 0, 1);
      const crowded = clip((beside - ROOM_GONE) / 0.1, 0, 1) * (1 - aggression) * at("at_ease", 1.0, d) * (1 - at("intent", 0.0, d));
      if (liking > 0) liking = liking * room;
      liking = clip(liking + chased - 2 * fled, -1, 1);
      const near_side = at("near_left", 0.0, d) - at("near_right", 0.0, d);
      let vyaw = ((wander + VYAW_PER_HZ * (steer + r[WIND_L] - r[WIND_R]))
                  + STINK_VYAW_PER_HZ * (r[STINK_L] - r[STINK_R]) * (like - avoid)
                  + MUSIC_VYAW * taste * (at("music_left", 0.0, d) - at("music_right", 0.0, d))
                  + COMPANIONSHIP_VYAW * liking * (at("duck_left", 0.0, d) - at("duck_right", 0.0, d))
                  + PLAY_VYAW * at("play", 0.0, d) * (at("ball_left", 0.0, d) - at("ball_right", 0.0, d))
                  + COOL_VYAW * (at("cool_left", 0.0, d) - at("cool_right", 0.0, d))
                  + INTENT_VYAW * at("intent_turn", 0.0, d)
                  + at("at_ease", 1.0, d) * (BOND_VYAW * (at("bond_near", 0.0, d) > 0 ? room : 1.0)
                                             * (at("bond_turn", 0.0, d) + at("bond_near", 0.0, d) * near_side)
                                             - CROWDED_VYAW * crowded * near_side
                                             + HAND_VYAW * at("hand_trust", 0.0, d) * (at("hand_left", 0.0, d) - at("hand_right", 0.0, d))
                                             + COMFORT_VYAW * at("comfort", 0.0, d) * (at("cry_left", 0.0, d) - at("cry_right", 0.0, d))));
      if (asleep) vyaw = 0.0;
      out.push({ vx, vy: 0.0, vyaw, escape: onset, feed: feeding, attack, preen, zoomies });
    }
    return out;
  }
}
