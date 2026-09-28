// body/stub2d/stub.py: kinematic ducks on the garden plane behind the microduck contract. The Python serves the
// contract on Unix sockets; here the brain server calls `robot(i, method, params)` and the page calls
// `control(method, params)` directly, and `send_frames` hands back what each duck senses this step.
import { CLIFF_M, DAY_S, DUCK_R, DUCK_SMELL_M, SHORE_M, SIZE_M, TREE, World, contacts, daylight, duck_odor_at, music_at,
         pymod, temperature_at, wind_on } from "./fields.js";
import { luminance } from "./retina.js";
import { Rng } from "./rng.js";

export const DEMO_GARDEN = {
  size: 6.0, tree: [1.5, 4.3, 0.9], food_xy: [[3.0, 2.2]], bites: 10, danger_xy: [[4.9, 1.3]],
  pond: [4.5, 4.4, 1.2], rocks: [[5.5, 5.47, 0.42], [5.95, 5.9, 0.55]], fruit_every_s: 10.0, wind: [0.0, -1.0],
  wind_turns_s: 0.7 * DAY_S, personal_m: 0.24, music: null,
};

export const DT = 0.02;
export const MAX_V = 0.45, MAX_VY = 0.15, MAX_VYAW = 2.0;
export const ANTENNA = [0.06, 0.05];
export const SHAKE_FRUIT = 2;
export const SHAKE_MOST = 12;
export const PUSH_M = 0.15;
export const DOWN_S = 10.0;
export const TRIP_S = 4.0, FLOUNDER_S = 3.0;
export const FUMBLE_M = [0.6, 1.2];
export const HARD = 1.6;
export const BLOCKED_VYAW = 1.5;
export const SWIM_SPEED = 0.5;
export const SOUND_TAGS = new Set(["alarm", "greet", "inquire", "peck", "chirp", "coo", "wheee"]);
export const SIT_AFTER_S = 3.0;
export const DRUM_REACH_M = 0.32;
export const KICK_REACH_M = 0.22, KICK_MS = 1.6;
export const COOL_SEEN_M = 3.0;
export const BALL_SEEN_M = 1.5;
export const NEAR_M = 1.0;
export const EARSHOT_M = 2.0;
export const WITNESS_M = 1.5;
export const GRAB_M = 0.3;
export const THROW_MS = 0.8, THROW_MAX_MS = 3.0, THROW_S = 0.35;
export const HAND_S = 4.0;
export const HAND_SEEN_M = 1.5;
export const CRY_S = 6.0;
export const AUDIENCE_M = 1.5;
export const PERFORMANCES = ["sing", "dance", "singdance"];
export const MAX_BALLS = 3;
export const INSTRUMENTS = ["xylophone", "maracas", "tambourine", "triangle", "toy piano", "guitar", "trumpet", "bell", "recorder", "harp"];
export const MAX_INSTRUMENTS = 4;
export const BALL_STYLES = 10;
export const HAT_REACH_M = 0.25;
export const BITE_S = 0.5;
export const WHEEL_VX = 0.2, WHEEL_VYAW = 1.2;
export const HEAD = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"];
// Fields a frame carries that name a duck, or a fruit for ate_kind; -1 is nobody
export const IDS = ["near_id", "bumped_by", "saw_shove_by", "saw_shove_of", "saw_fall_by", "show_by", "hat_taken_by", "comforted_by", "ate_kind"];

const LEVEL = [0.0, 0.0, 0.0, 0.0];
// How each feeling is acted out: a voice tag and head poses, each [seconds after the last, neck_pitch, head_pitch,
// head_yaw, head_roll] in radians, pitch positive down.
export const EMOTE_ACTS = {
  happy: ["wheee", [[0.0, -0.1, -0.2, 0.0, 0.3], [0.3, -0.1, -0.2, 0.0, -0.3], [0.3, -0.1, -0.2, 0.0, 0.3], [0.3, -0.1, -0.2, 0.0, -0.3]]],
  laugh: ["chirp", [[0.0, -0.2, -0.4, 0.0, 0.1], [0.2, -0.1, -0.2, 0.0, -0.1], [0.2, -0.2, -0.4, 0.0, 0.1], [0.2, -0.1, -0.2, 0.0, -0.1], [0.2, -0.2, -0.4, 0.0, 0.1]]],
  comfort: ["coo", [[0.0, 0.2, 0.3, 0.0, 0.25], [0.7, 0.2, 0.3, 0.0, -0.25], [0.7, 0.2, 0.3, 0.0, 0.25]]],
  playful: ["wheee", [[0.0, 0.0, -0.2, 0.4, 0.3], [0.3, 0.0, -0.2, -0.4, -0.3], [0.3, 0.0, -0.2, 0.4, 0.3]]],
  scared: [null, [[0.0, 0.3, 0.3, 0.0, 0.0], [0.4, 0.3, 0.3, 0.5, 0.0], [0.4, 0.3, 0.3, -0.5, 0.0], [0.6, 0.3, 0.3, 0.0, 0.0]]],
  angry: ["alarm", [[0.0, -0.2, 0.0, 0.0, 0.0], [0.25, 0.3, 0.2, 0.0, 0.0], [0.25, -0.2, 0.0, 0.0, 0.0], [0.25, 0.3, 0.2, 0.0, 0.0]]],
  sad: ["coo", [[0.0, 0.3, 0.4, 0.0, 0.0], [1.0, 0.3, 0.4, 0.2, 0.0], [1.0, 0.3, 0.4, -0.2, 0.0], [1.0, 0.3, 0.4, 0.0, 0.0]]],
  lonely: ["inquire", [[0.0, -0.2, -0.2, 0.6, 0.0], [0.9, -0.2, -0.2, -0.6, 0.0], [0.9, 0.2, 0.3, 0.0, 0.0], [0.8, 0.2, 0.3, 0.0, 0.0]]],
  bored: ["inquire", [[0.0, 0.0, 0.0, 0.6, 0.0], [1.0, 0.0, 0.0, -0.6, 0.0], [1.0, 0.0, 0.2, 0.0, 0.2], [0.8, 0.0, 0.2, 0.0, 0.2]]],
  hungry: ["peck", [[0.0, 0.3, 0.4, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0], [0.3, 0.3, 0.4, 0.0, 0.0], [0.3, 0.0, 0.0, 0.0, 0.0]]],
  thirsty: ["chirp", [[0.0, 0.3, 0.4, 0.0, 0.0], [0.5, -0.2, -0.4, 0.0, 0.0], [0.7, -0.2, -0.4, 0.0, 0.0]]],
  sleepy: ["coo", [[0.0, 0.2, 0.4, 0.0, 0.1], [0.8, 0.0, 0.0, 0.0, 0.0], [0.4, 0.3, 0.4, 0.0, 0.1], [1.0, 0.3, 0.4, 0.0, 0.1]]],
  curious: ["chirp", [[0.0, -0.1, 0.0, 0.3, 0.4], [1.0, -0.1, 0.0, -0.3, -0.4], [1.0, -0.1, 0.0, -0.3, -0.4]]],
  proud: ["greet", [[0.0, -0.3, -0.3, 0.0, 0.0], [0.6, -0.3, -0.3, 0.5, 0.0], [0.6, -0.3, -0.3, -0.5, 0.0], [0.6, -0.3, -0.3, 0.0, 0.0]]],
  stomp: ["alarm", [[0.0, -0.2, -0.2, 0.0, 0.0], [0.2, 0.35, 0.3, 0.0, 0.0], [0.2, -0.2, -0.2, 0.0, 0.0], [0.2, 0.35, 0.3, 0.0, 0.0]]],
  yawn: ["coo", [[0.0, -0.25, -0.5, 0.0, 0.1], [1.2, -0.3, -0.6, 0.0, 0.15], [0.8, 0.2, 0.3, 0.0, 0.0]]],
  splash: ["wheee", [[0.0, 0.4, 0.5, 0.0, 0.0], [0.2, -0.2, -0.3, 0.0, 0.0], [0.2, 0.4, 0.5, 0.0, 0.0], [0.2, -0.2, -0.3, 0.0, 0.0]]],
  sing: ["chirp", [[0.0, -0.25, -0.4, 0.35, 0.0], [0.35, -0.25, -0.4, -0.35, 0.0], [0.35, -0.25, -0.4, 0.35, 0.0], [0.35, -0.25, -0.4, -0.35, 0.0]]],
  cower: ["inquire", [[0.0, 0.45, 0.5, 0.0, 0.0], [0.3, 0.45, 0.5, 0.15, 0.0], [0.15, 0.45, 0.5, -0.15, 0.0], [0.15, 0.45, 0.5, 0.15, 0.0], [0.15, 0.45, 0.5, -0.15, 0.0], [0.8, 0.45, 0.5, 0.0, 0.0]]],
  dance: [null, [[0.0, 0.25, 0.2, 0.4, 0.15], [0.35, -0.1, -0.2, 0.0, 0.0], [0.35, 0.25, 0.2, -0.4, -0.15], [0.35, -0.1, -0.2, 0.0, 0.0],
                 [0.35, 0.25, 0.2, 0.4, 0.15], [0.35, -0.1, -0.2, 0.0, 0.0], [0.35, 0.25, 0.2, -0.4, -0.15], [0.35, -0.1, -0.2, 0.0, 0.0]]],
  singdance: ["chirp", [[0.0, 0.25, 0.2, 0.4, 0.15], [0.35, -0.1, -0.2, 0.0, 0.0], [0.35, 0.25, 0.2, -0.4, -0.15], [0.35, -0.1, -0.2, 0.0, 0.0],
                        [0.35, 0.25, 0.2, 0.4, 0.15], [0.35, -0.1, -0.2, 0.0, 0.0], [0.35, 0.25, 0.2, -0.4, -0.15], [0.35, -0.1, -0.2, 0.0, 0.0]]],
  cry: ["coo", [[0.0, 0.4, 0.5, 0.0, 0.0], [0.4, 0.4, 0.5, 0.2, 0.0], [0.4, 0.4, 0.5, -0.2, 0.0], [0.4, 0.4, 0.5, 0.2, 0.0], [0.4, 0.4, 0.5, -0.2, 0.0], [0.8, 0.4, 0.5, 0.0, 0.0]]],
};

export const DUCK_NAMES = [
  "Pip", "Bramble", "Nutmeg", "Willow", "Otto", "Juniper", "Miso", "Clover", "Rusty", "Pebble",
  "Tofu", "Marigold", "Bean", "Sorrel", "Waffle", "Hazel", "Gus", "Fennel", "Poppy", "Biscuit",
  "Sage", "Mochi", "Thistle", "Noodle", "Basil", "Dill", "Pickle", "Maple", "Comet", "Plum",
  "Ozzy", "Ginger", "Barnaby", "Olive", "Pumpkin", "Wren", "Fig", "Cricket", "Tansy", "Bumble",
  "Scout", "Peanut", "Cedar", "Muffin", "Ivy", "Quill", "Pepper", "Bluebell", "Toast", "Rye",
  "Nettle", "Birch", "Custard", "Sprout", "Ruby", "Domino", "Almond", "Parsley", "Mabel", "Cobweb",
  "Kipper", "Sooty", "Pansy", "Truffle", "Doris", "Walnut", "Chestnut", "Hollis", "Wendell", "Apricot",
  "Fernie", "Cinder", "Bobbin", "Tulip", "Amber", "Pomelo", "Grommet", "Salsa", "Bodhi", "Hopper",
  "Nubbin", "Yarrow", "Teasel", "Pipkin", "Widget", "Cumin", "Pudding", "Lark", "Bunty", "Snapper",
  "Marlow", "Radish", "Crumpet", "Elder", "Pocket", "Saffron", "Tadpole", "Verbena", "Wisteria", "Zinnia",
];

const clip = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const fill = (n, v) => new Array(n).fill(v);

export class Stub {
  constructor(n, seed, { food_xy = [[3.0, 3.0], [1.0, 1.0]], danger_xy = [], pond = null, bites = 1, fruit_every_s = null, pose = null,
                         wind = null, wind_turns_s = null, music = null, size = SIZE_M, tree = TREE, rocks = [], balls = [],
                         personal_m = 0.0, eyes = true } = {}) {
    const rng = new Rng(seed);
    this.n = n;
    this.fruit_rng = new Rng(seed + 1);
    this.fruit_every_s = fruit_every_s;
    this.world = new World({ food_xy, danger_xy, pond, bites, wind, wind_turns_s, music, size, tree, rocks });
    this.ball_rng = new Rng(seed + 4);
    this.ball_styles = [];
    for (const [x, y] of balls) { this.world.balls.push([x, y, 0, 0]); this.ball_styles.push(this.ball_rng.integers(BALL_STYLES)); }
    this.pose = pose ? pose.map(p => [...p]) : Array.from({ length: n }, () => [rng.uniform(0.5, size - 0.5), rng.uniform(0.5, size - 0.5), 0]);
    if (!pose) this.pose.forEach(p => { p[2] = rng.uniform(-Math.PI, Math.PI); });
    this.eyes = eyes;  // whether to draw what each duck sees; only the eyes and the ride view need it
    this.eaten = []; this.headbutts = []; this.emotes = [];
    this.acting = Array.from({ length: n }, () => []);
    this.bumped = fill(n, false); this.petted = fill(n, false); this.scared = fill(n, false); this.hats = fill(n, false);
    this.still_for = fill(n, 0);
    this.hat_style = fill(n, -1);
    this.hat_items = [];
    this.donned = []; this.kicks = []; this.kicked = fill(n, false);
    this.drums = []; this.drummed = fill(n, false); this.saw_show = fill(n, false);
    this.events = Object.fromEntries(IDS.filter(k => k !== "near_id").map(k => [k, fill(n, -1)]));
    this.heard = { heard_alarm: fill(n, 0), heard_joy: fill(n, 0) };
    this.hand_fed = fill(n, false);
    this.crying_until = fill(n, 0);
    this.hand_until = 0.0;
    this.held = null;  // [kind, index]
    this.thrown = fill(n, false);
    this.throws = []; this.given = [];
    this.velocity = Array.from({ length: n }, () => [0, 0]);
    this.hat_rng = new Rng(seed + 2);
    this.preened = []; this.pets = []; this.fails = [];
    this.slip_rng = new Rng(seed + 3);
    this.ate = fill(n, false); this.drank = fill(n, false);
    this.sounds = [];
    this.last_bite = fill(n, -Infinity);
    this.cmd = Array.from({ length: n }, () => [0, 0, 0]);
    this.head = Array.from({ length: n }, () => [0, 0, 0, 0]);
    this.relaxed = fill(n, false);
    this.seen = null;
    this.touch_m = 2 * DUCK_R;
    this.personal_m = personal_m;
    if (this.personal_m > 0) this.touch_m = this.personal_m + 0.05;
    this.down_until = fill(n, 0);
    this.t = 0.0;
    this.duck_names = new Rng((Math.random() * 2 ** 32) >>> 0).sample(DUCK_NAMES, n);
  }

  // ---- the robot contract, one duck at a time ----
  robot(i, method, p = {}) {
    ({ "robot.move": () => this._move(i, p), "robot.head": () => this._head(i, p), "robot.do": () => this._do(i, p),
       "robot.sound": () => this._sound(i, p), "robot.stop": () => { this.cmd[i] = [0, 0, 0]; },
       "robot.relax": () => { this.relaxed[i] = true; this.cmd[i] = [0, 0, 0]; },
       "robot.init": () => { this.relaxed[i] = false; } })[method]();
  }

  _move(i, p) {
    if (this.t < this.down_until[i]) this.cmd[i] = [0, 0, 0];
    else if (!this.relaxed[i]) this.cmd[i] = [clip(+p.vx, -MAX_V, MAX_V), clip(+p.vy, -MAX_VY, MAX_VY), clip(+p.vyaw, -MAX_VYAW, MAX_VYAW)];
  }

  _head(i, p) { this.head[i] = HEAD.map(k => +p[k]); }

  _do(i, p) {
    const skill = p.skill;
    if (skill.startsWith("emote_")) { this._emote(i, skill.slice(6)); return; }
    const act = { ground_pick: () => this._pick(i), headbutt: () => this._headbutt(i), drink: () => this._drink(i),
                  kick: () => this._kick(i), drum: () => this._tap_drum(i), sour: () => this._tap_drum(i, true),
                  wear: () => this._wear(i), preen: () => this._shed(i), trip: () => this._trip(i), flounder: () => this._flounder(i),
                  fumble: () => this._fumble(i), headbutt_hard: () => this._headbutt(i, true), whiff: () => this._whiff(i),
                  lose_hat: () => this._lose_hat(i) }[skill];
    if (act) act();
  }

  _near(i, m) {  // which ducks are within m of duck i, not counting it
    return this.pose.map((p, j) => j !== i && dist(p, this.pose[i]) < m);
  }

  _audience(i) {
    this._near(i, AUDIENCE_M).forEach((near, j) => { if (near) { this.saw_show[j] = true; this.events.show_by[j] = i; } });
  }

  _kick(i) {
    const fwd = [Math.cos(this.pose[i][2]), Math.sin(this.pose[i][2])];
    for (const ball of this.world.balls) {
      const rx = ball[0] - this.pose[i][0], ry = ball[1] - this.pose[i][1];
      if (Math.hypot(rx, ry) < KICK_REACH_M && rx * fwd[0] + ry * fwd[1] > 0) {
        ball[2] += KICK_MS * fwd[0]; ball[3] += KICK_MS * fwd[1];
        this.kicks.push([this.t, i]);
        this.kicked[i] = true;
        return;
      }
    }
  }

  _instruments() {  // [x, y, name, sound tag], the drum first
    const w = this.world;
    const drum = w.drum !== null ? [[w.drum[0], w.drum[1], "drum", "drum"]] : [];
    return drum.concat(w.instruments.map(([x, y, k]) => [x, y, INSTRUMENTS[k], `instrument:${k}`]));
  }

  _tap_drum(i, sour = false) {
    const near = this._instruments().map(([x, y, name, tag]) => [Math.hypot(x - this.pose[i][0], y - this.pose[i][1]), name, tag])
      .filter(r => r[0] < DRUM_REACH_M);
    if (!near.length) return;
    near.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
    const [, name, tag] = near[0];
    this.drums.push([this.t, i, name]);
    this.drummed[i] = true;
    this.sounds.push([this.t, i, tag + (sour ? ":sour" : "")]);
    if (sour) this._fell(i, `hit a sour note on the ${name}`);
    this._audience(i);
  }

  _wear(i) {
    const near = this.hat_items.map((h, k) => [k, h]).filter(([, [x, y]]) => Math.hypot(x - this.pose[i][0], y - this.pose[i][1]) < HAT_REACH_M);
    if (this.hats[i] || !near.length) return;
    this.hats[i] = true;
    this.hat_style[i] = this.hat_items.splice(near[0][0], 1)[0][2];
    this.donned.push([this.t, i]);
    this._near(i, WITNESS_M).forEach((saw, j) => { if (saw) this.events.hat_taken_by[j] = i; });
  }

  _shed(i) {
    if (!this.hats[i]) return;
    this.hats[i] = false;
    this.hat_items.push([this.pose[i][0], this.pose[i][1], Math.max(this.hat_style[i], 0)]);
    this.hat_style[i] = -1;
    this.preened.push([this.t, i]);
  }

  _emote(i, feeling) {
    if (this.acting[i].length || this.relaxed[i] || this.t < this.down_until[i]) return;
    const [tag, poses] = EMOTE_ACTS[feeling];
    this.emotes.push([this.t, i, feeling]);
    if (feeling === "cry") this.crying_until[i] = this.t + CRY_S;
    if (PERFORMANCES.includes(feeling)) this._audience(i);
    if (tag) this._sound(i, { tag });
    let at = this.t;
    for (const [after, ...pose] of [...poses, [0.5, ...LEVEL]]) {
      at += after;
      this.acting[i].push([at, pose]);
    }
  }

  _act() {
    this.acting.forEach((queue, i) => {
      while (queue.length && queue[0][0] <= this.t) {
        const pose = queue.shift()[1];
        this._head(i, Object.fromEntries(HEAD.map((k, j) => [k, pose[j]])));
      }
    });
  }

  _sound(i, p) {
    if (!SOUND_TAGS.has(p.tag)) throw new Error(`unknown sound tag ${p.tag}`);
    this.sounds.push([this.t, i, p.tag]);
    if (p.tag === "alarm" || p.tag === "wheee") {
      const heard = this.heard[p.tag === "alarm" ? "heard_alarm" : "heard_joy"];
      this._near(i, EARSHOT_M).forEach((near, j) => { if (near) heard[j] = 1.0; });
    }
  }

  _pick(i) {
    const [, , dish] = contacts(this.pose.map(p => [p[0], p[1]]), this.pose.map(p => p[2]), this.world.food);
    if (dish[i] >= 0 && this.t - this.last_bite[i] >= BITE_S) {
      this.events.ate_kind[i] = this.world.kinds[dish[i]];
      const hand = this.world.hand;
      this.hand_fed[i] = hand !== null && Math.hypot(hand[0] - this.pose[i][0], hand[1] - this.pose[i][1]) < 0.5;
      this.world.eat(dish[i]);
      this.eaten.push([this.t, i]);
      this.last_bite[i] = this.t;
      this.ate[i] = true;
    }
  }

  _drink(i) {
    const shore = Math.abs(this.world.pond_distance(this.pose[i])) <= SHORE_M;
    if (shore && this.t - this.last_bite[i] >= BITE_S) { this.last_bite[i] = this.t; this.drank[i] = true; }
  }

  posture() {
    return this.down_until.map((until, i) => this.t < until ? "down" : this.still_for[i] > SIT_AFTER_S ? "sat" : "up");
  }

  down_left() { return this.down_until.map(u => Math.max(u - this.t, 0)); }

  _swimming() { return this.pose.map(p => this.world.pond_distance(p) < -SHORE_M); }

  _headbutt(i, hard = false) {
    const fwd = [Math.cos(this.pose[i][2]), Math.sin(this.pose[i][2])];
    const hit = this.pose.map((p, j) => {
      const rx = p[0] - this.pose[i][0], ry = p[1] - this.pose[i][1];
      return j !== i && Math.hypot(rx, ry) < this.touch_m && rx * fwd[0] + ry * fwd[1] > 0 && this.t >= this.down_until[j];
    });
    hit.forEach((h, j) => {
      if (!h) return;
      const push = PUSH_M * (hard ? HARD : 1);
      this.pose[j][0] = clip(this.pose[j][0] + push * fwd[0], DUCK_R, this.world.size - DUCK_R);
      this.pose[j][1] = clip(this.pose[j][1] + push * fwd[1], DUCK_R, this.world.size - DUCK_R);
      this.bumped[j] = true;
      this.events.bumped_by[j] = i;
      this.pose.forEach((p, k) => {
        if (k !== i && k !== j && dist(p, this.pose[j]) < WITNESS_M) { this.events.saw_shove_by[k] = i; this.events.saw_shove_of[k] = j; }
      });
      this.headbutts.push([this.t, i, j]);
      this.knock_down(j, DOWN_S * (hard ? HARD : 1));
    });
  }

  knock_down(j, seconds = DOWN_S) {
    this.down_until[j] = Math.max(this.down_until[j], this.t + seconds);
    this.cmd[j] = [0, 0, 0];
    this.acting[j] = [];
  }

  _fell(i, what) {
    this.fails.push([this.t, i, what]);
    this._near(i, WITNESS_M).forEach((seen, j) => { if (seen) this.events.saw_fall_by[j] = i; });
  }

  _trip(i) { if (this.t >= this.down_until[i]) { this.knock_down(i, TRIP_S); this._fell(i, "tripped over"); } }
  _lose_hat(i) { if (this.hats[i]) { this._shed(i); this._fell(i, "lost its hat"); } }
  _whiff(i) { if (this.t >= this.down_until[i]) { this.knock_down(i, TRIP_S); this._fell(i, "swung and missed"); } }
  _flounder(i) {
    if (this._swimming()[i] && this.t >= this.down_until[i]) { this.knock_down(i, FLOUNDER_S); this._fell(i, "went under"); }
  }

  _fumble(i) {
    const food = this.world.food;
    if (!food.length) return;
    let dish = 0;
    food.forEach((f, k) => { if (dist(f, this.pose[i]) < dist(food[dish], this.pose[i])) dish = k; });
    const away = this.slip_rng.uniform(...FUMBLE_M);
    food[dish][0] = clip(food[dish][0] + away * Math.cos(this.pose[i][2]), DUCK_R, this.world.size - DUCK_R);
    food[dish][1] = clip(food[dish][1] + away * Math.sin(this.pose[i][2]), DUCK_R, this.world.size - DUCK_R);
    this._fell(i, "kicked the fruit away");
  }

  shake_tree() { return this.world.drop_fruit(this.fruit_rng, SHAKE_FRUIT, SHAKE_MOST); }

  // ---- the player's calls ----
  control(method, p = {}) {
    const handlers = {
      "garden.pet": () => { const i = Math.trunc(p.duck); this.petted[i] = true; this.pets.push([this.t, i]); },
      "garden.music": () => { this.world.music = +p.on ? [+p.x, +p.y] : null; },
      "garden.hat": () => {
        const i = Math.trunc(p.duck);
        this.hats[i] = !!+p.on;
        this.hat_style[i] = this.hats[i] ? this.hat_rng.integers(1_000_000) : -1;
      },
      "garden.drop_hat": () => { this.hat_items.push([+p.x, +p.y, this.hat_rng.integers(1_000_000)]); },
      "garden.drop_ball": () => {
        this.world.balls = [...this.world.balls, [+p.x, +p.y, 0, 0]].slice(-MAX_BALLS);
        this.ball_styles = [...this.ball_styles, this.ball_rng.integers(BALL_STYLES)].slice(-MAX_BALLS);
      },
      "garden.drum": () => { this.world.drum = +p.on ? [+p.x, +p.y] : null; },
      "garden.instrument": () => {
        const kind = this.ball_rng.integers(INSTRUMENTS.length);
        this.world.instruments = [...this.world.instruments, [+p.x, +p.y, kind]].slice(-MAX_INSTRUMENTS);
      },
      "garden.volume": () => { this.world.music_volume = clip(+p.level, 0, 1); },
      "garden.give": () => {
        const i = Math.trunc(p.duck), h = this.pose[i][2];
        this._hand({ x: this.pose[i][0] + 0.12 * Math.cos(h), y: this.pose[i][1] + 0.12 * Math.sin(h), feed: 3 });
        this.given.push([this.t, i]);
      },
      "garden.grab": () => this._grab(p),
      "garden.hand_at": () => this._hand_at(p),
      "garden.release": () => this._release(p),
      "garden.scare": () => { this.scared.fill(true); },
      "garden.hand": () => this._hand(p),
      "garden.shake_tree": () => ({ fell: this.shake_tree() }),
    };
    return handlers[method]() ?? {};
  }

  // The lowest and highest a duck's centre may go on either axis. A drawn duck keeps its whole body inside the fence
  // and off the cliffs, which stand into the lawn along the far edges; a duck in a gate is a point.
  _walls() {
    if (this.personal_m <= 0) return [DUCK_R, this.world.size - DUCK_R];
    const body = Math.max(DUCK_R, this.personal_m / 2);
    return [body, this.world.size - body - CLIFF_M];
  }

  _keep_apart() {
    const n = this.n;
    if (this.personal_m <= 0 || n < 2) return;
    const down = this.down_until.map(u => this.t < u);
    for (let pass = 0; pass < 3; pass++) {
      const moves = Array.from({ length: n }, () => [0, 0]);
      let any = false;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const rx = this.pose[i][0] - this.pose[j][0], ry = this.pose[i][1] - this.pose[j][1];
        const d = Math.hypot(rx, ry);
        if (d < this.personal_m) {
          any = true;
          const share = down[i] === down[j] ? 0.5 : down[i] ? 0.0 : 1.0;
          const [ax, ay] = d > 1e-6 ? [rx / Math.max(d, 1e-6), ry / Math.max(d, 1e-6)] : [1.0, 0.0];
          moves[i][0] += ax * (this.personal_m - d) * share;
          moves[i][1] += ay * (this.personal_m - d) * share;
        }
      }
      if (!any) return;
      moves.forEach((m, i) => { this.pose[i][0] += m[0]; this.pose[i][1] += m[1]; });
    }
    for (const p of this.pose) {
      p[0] = clip(p[0], DUCK_R, this.world.size - DUCK_R);
      p[1] = clip(p[1], DUCK_R, this.world.size - DUCK_R);
    }
  }

  _things() {  // everything the hand could pick up, as [kind, which, x, y], things before ducks
    const w = this.world;
    return [
      ...w.balls.map((b, k) => ["ball", k, b[0], b[1]]),
      ...this.hat_items.map((h, k) => ["hat", k, h[0], h[1]]),
      ...w.food.map((f, k) => ["food", k, f[0], f[1]]),
      ...(w.music ? [["music", 0, ...w.music]] : []),
      ...(w.drum ? [["drum", 0, ...w.drum]] : []),
      ...w.instruments.map(([x, y], k) => ["instrument", k, x, y]),
      ...this.pose.map((p, i) => ["duck", i, p[0], p[1]]),
    ];
  }

  _grab(p) {
    this._hand_at(p);
    let best = null;
    for (const [kind, which, x, y] of this._things()) {
      const extra = kind === "duck" ? 0.1 : 0.0;
      const d = Math.hypot(x - p.x, y - p.y) + extra;
      if (d < GRAB_M + extra && (best === null || d < best[0])) best = [d, kind, which];
    }
    this.held = best ? [best[1], best[2]] : null;
  }

  _hand_at(p) {
    const at = [clip(+p.x, 0.05, this.world.size - 0.05), clip(+p.y, 0.05, this.world.size - 0.05)];
    this.world.hand = at;
    this.hand_until = this.t + HAND_S;
    this._carry(at);
  }

  _carry(at, velocity = [0, 0]) {
    if (this.held === null) return;
    const [kind, which] = this.held, w = this.world;
    if (kind === "ball" && which < w.balls.length) w.balls[which] = [at[0], at[1], velocity[0], velocity[1]];
    else if (kind === "hat" && which < this.hat_items.length) { this.hat_items[which][0] = at[0]; this.hat_items[which][1] = at[1]; }
    else if (kind === "food" && which < w.food.length) w.food[which] = [at[0], at[1]];
    else if (kind === "music") w.music = [at[0], at[1]];
    else if (kind === "drum") w.drum = [at[0], at[1]];
    else if (kind === "instrument" && which < w.instruments.length) { w.instruments[which][0] = at[0]; w.instruments[which][1] = at[1]; }
    else if (kind === "duck") { this.pose[which][0] = at[0]; this.pose[which][1] = at[1]; this.cmd[which] = [0, 0, 0]; }
  }

  _release(p) {
    if (this.held === null) return;
    let v = [+p.vx, +p.vy];
    const speed = Math.hypot(v[0], v[1]);
    const cap = Math.min(1.0, THROW_MAX_MS / Math.max(speed, 1e-9));
    v = [v[0] * cap, v[1] * cap];
    const at = [+p.x, +p.y];
    const [kind, which] = this.held;
    const thrown = speed > THROW_MS && kind !== "food";
    if (kind === "ball") this._carry(at, thrown ? v : [0, 0]);
    else {
      const land = [clip(at[0] + (thrown ? v[0] * THROW_S : 0), 0.1, this.world.size - 0.1),
                    clip(at[1] + (thrown ? v[1] * THROW_S : 0), 0.1, this.world.size - 0.1)];
      this._carry(land);
      if (kind === "duck" && thrown) { this.thrown[which] = true; this.throws.push([this.t, which]); this.knock_down(which); }
    }
    this.held = null;
  }

  _hand(p) {
    this.world.hand = [+p.x, +p.y];
    this.hand_until = this.t + HAND_S;
    if (p.feed) this.world.add_food(this.world.hand, Math.trunc(p.feed));
  }

  step() {
    const n = this.n, before = this.pose.map(p => [p[0], p[1]]);
    const swim = this._swimming();
    const h = new Array(n);
    let speeds = new Array(n);
    for (let i = 0; i < n; i++) {
      const s = swim[i] ? SWIM_SPEED : 1.0;
      const [vx, vy, vyaw] = [this.cmd[i][0] * s, this.cmd[i][1] * s, this.cmd[i][2] * s];
      h[i] = this.pose[i][2] + vyaw * DT;
      this.pose[i][0] += (vx * Math.cos(h[i]) - vy * Math.sin(h[i])) * DT;
      this.pose[i][1] += (vx * Math.sin(h[i]) + vy * Math.cos(h[i])) * DT;
      speeds[i] = Math.hypot(vx, vy);
    }
    this.still_for = this.still_for.map((s, i) => (speeds[i] > 0.01 || this.t < this.down_until[i]) ? 0.0 : s + DT);
    let free = this.pose.map(p => [p[0], p[1]]);
    const [lo, hi] = this._walls();
    for (const p of this.pose) { p[0] = clip(p[0], lo, hi); p[1] = clip(p[1], lo, hi); }
    const blocked = this.pose.map((p, i) => dist(p, free[i]) > 1e-6);
    this._keep_apart();
    if (this.held !== null && this.held[0] === "duck" && this.world.hand !== null) {
      this.pose[this.held[1]][0] = this.world.hand[0]; this.pose[this.held[1]][1] = this.world.hand[1];
    }
    free = this.pose.map(p => [p[0], p[1]]);
    const xy = this.pose.map(p => [p[0], p[1]]);
    this.world.push_out(xy, Math.max(DUCK_R, this.personal_m / 2));  // a drawn duck keeps its body off the rock
    xy.forEach((p, i) => { this.pose[i][0] = p[0]; this.pose[i][1] = p[1]; if (dist(p, free[i]) > 1e-6) blocked[i] = true; });
    // explicit code: a duck stopped by the fence or a rock turns toward the middle
    for (let i = 0; i < n; i++) {
      if (!blocked[i]) continue;
      const ax = Math.cos(h[i]), ay = Math.sin(h[i]);
      const mx = this.world.size / 2 - this.pose[i][0], my = this.world.size / 2 - this.pose[i][1];
      h[i] += (ax * my - ay * mx >= 0 ? 1.0 : -1.0) * BLOCKED_VYAW * DT;
    }
    this.velocity = this.pose.map((p, i) => [(p[0] - before[i][0]) / DT, (p[1] - before[i][1]) / DT]);
    this.world.roll_balls(DT, this.pose.map(p => [p[0], p[1]]), this.velocity);
    if (this.world.hand !== null && this.t > this.hand_until) { this.world.hand = null; this.held = null; }
    this._act();
    for (let i = 0; i < n; i++) this.pose[i][2] = pymod(h[i] + Math.PI, 2 * Math.PI) - Math.PI;
    this.world.step(this.t);
    this.t += DT;
    if (this.fruit_every_s && Math.trunc(this.t / this.fruit_every_s) > Math.trunc((this.t - DT) / this.fruit_every_s)) {
      this.world.drop_fruit(this.fruit_rng);
    }
  }

  // What each duck senses this step, as {field: one value a duck} (body/frames.py's fields). Clears the one-step events.
  send_frames(want_sight = this.eyes) {
    const n = this.n, w = this.world, xy = this.pose.map(p => [p[0], p[1]]), h = this.pose.map(p => p[2]);
    const fwd = h.map(a => [Math.cos(a), Math.sin(a)]), left = h.map(a => [-Math.sin(a), Math.cos(a)]);
    const light = daylight(this.t);
    const f = { t: this.t, x: xy.map(p => p[0]), y: xy.map(p => p[1]), heading: [...h] };
    for (const [side, sign] of [["left", 1], ["right", -1]]) {
      const at = xy.map((p, i) => [p[0] + ANTENNA[0] * fwd[i][0] + sign * ANTENNA[1] * left[i][0],
                                   p[1] + ANTENNA[0] * fwd[i][1] + sign * ANTENNA[1] * left[i][1]]);
      f[`odor_${side}`] = at.map(p => w.odor_at(p));
      f[`danger_${side}`] = at.map(p => w.odor_at(p, w.danger_odor));
      f[`humidity_${side}`] = at.map(p => w.humidity_at(p));
      f[`temp_${side}`] = at.map(p => temperature_at(p, light, w.tree));
      f[`duck_${side}`] = at.map((p, i) => duck_odor_at(p, xy, i));
      f[`scent_${side}`] = at.map((p, i) => xy.map((q, j) => j === i ? 0.0 : Math.exp(-dist(q, p) / DUCK_SMELL_M)));
      f[`music_${side}`] = at.map(p => music_at(p, w.music) * w.music_volume);
    }
    [f.touch_left, f.touch_right] = contacts(xy, h, w.food, this.touch_m);
    const dish = contacts(xy, h, w.food, this.touch_m)[2];
    f.sugar = dish.map(d => +(d >= 0));
    const edge = xy.map(p => w.pond_distance(p));
    f.water = edge.map(e => +(Math.abs(e) <= SHORE_M));
    f.swimming = edge.map(e => +(e < -SHORE_M));
    f.bumped = this.bumped.map(Number); f.ate = this.ate.map(Number); f.drank = this.drank.map(Number);
    f.petted = this.petted.map(Number); f.scared = this.scared.map(Number);
    this.seen = f.lum = want_sight ? luminance(xy, h, w, light) : null;
    for (const a of [this.bumped, this.ate, this.drank, this.petted, this.scared]) a.fill(false);
    f.light = new Array(n).fill(light);
    f.hat = this.hats.map(Number);
    const seen = [new Array(n).fill(0), new Array(n).fill(0)], near = new Array(n).fill(false);
    for (const ball of w.balls) {
      for (let i = 0; i < n; i++) {
        const rx = ball[0] - xy[i][0], ry = ball[1] - xy[i][1], d = Math.hypot(rx, ry);
        const ahead = rx * fwd[i][0] + ry * fwd[i][1], to_left = rx * left[i][0] + ry * left[i][1];
        const size = ahead > -0.3 * d ? 1 / (1 + d / BALL_SEEN_M) : 0.0;
        if (to_left >= 0) seen[0][i] = Math.max(seen[0][i], size); else seen[1][i] = Math.max(seen[1][i], size);
        near[i] = near[i] || (d < KICK_REACH_M && ahead > 0);
      }
    }
    [f.ball_left, f.ball_right] = seen;
    f.ball_near = near.map(Number);
    f.drum_left = new Array(n).fill(0); f.drum_right = new Array(n).fill(0); f.drum_near = new Array(n).fill(0);
    const spots = this._instruments();
    if (spots.length) {
      for (let i = 0; i < n; i++) {
        let best = 0;
        spots.forEach((s, k) => { if (dist(s, xy[i]) < dist(spots[best], xy[i])) best = k; });
        const rx = spots[best][0] - xy[i][0], ry = spots[best][1] - xy[i][1], d = Math.hypot(rx, ry);
        const plain = rx * fwd[i][0] + ry * fwd[i][1] > -0.3 * d ? 1 / (1 + d / BALL_SEEN_M) : 0.0;
        const on_left = rx * left[i][0] + ry * left[i][1] >= 0;
        f.drum_left[i] = on_left ? plain : 0; f.drum_right[i] = on_left ? 0 : plain;
        f.drum_near[i] = +(d < DRUM_REACH_M);
      }
    }
    for (const [name, place] of [["pond", w.pond], ["shade", w.tree]]) {
      f[`${name}_left`] = new Array(n).fill(0); f[`${name}_right`] = new Array(n).fill(0);
      if (place === null) continue;
      for (let i = 0; i < n; i++) {
        const rx = place[0] - xy[i][0], ry = place[1] - xy[i][1];
        const plain = 1 / (1 + Math.max(Math.hypot(rx, ry) - place[2], 0) / COOL_SEEN_M);
        if (rx * left[i][0] + ry * left[i][1] >= 0) f[`${name}_left`][i] = plain; else f[`${name}_right`][i] = plain;
      }
    }
    f.drummed = this.drummed.map(Number); this.drummed.fill(false);
    f.kicked = this.kicked.map(Number); f.show = this.saw_show.map(Number);
    this.kicked.fill(false); this.saw_show.fill(false);
    const on_side = (i, x, y) => (x - xy[i][0]) * left[i][0] + (y - xy[i][1]) * left[i][1] >= 0;
    const d = xy.map((p, i) => xy.map((q, j) => i === j ? 1e9 : dist(q, p)));
    f.near_id = []; f.near_left = []; f.near_right = [];
    for (let i = 0; i < n; i++) {
      let nearest = 0;
      d[i].forEach((v, j) => { if (v < d[i][nearest]) nearest = j; });
      const with_one = d[i][nearest] < NEAR_M;
      const plain = with_one ? 1 - d[i][nearest] / NEAR_M : 0.0;
      const l = on_side(i, ...xy[nearest]);
      f.near_id.push(with_one ? nearest : -1);
      f.near_left.push(l ? plain : 0); f.near_right.push(l ? 0 : plain);
    }
    const crying = this.crying_until.map(u => u > this.t);
    f.cry_left = new Array(n).fill(0); f.cry_right = new Array(n).fill(0);
    crying.forEach((c, j) => {
      if (!c) return;
      for (let i = 0; i < n; i++) {
        const loud = d[i][j] < EARSHOT_M ? 1 - d[i][j] / EARSHOT_M : 0.0;
        if (on_side(i, ...xy[j])) f.cry_left[i] = Math.max(f.cry_left[i], loud); else f.cry_right[i] = Math.max(f.cry_right[i], loud);
      }
    });
    // a duck that comes right up to a crying one has comforted it
    crying.forEach((c, j) => {
      if (!c) return;
      const comforter = d.findIndex((row, i) => row[j] < this.touch_m);
      if (comforter < 0) return;
      this.events.comforted_by[j] = comforter;
      this.crying_until[j] = 0.0;
      this._emote(comforter, "comfort");
    });
    f.hand_left = new Array(n).fill(0); f.hand_right = new Array(n).fill(0);
    if (w.hand !== null) {
      for (let i = 0; i < n; i++) {
        const plain = 1 / (1 + dist(w.hand, xy[i]) / HAND_SEEN_M);
        if (on_side(i, ...w.hand)) f.hand_left[i] = plain; else f.hand_right[i] = plain;
      }
    }
    f.hand_fed = this.hand_fed.map(Number);
    f.held = xy.map((_, i) => +(this.held !== null && this.held[0] === "duck" && this.held[1] === i));
    f.thrown = this.thrown.map(Number);
    this.thrown.fill(false); this.hand_fed.fill(false);
    for (const [name, values] of Object.entries({ ...this.events, ...this.heard })) {
      f[name] = [...values];
      values.fill(IDS.includes(name) ? -1 : 0);
    }
    f.hat_near = xy.map(p => +this.hat_items.some(([x, y]) => Math.hypot(p[0] - x, p[1] - y) < HAT_REACH_M));
    const wind = h.map(a => wind_on(a, w.wind));
    f.wind = wind.map(x => x[0]); f.wind_from = wind.map(x => x[1]);
    return f;
  }

  state() {
    const w = this.world;
    return { t: this.t, pose: this.pose, food: w.food, eaten: this.eaten, music: w.music, drum: w.drum };
  }
}
