// viewer/godot/duck.gd: one duck, the microduck robot simplified (robot.json, from Pollen Robotics' meshes), each body on
// its own hinge under its parent, posed by motions recorded from the simulator (clips.json). It faces +x and stands on
// y = 0. Personality sets proportions: appetite widens it, timidity shrinks it, vanity grows the head.
import * as THREE from "three";
import * as Ink from "./ink.js";
import * as Hats from "./hats.js";
import { faceModel } from "./faces.js";

export const CELL = 5.0;
export const LOOK = 1.9;
export const EMOTE_S = 2.8;
export const PUSH_M = 0.15;
export const KNOCKED_S = 0.3;
const RIBBONS = [Ink.CORAL, Ink.TEAL, Ink.MUSTARD, new THREE.Color("#7d6bd0"), new THREE.Color("#e58ac0")];
export const DOES = { laugh: "laughs", comfort: "comforts", cry: "cries", dance: "dances", singdance: "sings and dances", stomp: "stomps",
                      yawn: "yawns", splash: "splashes", sing: "sings", cower: "cowers" };
const MOOD_SHAPES = { joy: "ball", fear: "spike", anger: "block", sorrow: "drop" };
const WIRE = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
              "head_yaw", "head_roll", "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"];
const ONCE = { sit_down: 2.0, stand_up: 3.0, kick: 2.0, roll: 1.3, get_up: 2.5, go_limp: 1.0 };
const TAU = 2 * Math.PI;
const UP = new THREE.Vector3(0, 1, 0);
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (x, lo, hi) => Math.min(Math.max(x, lo), hi);
export const angleDifference = (from, to) => { const d = (to - from) % TAU; return (2 * d) % TAU - d; };
export const lerpAngle = (from, to, t) => from + angleDifference(from, to) * t;
const fmod = (a, b) => a - b * Math.floor(a / b);

let robot = null, clips = null;
// robot.json with its meshes built, shared by every duck, and clips.json
export function setData(robotJson, clipsJson) {
  robot = robotJson;
  clips = clipsJson;
  for (const body of robot.bodies) {
    for (const p of body.parts) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(p.v.map(x => x * robot.unit), 3));
      const idx = [];
      for (let k = 0; k < p.i.length; k += 3) idx.push(p.i[k], p.i[k + 2], p.i[k + 1]);  // robot.json winds clockwise, Godot's front
      geo.setIndex(idx);
      geo.computeVertexNormals();
      p.geometry = geo;
    }
  }
  // Which way the recorded fall tips the duck, in the duck's own frame: where its trunk's top ends up.
  const q = clips.clips.go_limp.frames.at(-1)[2];
  const top = new THREE.Vector3(0, 0, 1).applyQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]).normalize());
  robot.fallWay = new THREE.Vector3(top.x, 0, -top.y).normalize();
}

export class Duck extends THREE.Group {
  constructor(index, knobs, calm = 1.0) {
    super();
    this.index = index;
    this.calm = calm;
    const k = name => knobs[name] ?? 0.5;
    const girth = 0.85 + 0.3 * k("appetite");
    const size = 0.9 + 0.25 * k("aggressiveness") - 0.2 * k("timidity");
    const inks = { cream: Ink.CREAM, navy: Ink.STONE, coral: Ink.CORAL, mustard: Ink.MUSTARD, stone: RIBBONS[index % RIBBONS.length] };
    this.tumbler = new THREE.Group();
    this.model = new THREE.Group();
    this.add(this.tumbler);
    this.tumbler.add(this.model);
    this.model.scale.set(LOOK * size, LOOK * size, girth * LOOK * size);
    this.baseScale = this.model.scale.x;
    this.rig = new THREE.Group();
    this.rig.rotation.x = -Math.PI / 2;  // MuJoCo: z up, y left
    this.rig.position.y = robot.stand_z;
    this.model.add(this.rig);
    this.trunk = new THREE.Group();
    this.rig.add(this.trunk);
    this.hinges = {};
    this.frames = {};
    for (const body of robot.bodies) {
      const node = new THREE.Group();
      if (body.parent === "world") this.trunk.add(node);
      else {
        node.position.set(...body.pos);
        node.quaternion.set(body.quat[1], body.quat[2], body.quat[3], body.quat[0]);
        this.frames[body.parent].add(node);
      }
      const turns = new THREE.Group();
      node.add(turns);
      this.frames[body.name] = turns;
      if (body.joint) this.hinges[body.joint.name] = [turns, new THREE.Vector3(...body.joint.axis).normalize()];
      for (const p of body.parts) {
        const shell = p.ink === "cream";
        Ink.part(turns, p.geometry, inks[p.ink], [0, 0, 0], 1, { cell: CELL, lift: 0.25, outlined: shell, grow: 0.0025 });
      }
    }
    this.head = this.frames[robot.hat.body];
    this.frames.neck_pitch.scale.setScalar(0.9 + 0.3 * k("vanity"));
    const foot = 0.1 * LOOK * size * girth;
    this.shadow = Ink.part(this, Ink.cone(foot, 0.001, foot, 12), Ink.GRASS, [0, 0.003, 0], 1, { tone: 0.35 });
    const ringGeo = new THREE.TorusGeometry(0.18 * LOOK, 0.01 * LOOK, 5, 10);
    ringGeo.rotateX(Math.PI / 2);
    this.ring = Ink.part(this, ringGeo, Ink.CORAL, [0, 0.004, 0], [1, 0.2, 1], { tone: 1.0 });
    this.ring.visible = false;
    this.mood = new THREE.Group();
    this.mood.position.y = this.moodAt = 0.48 * LOOK * size;  // just over the head
    this.add(this.mood);
    const face = mood => { const f = faceModel(mood, 0.075); this.mood.add(f); return f; };  // turned to the camera by the page
    this.moodShapes = { ball: face("joy"), spike: face("fear"), block: face("anger"), drop: face("sorrow") };
    this.signAt = this.mood.position.y + 0.12;
    this.tears = [0, 1].map(() => { const t = Ink.part(this, Ink.ball(0.012, 6), new THREE.Color("#4f9fe0"), [0, 0, 0], [1, 1.5, 1], { cell: CELL, tone: 0.9 }); t.visible = false; return t; });
    this.hat = null; this.hatStyle = -1;
    this.state = null;
    this.target = new THREE.Vector3();
    this.speed = 0; this.emote = ""; this.emoteAge = 99; this.emoteSeen = -1;
    this.sway = 0; this.hop = 0; this.lifted = false; this.headingWas = 0;
    this.clip = "stand"; this.clipT = 0;
    this.downWas = false; this.tumbleT = 99; this.tumbleYaw = 0; this.tumbleHop = 0;
    this.sign = "";  // the word over its head, for the page to draw
    this.zs = [];  // its z's while asleep: [local position, text, size, opacity]
  }

  showState(s, first) {
    const moved = new THREE.Vector3(s.x, 0, -s.y).sub(this.target);
    this.target.set(s.x, 0, -s.y);
    this.state = s;
    if (s.down && !this.downWas && !first) this._startTumble(moved, s.h);
    this.downWas = s.down;
    if (first) { this.position.copy(this.target); this.rotation.y = s.h; }
    if (s.emote && s.emote_t !== this.emoteSeen) { this.emote = s.emote; this.emoteAge = 0; this.emoteSeen = s.emote_t; }
  }

  update(dt, now) {
    const s = this.state;
    if (!s) return;
    const before = this.position.clone();
    this.position.lerp(this.target, Math.min(1, 12 * dt));
    this.rotation.y = lerpAngle(this.rotation.y, s.h, Math.min(1, 10 * dt));
    this.speed = lerp(this.speed, before.distanceTo(this.position) / Math.max(dt, 1e-4), Math.min(1, 6 * dt));
    this.emoteAge += dt;
    const t = now + this.index;
    const angle = this._clipPose(dt, t);
    this.headingWas = this.rotation.y;
    for (const leg of ["yaw2roll", "bearing_roll"]) this.frames[leg].visible = !s.swimming;
    this.shadow.visible = !s.swimming;
    this._wear();
    this._nod(angle, t);
    for (const [name, [node, axis]] of Object.entries(this.hinges)) {
      node.quaternion.slerp(new THREE.Quaternion().setFromAxisAngle(axis, angle[name] ?? 0), Math.min(1, 9 * dt));
    }
    this._actOut(dt);
    this._signs(dt, t);
    this._tumble(dt);
  }

  _play(want, dt) {
    const hz = clips.hz;
    let frames = clips.clips[this.clip].frames;
    let length = (frames.length - 1) / hz;
    const finishing = this.clip in ONCE && this.clipT < length && want !== "go_limp";
    if (want === "go_limp") {
      // down for a set time: get-up is placed by the time left, so the duck is standing when it is freed
      const up = clips.clips.get_up.frames.length - 1;
      const rise = up / hz / ONCE.get_up, left = this.state.down_left ?? 0;
      if (left < rise) {
        this.clip = "get_up";
        this.clipT = Math.max(up / hz - left * ONCE.get_up, 0);
        return this._poseAt(this.clip, this.clipT);
      }
    }
    if (!finishing) {
      const sat = this.clip === "sitting" || this.clip === "sit_down";
      let next = want;
      if (want === "sitting" && !sat) next = "sit_down";
      else if (sat && !(want === "sitting" || want === "go_limp")) next = "stand_up";
      else if (this.clip === "go_limp" && want !== "go_limp") next = "get_up";
      if (next !== this.clip) {
        this.clip = next;
        this.clipT = 0;
        frames = clips.clips[next].frames;
        length = (frames.length - 1) / hz;
      }
    }
    const rate = ONCE[this.clip] ?? (this.clip === "walk" ? clamp(this.speed / 0.12, 0.6, 2.5) : 1.0);
    this.clipT += dt * rate * (0.5 + 0.5 * this.calm);
    this.clipT = clips.clips[this.clip].loop ? fmod(this.clipT, length) : Math.min(this.clipT, length);
    return this._poseAt(this.clip, this.clipT);
  }

  _poseAt(which, seconds) {
    const frames = clips.clips[which].frames;
    const at = clamp(seconds * clips.hz, 0, frames.length - 1);
    const a = frames[Math.trunc(at)], b = frames[Math.min(Math.trunc(at) + 1, frames.length - 1)], u = at - Math.trunc(at);
    const qa = new THREE.Quaternion(a[2][1], a[2][2], a[2][3], a[2][0]).normalize();
    const qb = new THREE.Quaternion(b[2][1], b[2][2], b[2][3], b[2][0]).normalize();
    return [a[0].map((j, k) => lerp(j, b[0][k], u)), lerp(a[1], b[1], u), qa.slerp(qb, u)];
  }

  _startTumble(moved, heading) {
    const shoved = moved.length() > 0.05;
    const way = shoved ? moved.clone().normalize() : new THREE.Vector3(Math.cos(heading), 0, -Math.sin(heading));
    way.applyAxisAngle(UP, (Math.random() * 2 - 1) * 0.35);
    const falls = robot.fallWay.clone().applyAxisAngle(UP, this.rotation.y);
    this.tumbleYaw = Math.atan2(new THREE.Vector3().crossVectors(falls, way).y, falls.dot(way));
    const force = shoved ? clamp(moved.length() / PUSH_M, 1, 2) : 0.6;
    this.tumbleHop = (0.02 + Math.random() * 0.02) * force * LOOK;
    this.tumbleT = 0;
  }

  _tumble(dt) {
    this.tumbleT += dt;
    const down = this.state.down;
    const yaw = lerpAngle(this.tumbler.rotation.y, down ? this.tumbleYaw : 0, Math.min(1, (down ? 16 : 2.5) * dt));
    const u = clamp(this.tumbleT / KNOCKED_S, 0, 1);
    this.tumbler.rotation.set(0, yaw, 0);
    this.tumbler.position.set(0, this.tumbleHop * 4 * u * (1 - u), 0);
  }

  _wantedClip(dt) {
    const s = this.state;
    const actingUp = this.emoteAge < EMOTE_S && ["dance", "singdance", "happy", "playful", "stomp"].includes(this.emote);
    const low = (s.sat && !actingUp) || s.asleep;
    const yawRate = angleDifference(this.headingWas, this.rotation.y) / Math.max(dt, 1e-4);
    if (s.down) return "go_limp";
    if (low || s.swimming) return "sitting";
    if (s.kicking) return "kick";
    if (s.eating || s.drumming) return "peck";  // an instrument is played with the beak
    if (this.emote === "playful" && this.emoteAge < 0.4) return "roll";
    if (this.speed > 0.015) return "walk";
    if (Math.abs(yawRate) > 0.4) return yawRate > 0 ? "turn_left" : "turn_right";
    return "stand";
  }

  _clipPose(dt, t) {
    const angle = { ...robot.stand };
    const [joints, height, q] = this._play(this._wantedClip(dt), dt);
    WIRE.forEach((name, k) => { angle[name] = joints[k]; });
    this.rig.position.y = lerp(this.rig.position.y, height, Math.min(1, 14 * dt));
    this.trunk.quaternion.slerp(q, Math.min(1, 14 * dt));
    if ((this.emote === "dance" || this.emote === "singdance") && this.emoteAge < EMOTE_S) {  // no recorded dance: step in place
      const step = Math.sin(this.emoteAge * TAU / 0.7) * 0.3;
      for (const side of ["left", "right"]) {
        angle[`${side}_hip_pitch`] = (angle[`${side}_hip_pitch`] ?? 0) + step;
        angle[`${side}_ankle`] = (angle[`${side}_ankle`] ?? 0) - step;
      }
    }
    const s = this.state;
    const afloat = s.swimming ? -0.05 * this.model.scale.y : (this.lifted ? 0.32 : 0.0);
    this.model.position.y = lerp(this.model.position.y, afloat + this.hop * LOOK * this.calm, Math.min(1, 14 * dt));
    this.model.rotation.x = lerp(this.model.rotation.x, (s.swimming ? Math.sin(t * 1.7) * 0.05 * this.calm : 0) + this.sway * this.calm, Math.min(1, 8 * dt));
    return angle;
  }

  _wear() {
    const s = this.state, wearing = s.hat ? s.hat_style : -1;
    if (wearing === this.hatStyle) return;
    if (this.hat) { this.head.remove(this.hat); this.hat = null; }
    if (wearing >= 0) {
      this.hat = Hats.make(wearing);
      const seat = robot.hat, up = new THREE.Vector3(...seat.up), forward = new THREE.Vector3(...seat.forward);
      const basis = new THREE.Matrix4().makeBasis(forward, up, new THREE.Vector3().crossVectors(forward, up));
      this.hat.quaternion.setFromRotationMatrix(basis);
      this.hat.scale.setScalar(1.25);
      this.hat.position.set(...seat.at);
      this.head.add(this.hat);
    }
    this.hatStyle = wearing;
  }

  _nod(angle, t) {
    const s = this.state, told = s.head || [0, 0, 0, 0], c = this.calm;
    let nod = [told[0], told[1] + Math.sin(t * 0.9) * 0.03 * c, told[2] + Math.sin(t * 0.37) * 0.1 * c, told[3]];
    if (s.asleep) nod = [0.45, 0.6, 0.9, 0.0];
    else if (s.eating) nod = [0.45 + Math.sin(t * 9.0) * 0.2, 0.5, 0.0, 0.0];
    else if (s.drumming) nod = [0.3 + Math.abs(Math.sin(t * 14.0)) * 0.3, 0.4, 0.0, 0.0];  // quick taps
    ["neck_pitch", "head_pitch", "head_yaw", "head_roll"].forEach((name, k) => { angle[name] = (angle[name] ?? 0) + nod[k]; });
  }

  _actOut(dt) {
    let squash = 0;
    this.sway = 0; this.hop = 0;
    const age = this.emoteAge;
    if (age < EMOTE_S && !this.state.asleep) {
      const beat = Math.sin(age / EMOTE_S * TAU * 3);
      switch (this.emote) {
        case "happy": this.hop = Math.abs(beat) * 0.04; squash = -Math.abs(beat) * 0.08; break;
        case "playful": this.hop = Math.abs(beat) * 0.06; this.model.rotation.y = age / EMOTE_S * TAU; break;
        case "scared": squash = 0.18; break;
        case "proud": squash = -0.1; break;
        case "stomp": this.hop = Math.max(beat, 0) * 0.025; squash = Math.max(-beat, 0) * 0.1; break;
        case "yawn": squash = -0.07 * Math.sin(age / EMOTE_S * Math.PI); break;
        case "splash": this.hop = Math.abs(beat) * 0.03; squash = -Math.abs(beat) * 0.06; break;
        case "sing": this.sway = beat * 0.08; break;
        case "cower": squash = 0.24; break;
        case "dance": case "singdance": {
          const bar = Math.sin(age * TAU / 0.7);
          this.hop = Math.abs(bar) * 0.02;
          squash = Math.max(-bar, 0) * 0.1;
          this.sway = Math.sin(age * TAU / 1.4) * 0.12;
          this.model.rotation.y = Math.sin(age * TAU / 1.4) * 0.5;
          break;
        }
      }
    }
    if (age >= EMOTE_S || !["playful", "dance", "singdance"].includes(this.emote)) {
      this.model.rotation.y = lerpAngle(this.model.rotation.y, 0, Math.min(1, 8 * dt));
    }
    const sx = this.model.scale.x;
    this.model.scale.y = lerp(this.model.scale.y, sx * (1 - squash * this.calm), Math.min(1, 12 * dt));
  }

  _signs(dt, t) {
    const s = this.state, shape = MOOD_SHAPES[s.mood] ?? "";
    for (const [name, mesh] of Object.entries(this.moodShapes)) mesh.visible = name === shape && s.strength > 0.25 && !s.asleep;
    this.mood.scale.setScalar((0.6 + 0.9 * s.strength) * LOOK);
    const shaking = shape === "spike" || shape === "block";  // a scared face trembles and an angry one shakes
    this.mood.position.x = shaking ? Math.sin(t * (shape === "spike" ? 40 : 18)) * 0.006 * this.calm : 0;
    this.mood.position.y = this.moodAt + (shape === "ball" ? Math.abs(Math.sin(t * 3)) * 0.012 * this.calm : 0);  // joy bobs
    this.sign = s.asleep ? "" : (this.emoteAge < EMOTE_S + 0.6 ? DOES[this.emote] ?? this.emote : "");
    this.tears.forEach((tear, k) => {
      tear.visible = !!s.crying;
      const fall = fmod(t * 1.4 + 0.5 * k, 1);
      tear.position.set(0.05 * LOOK, (0.26 - 0.22 * fall) * LOOK, (k === 0 ? 0.06 : -0.06) * LOOK);
    });
    this.zs = [];
    if (s.asleep) {
      for (let k = 0; k < 4; k++) {
        const u = fmod(t * 0.3 * (0.5 + 0.5 * this.calm) + k / 4, 1);
        const at = new THREE.Vector3(0.04 + 0.12 * u + Math.sin(u * 7) * 0.02, 0.2 + 0.3 * u, 0).multiplyScalar(LOOK);
        at.y = 0.2 * LOOK + 0.3 * u * LOOK;
        this.zs.push([at, u < 0.45 ? "z" : "Z", 0.5 + 1.3 * u, Math.min(1, u * 6) * Math.sqrt(1 - u)]);
      }
    }
  }
}
