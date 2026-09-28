// The garden in one object: the body (stub.js), the brain server (server.js) and the snapshot a viewer draws
// (snapshot.js), stepped together as body/stub2d/stub.py's run loop steps them. `brains` are one per duck, each
// with an async step(message): web/worker.js behind a proxy on the page, or web/brain.js directly in a test.
import { Stub } from "./stub.js";
import { BrainServer } from "./server.js";
import { Snapshot } from "./snapshot.js";
import { WHEEL_VX, WHEEL_VYAW } from "./stub.js";

export const WHEEL_S = 0.5;  // a wheel not held for this long is let go

export class Garden {
  constructor({ n, seed = 0, garden = {}, pose = null, personality = {}, brains, sizes, learns = true, eyes = true, hunger = 0.5, thirst = 0.5 }) {
    this.stub = new Stub(n, seed, { ...garden, pose, eyes });
    this.server = new BrainServer({ stub: this.stub, brains, sizes, seed, personality, learns, eyes, hunger, thirst });
    this.snap = new Snapshot();
    this.wheel = { duck: -1, fwd: 0, turn: 0, at: -Infinity };
    this.riding = -1;
  }

  // One 20 ms step: the body moves, senses, and the brains answer.
  async step() {
    this.stub.step();
    const riding = this.wheel.duck >= 0;
    const f = this.stub.send_frames(this.server.eyes || riding);
    const intents = await this.server.step(f);
    this._hold_wheel();
    return intents;
  }

  snapshot() {
    const s = this.snap.build(this.stub, this.server, this.server.watched);
    if (this.riding >= 0) s.ride = this.snap.ride(this.stub, this.server, this.riding);
    if (this.server.watched >= 0) s.brain = { duck: this.server.watched, spikes: this.server.take_view() };
    return s;
  }

  // The player's calls (garden.*), plus the viewer's own: garden.watch {duck} and garden.wheel {duck, fwd, turn}.
  act(method, p = {}) {
    if (method === "garden.watch") { this.server.watched = p.duck >= 0 && p.duck < this.stub.n ? p.duck : -1; return {}; }
    if (method === "garden.wheel") { this.wheel = { duck: p.duck, fwd: Math.max(-1, Math.min(1, p.fwd)), turn: Math.max(-1, Math.min(1, p.turn)), at: this.stub.t }; this._hold_wheel(); return {}; }
    return this.stub.control(method, p);
  }

  _hold_wheel() {
    const { duck, fwd, turn, at } = this.wheel;
    const held = duck >= 0 && duck < this.stub.n && this.stub.t - at < WHEEL_S;
    if (this.riding >= 0 && (!held || duck !== this.riding)) this.server.possessed[this.riding] = false;
    this.riding = held ? duck : -1;
    if (held) {
      this.server.possessed[duck] = true;
      this.stub.robot(duck, "robot.move", { vx: WHEEL_VX * fwd, vy: 0, vyaw: WHEEL_VYAW * turn });
    } else this.wheel = { duck: -1, fwd: 0, turn: 0, at: -Infinity };
  }
}
