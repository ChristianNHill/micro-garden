// One duck's brain on its own thread (web/brain.js). Each message is {id, job, ...}; the reply is {id, ok} or {id, error}.
//   init {brain, eyes, seed, smarts, learns}   the gzipped bundles' bytes; eyes null for a blind duck
//   step {levels, lum, gain, reward, punish, watch}
//   save / load {w, slow} / rest {s} / bench {steps}
import { DuckBrain, bench } from "./brain.js";

let duck = null;

const jobs = {
  async init(m) {
    duck = await DuckBrain.create(m.brain, m.eyes, m);
    return { neurons: duck.meta.neurons, groups: duck.nGroups, eyes: !!duck.eyes };
  },
  step: m => duck.step(m),
  save: () => duck.save(),
  load: m => duck.load(m),
  rest: m => duck.rest(m.s),
  bench: m => bench(duck, m.steps),
};

self.onmessage = async ({ data }) => {
  try {
    self.postMessage({ id: data.id, ok: await jobs[data.job](data) });
  } catch (e) {
    self.postMessage({ id: data.id, error: String((e && e.stack) || e) });
  }
};
