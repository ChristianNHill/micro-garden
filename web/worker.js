// One duck's brain on its own thread. The page sends a job and gets the result back.
//   {job: "reference"}         run web/pack.py's reference schedule; spikes per tick and per neuron
//   {job: "bench", ticks}      run with food smelled on one side; milliseconds per tick
import { LIF, parseBrain } from "./lif.js";

const loaded = Promise.all([
  fetch("data/brain.bin").then(r => r.arrayBuffer()).then(parseBrain),
  fetch("data/sets.json").then(r => r.json()),
]);

function newBrain(brain, info) {
  const lif = new LIF(brain);
  for (const i of info.sets.kenyon_cells) lif.threshOffset[i] = info.kc_threshold;
  return lif;
}

// {set: release} -> the graded argument of LIF.step, sets in the order given.
function drive(info, release) {
  const idx = [], rel = [];
  for (const [name, r] of Object.entries(release)) for (const i of info.sets[name]) { idx.push(i); rel.push(r); }
  return idx.length ? { idx: Int32Array.from(idx), rel: Float32Array.from(rel) } : null;
}

self.onmessage = async ({ data }) => {
  const [brain, info] = await loaded;
  const lif = newBrain(brain, info);
  if (data.job === "reference") {
    const perTick = [];
    for (const seg of info.reference) {
      const g = drive(info, seg.release);
      for (let t = seg.from; t < seg.to; t++) perTick.push(lif.step(g));
    }
    self.postMessage({ perTick: Uint32Array.from(perTick), counts: lif.nSpikes });
  } else if (data.job === "bench") {
    const g = drive(info, { orn_food_left: 0.4, orn_food_right: 0.1, jo_push_left: 0.25 });
    let spikes = 0;
    const t0 = performance.now();
    for (let t = 0; t < data.ticks; t++) spikes += lif.step(g);
    const ms = performance.now() - t0;
    self.postMessage({ msPerTick: ms / data.ticks, spikesPerTick: spikes / data.ticks });
  }
};
