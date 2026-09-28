// The browser garden: five ducks, each on its own copy of the fly brain in a worker, stepped at real time, and drawn
// the way viewer/godot/main.gd draws the native garden. The garden only runs; this only draws and passes on what the
// visitor does.
import * as THREE from "three";
import * as Ink from "./ink.js";
import * as Scenery from "./scenery.js";
import * as Props from "./props.js";
import * as Hats from "./hats.js";
import { Duck, LOOK, setData, lerpAngle } from "./duck.js";
import { Panels } from "./ui.js";
import { Sound } from "./audio.js";
import { SpotifyBox } from "./spotify.js";
import { parseBundle } from "../bundle.js";
import { Garden } from "../garden/garden.js";
import { DEMO_GARDEN } from "../garden/stub.js";
import { HATCHABLE, preset, stack } from "../garden/personality.js";
import { Rng } from "../garden/rng.js";
import * as Save from "../garden/save.js";

const DUCKS = 5;
const STEP_MS = 20;
const PITCH = [0.35, 1.25];
const DRIFT = 0.12;
const DUCK_TALL = 0.42;
const KICKED_M = 0.3, KICKED_S = 0.7;
const SAVE_EVERY_S = 60;
const TAU = 2 * Math.PI;
const params = new URLSearchParams(location.search);
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0.5 : 1.0;

const $ = id => document.getElementById(id);
// The garden's clock runs in a worker: a page's own timers slow to about once a second while its tab is hidden, and
// the ducks should go on living when nobody is looking.
const sleep = (() => {
  let clock = null, waiting = new Map(), next = 0;
  try {
    clock = new Worker(URL.createObjectURL(new Blob(["onmessage = e => setTimeout(() => postMessage(e.data.id), e.data.ms)"], { type: "text/javascript" })));
    clock.onmessage = e => { waiting.get(e.data)(); waiting.delete(e.data); };
  } catch { clock = null; }
  return ms => new Promise(resolve => {
    if (!clock) { setTimeout(resolve, ms); return; }
    const id = ++next;
    waiting.set(id, resolve);
    clock.postMessage({ id, ms: Math.max(0, ms) });
  });
})();

// ---------- loading ----------
async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  if (!res.body || !onBytes) return res.arrayBuffer();
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onBytes(got);
  }
  const out = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.length; }
  return out.buffer;
}

class BrainProxy {
  constructor() {
    this.worker = new Worker(new URL("../worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.id = 0;
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      this.pending.delete(data.id);
      if (data.error) p.reject(new Error(data.error)); else p.resolve(data.ok);
    };
    this.worker.onerror = e => { for (const p of this.pending.values()) p.reject(new Error(e.message)); this.pending.clear(); };
  }
  call(job, msg = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => { this.pending.set(id, { resolve, reject }); this.worker.postMessage({ ...msg, id, job }); });
  }
  step(m) { return this.call("step", m); }
}

async function boot() {
  const loading = $("loading"), bar = $("loading-bar"), line = $("loading-line");
  // A tap while it loads counts: the garden opens as soon as it is ready. Browsers only allow sound after one.
  let tapped = null;
  const entered = new Promise(resolve => { tapped = resolve; });
  const tap = () => { removeEventListener("pointerdown", tap); removeEventListener("keydown", tap); tapped(); };
  addEventListener("pointerdown", tap);
  addEventListener("keydown", tap);
  const show = (text, frac) => { line.textContent = text; if (frac !== undefined) bar.style.width = `${Math.round(frac * 100)}%`; };
  const eyes = true;  // every duck sees
  const sizes = { brain: 8.5e6, eyes: 0.75e6 }, got = { brain: 0, eyes: 0 };
  const progress = () => show(`Loading the fly brain: ${((got.brain + got.eyes) / 1e6).toFixed(1)} of ${((sizes.brain + (eyes ? sizes.eyes : 0)) / 1e6).toFixed(1)} MB`,
                              (got.brain + got.eyes) / (sizes.brain + (eyes ? sizes.eyes : 0)));
  const [brainGz, eyesGz, robotJson, clipsJson] = await Promise.all([
    fetchWithProgress("data/brain.bin.gz", n => { got.brain = n; progress(); }),
    eyes ? fetchWithProgress("data/eyes.bin.gz", n => { got.eyes = n; progress(); }) : null,
    fetch("data/robot.json").then(r => r.json()), fetch("data/clips.json").then(r => r.json()),
  ]);
  show("Unpacking 139,248 neurons and 2.7 million connections", 1);
  const brain = await parseBundle(brainGz);
  setData(robotJson, clipsJson);
  const groupSizes = Array.from({ length: brain.meta.decoder_groups }, () => 0);
  for (const g of brain.arrays["decoder.group"]) groupSizes[g]++;

  // the ducks: the saved garden, or five new ones, no two the same personality
  const state = params.has("fresh") ? null : Save.saved();
  const hatch = new Rng((Math.random() * 2 ** 32) >>> 0);
  const labels = hatch.sample(HATCHABLE, DUCKS);
  const personality = stack(labels.map(l => preset(l, hatch)));
  const seed = hatch.integers(1_000_000);
  show(`Waking ${DUCKS} ducks, one brain each`, 1);
  const brains = Array.from({ length: DUCKS }, () => new BrainProxy());
  await Promise.all(brains.map((b, i) => b.call("init", { brain: brainGz, eyes: eyesGz, seed: seed + 101 * i,
                                                          smarts: personality.smarts[i], learns: true })));
  const garden = new Garden({ n: DUCKS, seed, garden: DEMO_GARDEN, personality, brains, sizes: groupSizes, learns: true, eyes });
  let welcome = "";
  if (state) {
    const [gap, weights] = Save.load(state, garden.server.body, garden.stub);
    garden.server.decoder.stink_affinity = [...garden.server.body.k.stink_affinity];
    if (weights) await Promise.all(brains.map((b, i) => b.call("load", weights[i]).then(() => b.call("rest", { s: gap }))));
    welcome = gap > 90 ? `welcome back: ${Math.round(gap / 60)} minutes away` : "welcome back";
  }
  loading.classList.add("ready");
  show("Tap to enter the garden", 1);
  await entered;
  loading.hidden = true;
  run({ garden, brain, brains, eyes, welcome });
}

// ---------- the running garden ----------
function run({ garden, brain, brains, eyes, welcome }) {
  const canvas = $("garden");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 200);
  const panels = new Panels($("panels"));
  panels.setBrainPoints(brain.arrays["view.xy"], brain.arrays["view.group"], brain.meta.view_groups, brain.meta.neurons);
  const sound = new Sound();
  sound.start();
  const playlist = new SpotifyBox(document.body);
  if (welcome) panels.say(welcome);
  let pixelRatio = Math.min(devicePixelRatio || 1, 2);
  const resize = () => {
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(innerWidth, innerHeight, false);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
  };
  addEventListener("resize", resize);
  resize();

  let snap = garden.snapshot();
  const size = snap.size;
  const orbit = { yaw: 2.2, pitch: 0.5, dist: 1.7 * size };
  // a narrow screen stands further back, so the whole garden fits across it
  orbit.dist *= Math.min(Math.max(1, 0.9 * innerHeight / innerWidth), 2.2);
  const farthest = Math.max(12, orbit.dist);
  const eye0 = new THREE.Vector3(size / 2, 0, -size / 2).add(new THREE.Vector3(Math.cos(orbit.yaw), 0, Math.sin(orbit.yaw)).multiplyScalar(orbit.dist));
  const setting = Scenery.build(scene, snap, eye0.toArray());
  const flags = setting.summits.filter(Boolean).map(s => Props.flag(scene, s));
  const ducks = snap.ducks.map((d, i) => { const duck = new Duck(i, d.knobs, calm); scene.add(duck); return duck; });
  const glove = Props.glove();
  scene.add(glove);
  const props = {};
  let selected = -1, possessing = false, handMode = false, helpOn = false, still = 0, centre = null;
  let first = true;

  // the garden runs at real time on its own loop; drawing follows the newest snapshot
  let running = true, simSteps = 0, simSince = performance.now(), speed = 1;
  let lastSave = performance.now();
  const step = async () => {
    let next = performance.now();
    while (running) {
      await garden.step();
      snap = garden.snapshot();
      simSteps++;
      next += STEP_MS;
      const lag = next - performance.now();
      if (lag > 0) await sleep(lag);
      else { if (lag < -250) next = performance.now(); await sleep(0); }
      const now = performance.now();
      if (now - simSince > 3000) { speed = simSteps * STEP_MS / (now - simSince); simSteps = 0; simSince = now; }
      if (now - lastSave > SAVE_EVERY_S * 1000) { lastSave = now; save(); }
    }
  };
  step().catch(e => { panels.status.textContent = `the garden stopped: ${e.message}`; console.error(e); });
  // A new garden throws the save away; nothing may write it back on the way out.
  let resetting = false, lastWeights = null;
  const save = async () => {
    if (resetting) return;
    try {
      lastWeights = await Promise.all(brains.map(b => b.call("save")));
      if (!resetting) Save.save(garden.server.body, lastWeights, garden.stub);
    } catch { /* storage refused; the garden goes on */ }
  };
  addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") save(); });
  // the weights from the last save: asking the brains for fresh ones takes longer than a closing page waits
  addEventListener("pagehide", () => { if (!resetting) Save.save(garden.server.body, lastWeights, garden.stub); });

  const act = (method, p = {}) => garden.act(method, p);

  // ---------- drawing the snapshot ----------
  const syncProps = (kind, items, make) => {
    let nodes = props[kind] || [];
    if (nodes.length !== items.length) {
      nodes.forEach(n => scene.remove(n));
      nodes = items.map(item => { const n = new THREE.Group(); n.position.set(item[0], 0, -item[1]); scene.add(n); make(n, item); n.userData.key = item[2]; return n; });
      props[kind] = nodes;
    }
    items.forEach((item, i) => {
      const n = nodes[i], to = new THREE.Vector3(item[0], n.position.y, -item[1]);
      const carried = snap.held && snap.held[0] === "food" && snap.held[1] === i;
      if (kind === "food" && !carried && !n.userData.kicked && n.position.distanceTo(to) > KICKED_M) n.userData.kicked = { from: n.position.clone(), t: 0, to };
      if (n.userData.kicked) n.userData.kicked.to = to;
      else n.position.copy(to);
    });
    return nodes;
  };
  // the oldest goes when one too many is put down, so the count can stay while the kinds change
  const rebuildIfChanged = (kind, items, make) => {
    (props[kind] || []).forEach((n, i) => {
      if (n.userData.key !== items[i][2]) { n.clear(); make(n, items[i]); }
      n.userData.key = items[i][2];
    });
  };

  const show = dt => {
    Ink.globals.daylight.value = snap.light;
    const day = snap.day * TAU;
    Ink.globals.sun_dir.value.set(Math.cos(day), 0.35 + 0.65 * snap.light, 0.5 * Math.sin(day)).normalize();
    scene.background = Scenery.SKY_NIGHT.clone().lerp(Scenery.SKY_DAY, snap.light);
    snap.ducks.forEach((d, i) => {
      ducks[i].showState(d, first);
      ducks[i].ring.visible = i === selected && !possessing;
      ducks[i].visible = !(possessing && i === selected);
      ducks[i].lifted = !!snap.held && snap.held[0] === "duck" && snap.held[1] === i;
    });
    first = false;
    syncProps("food", snap.food, Props.fruit);
    syncProps("hats", snap.hats, (n, h) => { const lying = Hats.make(h[2]); lying.scale.setScalar(1.9); n.add(lying); });
    rebuildIfChanged("hats", snap.hats, (n, h) => { const lying = Hats.make(h[2]); lying.scale.setScalar(1.9); n.add(lying); });
    syncProps("instruments", snap.instruments, Props.instrument);
    rebuildIfChanged("instruments", snap.instruments, Props.instrument);
    syncProps("balls", snap.balls, Props.ballProp);
    rebuildIfChanged("balls", snap.balls, Props.ballProp);
    for (const n of props.balls || []) {  // roll by the distance moved
      const was = n.userData.was || n.position.clone(), moved = n.position.clone().sub(was);
      n.userData.was = n.position.clone();
      if (moved.length() > 1e-5 && n.userData.ball) n.userData.ball.rotateOnWorldAxis(new THREE.Vector3(moved.z, 0, -moved.x).normalize(), moved.length() / (0.06 * 1.9));
    }
    syncProps("drum", snap.drum ? [snap.drum] : [], Props.drum);
    syncProps("danger", snap.danger, Props.stink);
    const gust = snap.wind, drift = gust ? new THREE.Vector3(gust[0], 0, -gust[1]).multiplyScalar(0.35) : new THREE.Vector3();
    const tnow = performance.now();
    for (const n of props.danger || []) {
      n.userData.puffs.forEach((puff, k) => {
        const strand = k % 3, up = ((tnow / 2600 * calm + (k + 1) / Props.STINK_PUFFS) % 1);
        const swing = Math.sin(up * TAU + strand * 2.1) * 0.07, a = TAU * strand / 3 + 0.4;
        puff.position.set(Math.cos(a) * 0.08 + swing, 0.04 + up * 0.55, Math.sin(a) * 0.08 + swing * 0.6).addScaledVector(drift, up * up);
        puff.scale.setScalar(1 - 0.75 * up);
      });
    }
    syncProps("music", snap.music ? [snap.music] : [], Props.musicBox);
    for (const n of props.music || []) n.scale.setScalar(1 + 0.06 * calm * Math.sin(tnow / 90));
    // the wind: flags turn and flutter, reeds lean, the falls pulse
    const blowing = gust && Math.hypot(gust[0], gust[1]) > 0.05;
    flags.forEach((flag, i) => {
      const flutter = Math.sin(tnow / 160 + i * 0.9) * 0.14 * calm;
      flag.rotation.order = "YXZ";
      flag.rotation.y = lerpAngle(flag.rotation.y, (blowing ? Math.atan2(gust[1], gust[0]) : flag.rotation.y) + flutter, 0.08);
      flag.rotation.z += ((blowing ? 0 : -1.35) - flag.rotation.z) * 0.05;
    });
    setting.falls.forEach((f, i) => { f.scale.z = 0.5 * (1 + 0.12 * calm * Math.sin(tnow / 130 + i * 1.7)); });
    const strength = blowing ? Math.min(Math.hypot(gust[0], gust[1]), 1) : 0;
    const toward = blowing ? new THREE.Vector3(gust[0], 0, -gust[1]).normalize() : new THREE.Vector3(0, 0, -1);
    const axis = new THREE.Vector3(toward.z, 0, -toward.x).normalize();
    setting.reeds.forEach((reed, i) => {
      const lean = calm * (0.2 * strength + 0.04) * (0.75 + 0.25 * Math.sin(tnow / 520 + i * 0.8));
      reed.quaternion.slerp(new THREE.Quaternion().setFromAxisAngle(axis, lean), 0.08);
    });
    // lift whatever the hand holds
    const held = snap.held;
    for (const kind of ["balls", "hats", "food", "music", "drum", "instruments"]) {
      (props[kind] || []).forEach((n, k) => {
        const up = held && `${held[0]}${["ball", "hat", "instrument"].includes(held[0]) ? "s" : ""}` === kind && held[1] === k;
        n.position.y += ((up ? 0.3 : 0) - n.position.y) * 0.3;
      });
    }
    // sounds: [t, duck, tag], each played once
    for (const [t, duck, tag] of snap.sounds) {
      if (t > (show.heard ?? -1)) { show.heard = t; sound.quack(duck, tag, snap.ducks[duck]?.tune ?? 1); }
    }
    sound.pond(!!snap.pond, snap.light);
    // the Spotify playlist, or the box's own tune if Spotify will not load
    const useSpotify = !playlist.failed;
    playlist.update(useSpotify && !!snap.music, snap.music_volume * (sound.on ? 1 : 0));
    sound.music(!useSpotify && !!snap.music, snap.music_volume);
  };

  const rollKicked = dt => {
    for (const n of props.food || []) {
      const roll = n.userData.kicked;
      if (!roll) continue;
      roll.t += dt;
      const u = Math.min(roll.t / KICKED_S, 1), along = roll.to.clone().sub(roll.from);
      n.position.copy(roll.from).addScaledVector(along, 1 - (1 - u) ** 2);
      n.position.y += 0.22 * Math.abs(Math.sin(u * TAU)) * (1 - u);
      if (along.length() > 0.01) n.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0).cross(along).normalize(), -dt * 14 * (1 - u));
      if (u >= 1) delete n.userData.kicked;
    }
  };

  // ---------- the camera ----------
  const cameraTo = dt => {
    still += dt;
    if (possessing && selected >= 0) {
      const d = ducks[selected];
      camera.fov = 100;
      Ink.globals.screen.value = 0.25;
      renderer.setPixelRatio(0.45 * pixelRatio);
      camera.position.copy(d.position).add(new THREE.Vector3(0, 0.3, 0));
      camera.rotation.set(-0.15, d.rotation.y - Math.PI / 2, 0, "YXZ");
      camera.updateProjectionMatrix();
      return;
    }
    if (camera.fov !== 38) { camera.fov = 38; camera.updateProjectionMatrix(); renderer.setPixelRatio(pixelRatio); }
    Ink.globals.screen.value = 1.0;
    const sway = Math.sin(performance.now() / 9000) * DRIFT * (calm === 1 ? 1 : 0) * Math.min(Math.max(still - 3, 0), 1);
    const yaw = orbit.yaw + sway;
    const want = selected >= 0 ? ducks[selected].position.clone().add(new THREE.Vector3(0, 0.15, 0)) : new THREE.Vector3(size / 2, 0.1, -size / 2);
    centre = centre ? centre.lerp(want, Math.min(1, 3 * dt)) : want;
    camera.position.copy(centre).add(new THREE.Vector3(Math.cos(yaw) * Math.cos(orbit.pitch), Math.sin(orbit.pitch), Math.sin(yaw) * Math.cos(orbit.pitch)).multiplyScalar(orbit.dist));
    camera.lookAt(centre);
  };

  const ground = (x, y) => {
    const ndc = new THREE.Vector2(x / innerWidth * 2 - 1, -(y / innerHeight) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, camera);
    const { origin, direction } = ray.ray;
    if (direction.y >= 0) return null;
    const hit = origin.clone().addScaledVector(direction, -origin.y / direction.y);
    return [hit.x, -hit.z];
  };
  const project = v => { const p = v.clone().project(camera); return [(p.x + 1) / 2 * innerWidth, (1 - p.y) / 2 * innerHeight, p.z < 1]; };
  const duckAt = (x, y) => {
    let best = -1, nearest = Infinity;
    ducks.forEach((d, i) => {
      const feet = d.position.clone(), [fx, fy, front] = project(feet);
      if (!front) return;
      const [tx, ty] = project(feet.clone().add(new THREE.Vector3(0, DUCK_TALL, 0)));
      const reach = Math.max(0.6 * Math.hypot(tx - fx, ty - fy), 18);
      const off = Math.hypot(x - (tx + fx) / 2, y - (ty + fy) / 2);
      if (off < reach && off < nearest) { nearest = off; best = i; }
    });
    return best;
  };

  // ---------- the visitor ----------
  let pointer = [innerWidth / 2, innerHeight / 2], pointers = new Map(), dragged = 0, pressedAt = 0, gripping = false, trail = [];
  let pinch = null;
  // Where a thing is put down: at the pointer for a key, somewhere open in the garden for a toolbar button, clear of
  // the pond, the rocks and the tree's trunk and a little way in from the fence.
  let fromToolbar = false;
  const openSpot = () => {
    const margin = 0.5;
    for (let tries = 0; tries < 200; tries++) {
      const xy = [margin + Math.random() * (size - 2 * margin), margin + Math.random() * (size - 2 * margin)];
      const clear = (x, y, r) => Math.hypot(xy[0] - x, xy[1] - y) > r;
      if ((!snap.pond || clear(snap.pond[0], snap.pond[1], snap.pond[2] + 0.3)) && snap.rocks.every(([x, y, r]) => clear(x, y, r + 0.3))
          && clear(snap.tree[0], snap.tree[1], 0.3)) return xy;
    }
    return [size / 2, size / 2];
  };
  const where = () => fromToolbar ? openSpot() : ground(...pointer) || ground(innerWidth / 2, innerHeight / 2);
  const inGarden = xy => xy && xy[0] > 0 && xy[1] > 0 && xy[0] < size && xy[1] < size;
  const click = (x, y) => {
    const xy = ground(x, y);
    if (!xy) return;
    const tapped = duckAt(x, y);
    if (tapped >= 0) { selected = tapped; return; }
    if (snap.music && Math.hypot(xy[0] - snap.music[0], xy[1] - snap.music[1]) < 0.4) {  // each tap steps the volume
      const levels = [0, 0.25, 0.5, 0.75, 1];
      const at = levels.findIndex(l => Math.abs(l - snap.music_volume) < 0.13);
      const volume = levels[(at + 1) % levels.length];
      act("garden.volume", { level: volume });
      panels.say(volume === 0 ? "music off" : `music volume ${Math.round(volume * 100)}%`);
      return;
    }
    if (Math.hypot(xy[0] - snap.tree[0], xy[1] - snap.tree[1]) < 0.35) act("garden.shake_tree");
    else if (possessing) return;
    else selected = -1;
  };
  const letGo = () => {
    const now = performance.now() / 1000;
    let v = [0, 0];
    if (trail.length > 1 && now - pressedAt > 0.25) {
      const [t0, a] = trail[0], [t1, b] = trail[trail.length - 1];
      const span = Math.max(t1 - t0, 0.02);
      v = [(b[0] - a[0]) / span, (b[1] - a[1]) / span];
    }
    const at = trail.length ? trail[trail.length - 1][1] : [0, 0];
    act("garden.release", { x: at[0], y: at[1], vx: v[0], vy: v[1] });
  };
  canvas.addEventListener("pointerdown", e => {
    sound.start();
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    pointer = [e.clientX, e.clientY];
    dragged = 0;
    pressedAt = performance.now() / 1000;
    if (helpOn) { helpOn = false; return; }
    if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), dist: orbit.dist }; return; }
    if (handMode && e.button === 0) {
      gripping = true;
      const xy = ground(e.clientX, e.clientY);
      if (xy) act("garden.grab", { x: xy[0], y: xy[1] });
    }
  });
  canvas.addEventListener("pointermove", e => {
    const last = pointers.get(e.pointerId);
    pointer = [e.clientX, e.clientY];
    if (!last) return;
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    const dx = e.clientX - last[0], dy = e.clientY - last[1];
    dragged += Math.hypot(dx, dy);
    if (pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      orbit.dist = Math.min(Math.max(pinch.dist * pinch.d / Math.max(Math.hypot(a[0] - b[0], a[1] - b[1]), 1), 2), farthest);
      return;
    }
    if (gripping || (handMode && e.button === 0 && e.pointerType === "mouse" && e.buttons === 1)) return;
    if (!possessing) {
      orbit.yaw += dx * 0.006;
      orbit.pitch = Math.min(Math.max(orbit.pitch + dy * 0.006, PITCH[0]), PITCH[1]);
      still = 0;
    }
  });
  const pointerUp = e => {
    pointers.delete(e.pointerId);
    if (pinch) { if (pointers.size < 2) pinch = null; return; }
    const quick = performance.now() / 1000 - pressedAt < 0.3 && dragged < 8;
    if (gripping) { gripping = false; letGo(); }
    if (quick) click(e.clientX, e.clientY);
  };
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", e => { pointers.delete(e.pointerId); pinch = null; if (gripping) { gripping = false; letGo(); } });
  canvas.addEventListener("contextmenu", e => e.preventDefault());
  canvas.addEventListener("wheel", e => { e.preventDefault(); orbit.dist = Math.min(Math.max(orbit.dist * (e.deltaY < 0 ? 0.92 : 1.08), 2), farthest); }, { passive: false });

  const keys = new Set();
  const verbs = {
    ride: () => { if (possessing || selected >= 0) { possessing = !possessing; if (!possessing) act("garden.wheel", { duck: -1, fwd: 0, turn: 0 }); } },
    help: () => { helpOn = !helpOn; panels.menu.hidden = !helpOn; },
    overlay: () => { panels.overlayOn = !panels.overlayOn; },
    clap: () => act("garden.scare"),
    shake: () => act("garden.shake_tree"),
    music: () => { const xy = where(); if (xy) act("garden.music", { x: xy[0], y: xy[1], on: snap.music ? 0 : 1 }); },
    give: () => { if (selected >= 0) act("garden.give", { duck: selected }); },
    pet: () => { if (selected >= 0) act("garden.pet", { duck: selected }); },
    drum: () => { const xy = where(); if (xy) act("garden.drum", { x: xy[0], y: xy[1], on: snap.drum ? 0 : 1 }); },
    // the drum is one of the instruments: while none is down, one time in eleven the button puts the drum down
    instrument: () => {
      const xy = where();
      if (!xy) return;
      if (fromToolbar && !snap.drum && Math.random() < 1 / 11) act("garden.drum", { x: xy[0], y: xy[1], on: 1 });
      else act("garden.instrument", { x: xy[0], y: xy[1] });
    },
    ball: () => { const xy = where(); if (inGarden(xy)) act("garden.drop_ball", { x: xy[0], y: xy[1] }); },
    hat: () => { const xy = where(); if (xy) act("garden.drop_hat", { x: xy[0], y: xy[1] }); },
    hand: () => {
      handMode = !handMode;
      if (!handMode && gripping) { gripping = false; letGo(); }
      panels.say(handMode ? "the hand is out: hold and drag to pick things up" : "the hand is put away");
    },
    sound: () => { sound.setOn(!sound.on); panels.say(sound.on ? "sound on" : "sound off"); },
    regenerate: () => { $("regenerate").hidden = false; },
  };
  const KEYS = { Tab: "ride", "/": "help", o: "overlay", c: "clap", f: "shake", m: "music", g: "give", p: "pet", d: "drum", i: "instrument", b: "ball", h: "hand", t: "hat" };
  addEventListener("keydown", e => {
    sound.start();
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    keys.add(k);
    if (e.repeat) return;
    if (k === "Tab") e.preventDefault();
    if (possessing && ["w", "a", "s", "d"].includes(k)) return;  // D steers while riding
    if (KEYS[k]) verbs[KEYS[k]]();
  });
  addEventListener("keyup", e => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));

  // the toolbar, for a touch screen or a mouse
  for (const b of document.querySelectorAll("[data-verb]")) b.addEventListener("click", e => { e.stopPropagation(); sound.start(); fromToolbar = true; verbs[b.dataset.verb](); fromToolbar = false; });
  panels.menu.addEventListener("pointerdown", e => { e.stopPropagation(); helpOn = false; panels.menu.hidden = true; });  // tap anywhere to close
  const pad = {};  // the ride view's on-screen wheel
  for (const b of document.querySelectorAll("[data-steer]")) {
    const key = b.dataset.steer;
    b.addEventListener("pointerdown", e => { e.preventDefault(); pad[key] = true; });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, () => { pad[key] = false; });
  }
  $("regenerate-keep").addEventListener("click", e => { e.stopPropagation(); $("regenerate").hidden = true; });
  $("regenerate-go").addEventListener("click", e => {
    e.stopPropagation();
    resetting = true;
    Save.forget();
    location.reload();
  });

  // ---------- each frame ----------
  let before = performance.now(), frames = 0, frameSince = before;
  // ?record=N sends N frames, ten a second, to a server that saves them (not part of the garden's own hosting)
  const recording = params.has("record") ? { frames: +params.get("record") || 150, sent: 0, last: 0 } : null;
  // A recording keeps drawing in a background tab, where the browser stops animation frames
  const nextFrame = recording ? fn => sleep(33).then(() => fn(performance.now())) : fn => requestAnimationFrame(fn);
  const frame = () => {
    nextFrame(frame);
    const now = performance.now(), dt = Math.min((now - before) / 1000, 0.1);
    before = now;
    show(dt);
    rollKicked(dt);
    ducks.forEach(d => d.update(dt, now / 1000));
    cameraTo(dt);
    if (possessing && selected >= 0) {
      const fwd = (keys.has("w") || pad.fwd ? 1 : 0) - (keys.has("s") || pad.back ? 1 : 0);
      const turn = (keys.has("a") || pad.left ? 1 : 0) - (keys.has("d") || pad.right ? 1 : 0);
      act("garden.wheel", { duck: selected, fwd, turn });
    }
    if (garden.server.watched !== selected) act("garden.watch", { duck: selected });
    // the glove follows the pointer over the lawn
    const xy = handMode && !possessing ? ground(...pointer) : null;
    glove.visible = !!xy;
    if (xy) {
      const at = [Math.min(Math.max(xy[0], 0), size), Math.min(Math.max(xy[1], 0), size)];
      glove.position.lerp(new THREE.Vector3(at[0], gripping ? 0.22 : 0.4, -at[1]), 0.5);
      glove.rotation.y = orbit.yaw + Math.PI;
      trail.push([now / 1000, at]);
      while (trail.length > 1 && now / 1000 - trail[0][0] > 0.12) trail.shift();
      if (gripping) act("garden.hand_at", { x: at[0], y: at[1] });
    }
    canvas.style.cursor = handMode && !possessing ? "none" : "";
    renderer.render(scene, camera);
    if (recording && now - recording.last > 100 && recording.sent < recording.frames) {  // ?record: frames for a gif
      recording.last = now;
      const n = recording.sent++;
      canvas.toBlob(blob => fetch(`frame?n=${n}`, { method: "POST", body: blob }), "image/png");
    }
    // panels
    const d = selected >= 0 ? snap.ducks[selected] : null;
    panels.showCard(possessing ? null : d);
    if (!possessing && d) panels.bars.style.top = `${panels.card.offsetTop + panels.card.offsetHeight + 8}px`;
    panels.showBrain(!possessing && snap.brain && snap.brain.duck === selected ? snap.brain : null, d ? d.name : "", dt);
    if (snap.brain) snap.brain.spikes = [];  // each spike glows once
    panels.showRide(possessing ? snap.ride : null);
    const playing = playlist.down ? "" : sound.title;
    panels.showToasts(snap.toasts, snap.told, snap.music && playing ? `♪ ${playing}` : "");
    panels.menu.hidden = !helpOn;
    document.body.classList.toggle("riding", possessing);
    document.body.classList.toggle("watching", selected >= 0);
    document.body.classList.toggle("hand", handMode);
    panels.hint.textContent = "W A S D  steer      O  hide what it sees      Tab  get off";
    const labels = [];
    if (!possessing) {
      ducks.forEach((duck, i) => {
        if (!duck.state) return;
        if (duck.sign) {
          const [x, y, front] = project(duck.localToWorld(new THREE.Vector3(0, duck.signAt, 0)));
          if (front) labels.push([x, y, duck.sign, "sign"]);
        }
        for (const [at, text, scale, alpha] of duck.zs) {
          const [x, y, front] = project(duck.localToWorld(at.clone()));
          if (front) labels.push([x, y, text, "z", alpha, scale]);
        }
      });
    }
    panels.showLabels(labels);
    frames++;
    if (now - frameSince > 2000) {
      const fps = frames * 1000 / (now - frameSince);
      frames = 0; frameSince = now;
      if (fps < 30 && pixelRatio > 1) { pixelRatio = Math.max(1, pixelRatio - 0.5); resize(); }
      panels.status.textContent = speed < 0.93 ? `the brains are running at ${speed.toFixed(2)}× real time on this device` : "";
    }
  };
  nextFrame(frame);
}

boot().catch(e => {
  console.error(e);
  $("loading-line").textContent = `The garden could not start: ${e.message}`;
});
