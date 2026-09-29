// The panels over the garden, as viewer/godot/main.gd and brainview.gd draw them: the selected duck's card and its
// needs, moods and wants, the news, the brain as it fires, the ride view's retinas and descending neurons, the
// controls, and a toolbar for a touch screen. All HTML over the canvas.
import { HEX } from "../garden/snapshot.js";
import { faceImage, heartImage } from "./faces.js";

export const FEELS = { joy: "happy", fear: "scared", anger: "angry", sorrow: "sad" };
const SKILLS = ["swimming", "walking", "dancing", "eating", "fighting", "fashion", "music"];
const TINTS = { needs: "var(--coral)", moods: "var(--mustard)", wants: "var(--teal)" };
// the brain view's groups, dim and lit (brainview.gd)
const DIM = ["#3f6fb0", "#b0428f", "#3f9a62", "#c9772e", "#77798a"];
const LIT = ["#bfe0ff", "#ffb3ec", "#c6ffd6", "#ffe08a", "#ffffff"];
const FADE_S = 0.12;
const RECENT_MS = 4000;  // how long a piece of news counts as new

// The marks drawn over a duck's head (view/duck.js), drawn again small for the controls card.
const svg = (inner, label) => ({ label, html: `<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">${inner}</svg>` });
const face = (mood, label) => ({ label, html: `<img src="${faceImage(mood)}" width="22" height="22" alt="">` });
const MARKS = {
  ball: face("joy", "happy face"),
  spike: face("fear", "scared face"),
  block: face("anger", "angry face"),
  drop: face("sorrow", "sad face"),
  zs: svg('<text x="2" y="18" font-family="sans-serif" font-weight="700" font-size="13" fill="#1c2a4d">Z</text><text x="12" y="11" font-family="sans-serif" font-weight="700" font-size="9" fill="#1c2a4d">z</text>', "Z z z"),
  bandage: svg('<g transform="rotate(45 11 11)"><rect x="2" y="8" width="18" height="6" rx="2" fill="#f0c8a0" stroke="#1c2a4d" stroke-width="1.2"/></g><g transform="rotate(-45 11 11)"><rect x="2" y="8" width="18" height="6" rx="2" fill="#f0c8a0" stroke="#1c2a4d" stroke-width="1.2"/></g><rect x="8.5" y="8.5" width="5" height="5" transform="rotate(45 11 11)" fill="#dca27a"/>', "bandage"),
  giving: { label: "pink heart", html: heartImage("giving") },
  glad: { label: "red heart", html: heartImage("glad") },
  ring: svg('<ellipse cx="11" cy="14" rx="9" ry="4" fill="none" stroke="#ee6f5c" stroke-width="2.5"/>', "red ring"),
};

// On a touch screen the card names the on-screen buttons instead of keys, and leaves out what only a keyboard does.
const TOUCH = {
  "Tab": ["Ride", "ride the duck you are following. The arrows on screen steer it"],
  "P": ["Pet", "pet the duck you are following"],
  "G": ["Give a fruit", "give the duck you are following a fruit"],
  "H": ["Hand", "reach into the garden, or take your hand back out"],
  "F, or tap the tree": ["Shake the tree, or tap the tree", "shake some fruit off the tree"],
  "T": ["Hat", "drop a hat in the garden, for any duck that wants it"],
  "B": ["Ball", "drop a ball in the garden"],
  "D": null,
  "I": ["Instrument", "drop a random instrument, one of ten, in the garden. Sometimes it is the drum"],
  "M": ["Music box", "put the music box in the garden, or take it away"],
  "C": ["Clap", "clap your hands, which startles every duck"],
  "drag": ["drag", "turn the camera"],
  "scroll or pinch": ["pinch", "zoom in and out"],
  "/": ["/ controls", "open or close this card"],
};
const touch = matchMedia("(pointer: coarse)").matches;

export const CONTROLS = [
  ["the ducks", [
    ["tap a duck", "follow that duck and see its needs, moods, brain and friends"],
    ["tap the grass", "stop following it"],
    ["Tab", "ride the duck you are following. W A S D steers it, and O hides what it sees"],
    ["P", "pet the duck you are following"],
    ["G", "give the duck you are following a fruit"],
  ]],
  ["your hand", [
    ["H", "reach into the garden, or take your hand back out"],
    ["hold and drag", "carry a fruit, a hat, a ball, an instrument, the music box or a duck"],
    ["let go while moving", "throw what you are carrying. A fruit drops instead of flying. A duck you throw will trust you less"],
    ["the garbage can", "carry something to it and the lid opens. Drop or throw it in to get rid of it. Ducks cannot go in"],
  ]],
  ["above a duck's head", [
    [MARKS.ball, "the duck is happy. The bigger the face, the happier it is"],
    [MARKS.spike, "the duck is scared. The face trembles until it calms down"],
    [MARKS.block, "the duck is angry. The face shakes until it calms down"],
    [MARKS.drop, "the duck is sad. Blue tears fall when it cries"],
    [MARKS.zs, "the duck is asleep"],
    [MARKS.bandage, "the duck was knocked over, by a shove, a throw or a trip, and has not got over it yet"],
    [MARKS.giving, "the duck is comforting a friend who is upset"],
    [MARKS.glad, "the duck was comforted and is glad of it. The heart beats"],
    ["hungry, laughs, dances", "a word shows for a few seconds when the duck feels or does something"],
    [MARKS.ring, "this marks the duck you are following"],
  ]],
  ["the garden", [
    ["F, or tap the tree", "shake some fruit off the tree"],
    ["T", "drop a hat where the pointer is, for any duck that wants it"],
    ["B", "drop a ball where the pointer is"],
    ["D", "put the drum down where the pointer is, or pick it back up"],
    ["I", "drop a random instrument, one of ten, where the pointer is. The Instrument button sometimes gives you the drum"],
    ["M", "put the music box down where the pointer is, or pick it back up"],
    ["tap the music box", "make it louder, step by step, then off"],
    ["Connect Spotify", "for Spotify Premium accounts Chris has added: the box plays his playlist, and you can skip and shuffle"],
    ["C", "clap your hands, which startles every duck"],
  ]],
  ["the view", [
    ["drag", "turn the camera. While your hand is in the garden, drag with the right button instead"],
    ["scroll or pinch", "zoom in and out"],
    ["Sound", "tap to switch between all sound, the ducks with no music, and silence"],
    ["/", "open or close this card"],
  ]],
];

const el = (tag, cls, parent, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
};

export class Panels {
  constructor(root) {
    this.root = root;
    this.toasts = el("div", "paper toasts", root);
    this.card = el("div", "paper card", root);
    this.cardName = el("div", "card-name", this.card);
    this.cardTrait = el("div", "card-trait", this.card);
    this.cardBody = el("div", "card-body", this.card);
    this.cardRule = el("div", "card-rule", this.card);
    this.cardSkills = el("div", "card-skills", this.card);
    this.bars = el("div", "paper bars", root);
    this.brain = el("div", "brain", root);
    this.brainTitle = el("div", "brain-title", this.brain);
    this.brainUnder = el("div", "brain-under", this.brain);
    this.brainCanvas = el("canvas", "brain-canvas", this.brain);
    this.brainLegend = el("div", "brain-legend", this.brain);
    this.ride = el("div", "ride", root);
    this.eyes = [0, 1].map(() => el("canvas", "eye", this.ride));
    this.dn = el("div", "paper dn", this.ride);
    this.hint = el("div", "paper hint", root, "/  controls");
    this.status = el("div", "status", root);
    this.labels = el("div", "labels", root);
    this.menu = this._menu(root);
    this.glow = new Map();
    this.said = ""; this.saidUntil = 0;
    this.overlayOn = true;
    this.card.hidden = this.bars.hidden = this.brain.hidden = this.ride.hidden = true;
  }

  _menu(root) {
    const menu = el("div", "menu", root);
    const card = el("div", "menu-card", menu);
    el("div", "menu-title", card, "MICRO GARDEN");
    el("div", "menu-under", card, "controls");
    const spread = el("div", "menu-spread", card);
    const sides = [el("div", "menu-side", spread), el("div", "menu-side", spread)];
    CONTROLS.forEach(([head, rows], s) => {
      const holder = el("div", "menu-section", sides[s < CONTROLS.length / 2 ? 0 : 1]);
      el("div", "menu-head", holder, head);
      const grid = el("div", "menu-grid", holder);
      for (let [key, what] of rows) {
        if (touch && typeof key === "string" && key in TOUCH) {
          if (!TOUCH[key]) continue;
          [key, what] = TOUCH[key];
        }
        if (typeof key === "string") el("span", "key", grid, key);
        else { const mark = el("span", "key mark", grid); mark.innerHTML = key.html; el("span", null, mark, key.label); }
        el("span", "what", grid, what);
      }
    });
    el("div", "menu-foot", card, "Tap anywhere to close.");
    menu.hidden = true;
    return menu;
  }

  say(line) { this.said = line; this.saidUntil = performance.now() + 2500; }

  // The brain view's neuron positions: xy (x, y pairs, 0 to 1 across) and a group each, from the brain bundle.
  setBrainPoints(xy, group, groups, of) {
    this.points = { xy, group, n: group.length, of };
    const across = 396, top = 0;
    let maxY = 0;
    for (let k = 0; k < group.length; k++) maxY = Math.max(maxY, xy[2 * k + 1]);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const c = this.brainCanvas;
    c.width = across * dpr; c.height = (top + across * maxY + 4) * dpr;
    c.style.width = `${across}px`; c.style.height = `${top + across * maxY + 4}px`;
    this.brainScale = { across, top, dpr };
    const base = document.createElement("canvas");
    base.width = c.width; base.height = c.height;
    const g = base.getContext("2d");
    g.scale(dpr, dpr);
    for (let k = 0; k < group.length; k++) {
      g.fillStyle = DIM[group[k]];
      g.fillRect(xy[2 * k] * across - 1.1, top + xy[2 * k + 1] * across - 1.1, 2.2, 2.2);
    }
    this.brainBase = base;
    this.brainLegend.innerHTML = groups.map((name, k) => `<span style="color:${LIT[k]}">${name}</span>`).join("");
  }

  showBrain(brain, name, dt) {
    const on = !!brain && !!this.points;
    this.brain.hidden = !on;
    if (!on) { this.glow.clear(); return; }
    this.brainTitle.textContent = `${name}'s brain`;
    this.brainUnder.textContent = `${this.points.n.toLocaleString()} of its ${this.points.of.toLocaleString()} neurons, lighting up as they fire`;
    for (const k of brain.spikes || []) this.glow.set(k, 1);
    const c = this.brainCanvas, g = c.getContext("2d"), { across, top, dpr } = this.brainScale, { xy, group } = this.points;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(this.brainBase, 0, 0);
    g.scale(dpr, dpr);
    for (const [k, v] of this.glow) {
      const left = v - dt / FADE_S;
      if (left <= 0) { this.glow.delete(k); continue; }
      this.glow.set(k, left);
      g.globalAlpha = Math.min(1, left * 1.4);
      g.fillStyle = LIT[group[k]];
      g.fillRect(xy[2 * k] * across - 1.6, top + xy[2 * k + 1] * across - 1.6, 3.2, 3.2);
    }
    g.globalAlpha = 1;
  }

  showCard(d) {
    this.card.hidden = !d;
    if (!d) { this.bars.hidden = true; return; }
    const state = d.asleep ? "asleep" : d.crying ? "crying" : FEELS[d.mood] ?? d.mood;
    this.cardName.textContent = d.name;
    this.cardTrait.textContent = d.label ? `${d.label}  ·  ${state}` : state;
    const a = d.among || {};
    const has = a.favourite !== undefined;
    this.cardRule.hidden = this.cardSkills.hidden = !has;
    if (!has) { this.cardBody.textContent = ""; this.bars.hidden = true; return; }
    const lines = [`likes ${a.favourite} best`, a.hand];
    if (a.friend) lines.push(`friends with ${a.friend}`);
    if (a.grudge) lines.push(`holds a grudge against ${a.grudge}`);
    this.cardBody.textContent = lines.join("\n");
    this.cardSkills.innerHTML = SKILLS.map((s, i) => `<span>${s} ${Math.trunc(a.skills[i] * 100)}%</span>`).join("");
    this._bars(d.readout);
  }

  _bars(rows) {
    this.bars.hidden = !rows || !rows.length;
    if (this.bars.hidden) return;
    if (!this.barRows || this.barRows.length !== rows.length) {
      this.bars.innerHTML = "";
      this.barRows = [];
      let last = "";
      for (const [section, name] of rows) {
        if (section !== last) {
          last = section;
          const head = el("div", "bars-head", this.bars);
          head.innerHTML = `<i style="background:${TINTS[section]}"></i>${section.toUpperCase()}`;
          head.style.borderColor = TINTS[section];
        }
        const line = el("div", "bar", this.bars);
        el("span", "bar-name", line, name);
        const track = el("span", "bar-track", line);
        const fill = el("span", "bar-fill", track);
        fill.style.background = TINTS[section];
        const num = el("span", "bar-num", line);
        this.barRows.push([fill, num]);
      }
    }
    rows.forEach(([, , v], k) => { this.barRows[k][0].style.width = `${v * 100}%`; this.barRows[k][1].textContent = Math.trunc(v * 100); });
  }

  // The news, newest at the bottom, older lines fading. The newest line carries a dot and full ink for RECENT_MS after it
  // arrives; `told` counts every line the garden has added, so a line that repeats earlier words still counts as new.
  showToasts(lines, told, extra) {
    const now = performance.now();
    if (told !== this.told) { this.told = told; this.newAt = now; }
    const fresh = now - this.newAt < RECENT_MS;
    const rows = lines.map((line, i) => [line, fresh && i === lines.length - 1, 0.5 + 0.5 * (i + 1) / lines.length]);
    if (now < this.saidUntil) rows.push([this.said, false, 1]);
    else if (extra) rows.push([extra, false, 1]);
    this.toasts.hidden = !rows.length;
    const key = `${told}\n` + rows.map(([line, recent]) => `${recent ? "*" : ""}${line}`).join("\n");
    if (key === this.toastKey) return;
    this.toastKey = key;
    this.toasts.replaceChildren(...rows.map(([line, recent, ink]) => {
      const row = el("div", recent ? "toast recent" : "toast", null, line);
      row.style.opacity = recent ? 1 : ink;
      return row;
    }));
  }

  // The ridden duck's brain input and output: each eye's 721 columns on the fly's lattice, and six readouts in Hz.
  showRide(ride) {
    this.ride.hidden = !ride || !this.overlayOn;
    if (this.ride.hidden) return;
    const r = 96;
    this.eyes.forEach((c, eye) => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2), size = 2 * r + 16;
      if (c.width !== size * dpr) { c.width = c.height = size * dpr; c.style.width = c.style.height = `${size}px`; }
      const g = c.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, size, size);
      g.fillStyle = "rgba(28,42,77,0.85)";
      g.beginPath(); g.arc(size / 2, size / 2, r + 8, 0, 2 * Math.PI); g.fill();
      for (let k = 0; k < 721; k++) {
        const v = ride.lum[eye * 721 + k];
        const mix = (a, b) => Math.round(a + (b - a) * Math.min(Math.max(v, 0), 1));
        g.fillStyle = `rgb(${mix(28, 244)},${mix(42, 234)},${mix(77, 213)})`;
        g.beginPath(); g.arc(size / 2 + HEX.az[k] * r, size / 2 - HEX.el[k] * r, 2.3, 0, 2 * Math.PI); g.fill();
      }
    });
    this.dn.innerHTML = ride.dn.map(([name, hz]) =>
      `<div class="dn-row"><span class="dn-track"><span style="width:${Math.min(Math.max(hz / 5, 0), 1) * 100}%"></span></span>${name} ${hz.toFixed(1)} Hz</div>`).join("");
  }

  // Words over the ducks: [screen x, screen y, text, class, opacity, scale]
  showLabels(items) {
    while (this.labels.children.length < items.length) el("div", "label", this.labels);
    [...this.labels.children].forEach((node, k) => {
      const it = items[k];
      if (!it) { node.hidden = true; return; }
      node.hidden = false;
      const [x, y, text, cls, alpha = 1, scale = 1] = it;
      if (node.textContent !== text) node.textContent = text;
      node.className = `label ${cls}`;
      node.style.transform = `translate(${x}px, ${y}px) translate(-50%, -50%) scale(${scale})`;
      node.style.opacity = alpha;
    });
  }
}
