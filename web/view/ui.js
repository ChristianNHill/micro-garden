// The panels over the garden, as viewer/godot/main.gd and brainview.gd draw them: the selected duck's card and its
// needs, moods and wants, the news, the brain as it fires, the ride view's retinas and descending neurons, the
// controls, and a toolbar for a touch screen. All HTML over the canvas.
import { HEX } from "../garden/snapshot.js";

export const FEELS = { joy: "happy", fear: "scared", anger: "angry", sorrow: "sad" };
const SKILLS = ["swimming", "walking", "dancing", "eating", "fighting", "fashion", "music"];
const TINTS = { needs: "var(--coral)", moods: "var(--mustard)", wants: "var(--teal)" };
// the brain view's groups, dim and lit (brainview.gd)
const DIM = ["#3f6fb0", "#b0428f", "#3f9a62", "#c9772e", "#77798a"];
const LIT = ["#bfe0ff", "#ffb3ec", "#c6ffd6", "#ffe08a", "#ffffff"];
const FADE_S = 0.12;
const RECENT_MS = 4000;  // how long a piece of news counts as new

export const CONTROLS = [
  ["the ducks", [
    ["tap a duck", "watch this one: its needs, its moods, its brain and who its friends are"],
    ["tap the grass", "stop watching"],
    ["Tab", "ride the duck you are watching. W A S D steer it, O hides what it sees"],
    ["P", "pet it"],
    ["G", "hand it a fruit"],
  ]],
  ["your hand", [
    ["H", "reach into the garden, or take your hand back out"],
    ["hold and drag", "carry a fruit, a hat, a ball, an instrument, the music box or a duck"],
    ["let go while moving", "throw it, except a fruit, which just drops. A duck you throw thinks less of you"],
  ]],
  ["the garden", [
    ["F, or tap the tree", "shake fruit down"],
    ["T", "leave a hat at the pointer, for whoever wants it"],
    ["B", "leave a ball at the pointer"],
    ["D", "set the drum down at the pointer, or pick it back up"],
    ["I", "leave an instrument at the pointer, one of ten, picked at random. The toolbar's may be the drum"],
    ["M", "set the music box down at the pointer, or pick it back up"],
    ["tap the music box", "turn it off and on; your device sets how loud Spotify plays"],
    ["C", "clap, which startles every duck"],
  ]],
  ["the view", [
    ["drag", "turn the camera. Right-drag instead while your hand is out"],
    ["scroll or pinch", "zoom in and out"],
    ["/", "open and close this"],
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
    el("div", "menu-under", card, "the controls");
    const spread = el("div", "menu-spread", card);
    const sides = [el("div", "menu-side", spread), el("div", "menu-side", spread)];
    CONTROLS.forEach(([head, rows], s) => {
      const holder = el("div", "menu-section", sides[s < CONTROLS.length / 2 ? 0 : 1]);
      el("div", "menu-head", holder, head);
      const grid = el("div", "menu-grid", holder);
      for (const [key, what] of rows) { el("span", "key", grid, key); el("span", "what", grid, what); }
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
    this.brainTitle.textContent = `${name}'s Brain`;
    this.brainUnder.textContent = `${this.points.n.toLocaleString()} of its ${this.points.of.toLocaleString()} neurons, as they fire`;
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
