// The hover card for Micro Garden on chrisnhill.com's portfolio pages, in the style of their other project cards: a
// dark stage fades in over the picture, five little ducks come to the pointer as if it held a fruit, and each duck's
// brain flickers over its head. Any [data-ch-fx="garden"] on the page gets one. A touch screen, or a visitor who asks
// for less motion, keeps the plain picture.
(() => {
  const fine = matchMedia("(hover: hover) and (pointer: fine)").matches;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!fine || still) return;
  const INK = "rgba(214,224,234,";
  const HUES = [8, 172, 42, 252, 322];  // the garden's ribbons: coral, teal, mustard, violet, pink

  function card(media) {
    if (media.dataset.gardenFx) return;
    media.dataset.gardenFx = "1";
    const stage = document.createElement("div");
    stage.className = "ch-fx-stage";
    const canvas = document.createElement("canvas");
    const hint = document.createElement("div");
    hint.className = "ch-fx-hint";
    hint.textContent = "lead the ducks";
    stage.append(canvas, hint);
    stage.style.display = "block";
    media.appendChild(stage);
    const g = canvas.getContext("2d");
    let w = 0, h = 0, ducks = null, pointer = null, running = false, last = 0, time = 0, sparks = [];

    const size = () => {
      const r = media.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
      w = r.width; h = r.height;
      canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const hatch = () => HUES.map((hue, i) => ({
      x: w * (0.2 + 0.15 * i), y: h * (0.35 + 0.3 * ((i * 7) % 3) / 2), a: Math.random() * 6.28, hue,
      wander: Math.random() * 6.28, bob: Math.random() * 6.28,
    }));

    const drawDuck = d => {
      g.save();
      g.translate(d.x, d.y);
      g.rotate(d.a);
      const peck = d.pecking ? Math.max(0, Math.sin(d.bob * 3)) * 2.5 : 0;
      g.fillStyle = `hsla(${d.hue},62%,64%,0.95)`;
      g.beginPath(); g.ellipse(-2, 0, 11, 8, 0, 0, 6.283); g.fill();
      g.fillStyle = "rgba(244,234,213,0.95)";
      g.beginPath(); g.arc(8 + peck, 0, 5.5, 0, 6.283); g.fill();
      g.fillStyle = "hsla(36,85%,58%,1)";
      g.beginPath(); g.moveTo(12.5 + peck, -2); g.lineTo(17.5 + peck, 0); g.lineTo(12.5 + peck, 2); g.closePath(); g.fill();
      g.restore();
    };

    const frame = now => {
      if (!running) return;
      requestAnimationFrame(frame);
      const dt = Math.min((now - (last || now)) / 1000, 0.05);
      last = now;
      time += dt;
      if (!ducks) ducks = hatch();
      g.fillStyle = "#0a0c10";
      g.fillRect(0, 0, w, h);
      ducks.forEach((d, i) => {
        let tx, ty;
        if (pointer) {  // everyone wants the fruit, each to its own spot round it
          const around = (i / ducks.length) * 6.283;
          tx = pointer.x + Math.cos(around) * 24; ty = pointer.y + Math.sin(around) * 18;
        } else {
          d.wander += (Math.random() - 0.5) * dt * 2;
          tx = d.x + Math.cos(d.wander) * 40; ty = d.y + Math.sin(d.wander) * 40;
        }
        const dx = tx - d.x, dy = ty - d.y, dist = Math.hypot(dx, dy) || 1;
        const speed = pointer ? 70 : 18;
        d.pecking = !!pointer && dist < 8;
        if (!d.pecking) {
          d.x += (dx / dist) * speed * dt; d.y += (dy / dist) * speed * dt;
          d.a += Math.atan2(Math.sin(Math.atan2(dy, dx) - d.a), Math.cos(Math.atan2(dy, dx) - d.a)) * Math.min(1, 6 * dt);
        } else d.a += Math.atan2(Math.sin(Math.atan2(pointer.y - d.y, pointer.x - d.x) - d.a), Math.cos(Math.atan2(pointer.y - d.y, pointer.x - d.x) - d.a)) * Math.min(1, 6 * dt);
        d.x = Math.min(Math.max(d.x, 14), w - 14); d.y = Math.min(Math.max(d.y, 14), h - 58);
        d.bob += dt * 6;
        // its brain: a few neurons fire over its head, more when food is near
        const rate = pointer ? (dist < 60 ? 70 : 30) : 10;
        if (Math.random() < rate * dt) sparks.push({ x: d.x + (Math.random() - 0.5) * 22, y: d.y - 16 - Math.random() * 12, life: 0, hue: d.hue });
      });
      for (const [i, a] of ducks.entries()) for (const b of ducks.slice(i + 1)) {  // no two stand in one place
        const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy);
        if (d > 0 && d < 20) { const push = (20 - d) / 2 / d; a.x -= dx * push; a.y -= dy * push; b.x += dx * push; b.y += dy * push; }
      }
      if (pointer) {  // the fruit
        g.fillStyle = "hsla(28,88%,58%,0.95)";
        g.beginPath(); g.arc(pointer.x, pointer.y, 6, 0, 6.283); g.fill();
        g.fillStyle = "hsla(130,55%,45%,0.95)";
        g.beginPath(); g.ellipse(pointer.x + 3, pointer.y - 7, 3.5, 1.6, -0.5, 0, 6.283); g.fill();
      }
      ducks.forEach(drawDuck);
      sparks = sparks.filter(s => (s.life += dt) < 0.35);
      for (const s of sparks) {
        const a = 1 - s.life / 0.35;
        g.fillStyle = `hsla(${s.hue},90%,82%,${a})`;
        g.fillRect(s.x - 1.2, s.y - 1.2, 2.4, 2.4);
      }
      g.font = "10px ui-monospace, Menlo, monospace";
      g.fillStyle = INK + "0.5)";
      g.fillText(pointer ? "five ducks · they come for the fruit" : "five ducks · five fly brains", w * 0.055, h - 44);
      g.fillStyle = INK + "0.28)";
      g.fillText("139,248 neurons each, live in your browser", w * 0.055, h - 32);
    };

    const where = e => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
    media.addEventListener("pointerenter", () => {
      size();
      stage.classList.add("on");
      if (!running) { running = true; last = 0; requestAnimationFrame(frame); }
    });
    media.addEventListener("pointermove", e => { pointer = where(e); });
    media.addEventListener("pointerleave", () => {
      pointer = null;
      stage.classList.remove("on");
      setTimeout(() => { if (!stage.classList.contains("on")) running = false; }, 450);
    });
    addEventListener("resize", () => { if (running) size(); });
  }

  const start = () => document.querySelectorAll('[data-ch-fx="garden"]').forEach(card);
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", start); else start();
})();
