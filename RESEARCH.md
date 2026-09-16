# Quiet Garden — research notes

## Goal (settled 2026-09-16)

**Show five microducks driven by real AI, with emergent behavior and distinct personalities, responding to the garden's environment and their own bodily needs.** The fly brain is the hook: each duck is a puppet on its own running copy of a real fruit fly connectome. The Dreamcast-era aesthetic and the light gameplay (feed, pet, watch, come back tomorrow) are what make it sticky enough that people keep it open.

Ranked, so trade-offs resolve the same way every time:

1. **Emergence and personality are the product.** If five ducks with the same wiring do not visibly behave differently, and if nothing unscripted happens in ten minutes, nothing else matters.
2. **Environment and body drive behavior.** Hunger, fatigue, sleep pressure, water versus land, day and night, weather, where the food is. Behavior has to be readable as a response to these, not as noise.
3. **The fly brain is the mechanism and the story.** Real wiring, real descending neurons for output, real sensory neurons for input, real mushroom-body plasticity for learning. Honestly labeled, with a shuffled-weights control behind a flag.
4. **Aesthetic and gameplay make it sticky.** Low-poly Dreamcast look, orbit camera, click to feed or pet, persistence across sessions. Enough to return to, never enough to grind.

Platform: **native desktop app**, not a web page. Brain server in Python. Body and renderer are open decisions (`ARCHITECTURE.md` §6.2); candidates are the microduck MuJoCo sim, a Godot 4 garden, and physical microducks. See §8 and `ARCHITECTURE.md`.

Non-goals: races, karate, breeding, permanent death, naming UI, multiplayer, a win state.

Revised 2026-09-16 (see `ARCHITECTURE.md`): **vision is in.** Ducks see through a camera or rendered view fed to the optic lobes, and the player can possess a duck to see its POV. **Microducks are a real robot** (pollen-robotics/microduck, Apache-2.0, MuJoCo sim with the same command contract), so the body may be the MuJoCo sim now and physical robots in a real room later. Personalities are sillier, Chao-label style. Three architectural decisions remain open in `ARCHITECTURE.md` §6.

---

Everything gathered so far: the Chao Garden inspiration, two academic papers on emergent NPC behavior, the fruit fly connectome scene, and what a PS1/PS2-era look would actually take. The sections below are the option space as it was researched; the goal block above and §8 are what was decided.

**Reviewed 2026-09-16.** Inline corrections are marked *Review:*; the consolidated list of gaps, blindspots, and open decisions is §7. The biggest single correction: Chao Garden is a Dreamcast game, not a PlayStation one, so the PS1 rendering recipe in §5 is the wrong reference (see §5 correction). An earlier WebGL2 particle-life prototype was reviewed and then dropped on 2026-09-16 as not the vision; the 3D build is greenfield. Three lessons from it are kept in §7d.

---

## 1. Starting point: Chao Garden

Sonic Adventure 1/2's Chao Garden — an optional virtual-pet/artificial-life feature, credited in SA1 as an "A-Life" system, itself descended from the Nights into Dreams "Nightopian" ecosystem. The parts worth stealing:

- **Stat influence, not direct stats.** A Chao's shape changes based on the *balance* of what it's been fed/given (run vs. power, swim vs. fly), not a number you set. Two sliders, diametrically opposed, decide evolution — and the effect is gradual, not instant.
- **Alignment as accumulated treatment.** Hero/dark alignment drifts from how the creature is treated over time (or, per the Nightopian precursor, from ambient mood), not a toggle.
- **Mood as a visible, momentary readout**, separate from the slow permanent trait — the emotion ball is instant; evolution is a lifetime.
- **Nobody has to play it.** Fully optional, ambient, comes alive on its own.

### 1a. Deeper mechanics, verified against Chao Island (the community's reference wiki)

The earlier summary above came from an informal video-essay transcript. Checked against the actual reference wiki, the real numbers are more precise (and occasionally correct the transcript):

**Stats** — seven total, not five:
- *Swim, Fly, Run, Power* (visible, max **3,266 points** each, levels 0–99): raised by animals/Chaos Drives from color-matched groups. Concrete usability thresholds: Run lets a Chao walk at 50 points (100 on the original Dreamcast release); Swim stops the flailing and lets it actually swim at 100; further speed breakpoints at 667 and 1,334 points.
- *Stamina*: raised only by fruit (not animals/drives) — the stat that drains during a race and can be spent for a cheer-boost.
- *Intelligence* and *Luck* (hidden, max **4,000 points**): raised by every animal/drive a little, but Chaos Drives raise them **twice as fast** as animals.
- Each stat additionally has a **grade, E through S**, fixed at birth-ish and only improvable by evolving into that stat's type — a grade persists through reincarnation, which is the actual reason the "optimal" way to min-max a Chao is evolve → reincarnate → evolve again, repeatedly.

**Evolution** — not "highest stat wins." Every animal/drive nudges a pair of **opposed influence sliders** (Run↔Power, Swim↔Fly) rather than the visible stat number directly; whichever slider is most maxed at the moment of first evolution decides the shape, regardless of which stat number is actually highest. It takes **20 animals or 40 Chaos Drives** to max a slider from neutral. A second, ongoing "Evolution Strength" value climbs after that first evolution and continues to reshape the adult's appearance over time (this is the "second evolution" — same slider system, no new cocoon).

**Alignment** — a hidden slider from **−1 (Dark) to +1 (Hero)**, starting at 0. Shifts toward Hero from positive treatment by Hero characters (or, unusually, from being mistreated by Dark characters), and symmetrically for Dark; one sourced figure for the shift-per-heart is **±0.05**. The wiki itself flags that it does not have exact numbers for every action — treat any single figure here as approximate, not verified game code.

**Personality** — this is where the source materially disagrees with common knowledge. In the original Dreamcast Sonic Adventure, personality is **three continuous traits** — Kindness, Aggressiveness, Curiosity — each low/medium/high, giving 27 face combinations, fixed at egg creation and tied directly to the Chao's face. They aren't cosmetic: **Aggressiveness controls how fast Anger and Fear decay** (low aggressiveness drains Anger faster, high aggressiveness drains Fear faster), and **Curiosity controls how fast Sorrow decays**. The 15 discrete labels (Gentle, Naughty, Energetic, etc.) that later games surface are a modern-games-only display layer over a system the wiki says is still under-documented for SA2 specifically.

**Emotion/drive system** — considerably richer than "one mood value." Two tiers of internal state, sourced from the actual memory-offset documentation:
- *Slow drives*, 0–10,000 range, rising and falling continuously: **Hunger** (down on eating), **Sleepiness** and **Tiredness** (paired, both must hit 0 to wake), **Boredom** (high boredom → more likely to use animal abilities/toys, unless Energy is also high, in which case it wanders instead), **Energy** (high energy → more likely to move at all), **Desire to Mate** (rises passively, maxed instantly by Heart Fruit).
- *Fast, visibly-animated reactions*, 0–200 range, each spiking from a specific cause and auto-decaying: **Joy** (jump/dance), **Urge to Cry**, **Fear** (shaking), **Dizziness** (head-spin) — the last three all driven up specifically by abuse, and their decay rate is the thing Aggressiveness/Curiosity modulate above.

**Reincarnation** — happiness must exceed **30 (SADX) or 50 (SA2 Battle/HD)** at death for the pink-cocoon path instead of permanent death; the reincarnated Chao keeps its name and **10% of its stat points** (rounded down), but personality resets and alignment snaps to whichever extreme (+1/−1/0) it was already leaning toward, rather than staying at its exact prior value.

**Genetics/breeding** — offspring inherit two alleles per trait across three genetic categories (appearance, stats, personality); a bred child starts around **10% of the parents' averaged stat points** and **~50%** of their averaged HP/stamina baseline. Personality, notably, is **not inherited** in the original game's model — a bred Chao gets a fresh personality roll, with only "a small chance" (per the wiki, still under community research) of copying the parent that was actively in its mating season.

**Sources**: chao-island.com's stats, evolution, alignment, personality, and emotion-offset reference pages (a community-maintained wiki built from years of datamining and reverse-engineering — treat exact numbers as well-sourced community findings, not primary Sega documentation, and note the wiki itself flags some figures, e.g. alignment shift amounts, as still incomplete).

### What changes with a hard cap of 5 creatures

Everything above — and everything in §2–§5 — gets meaningfully easier at 5 individuals instead of hundreds or thousands:

- **Chao-fidelity**: the full multi-channel drive/emotion system above (10 named values per creature, not 1 blended scalar), real stat-influence evolution, alignment, and even lightweight genetics/breeding between the 5 are all cheap at this scale — this was never really an option for a 3,000-particle garden, and now it plainly is.
- **The psychosocial paper (§2)**: its own test scenarios used **9 named characters** — almost exactly this scale. A 5-creature garden can implement the actual Actor/Reactor system faithfully (per-pair social ties, not just a shared species matrix) rather than the current cheap aggregate contagion.
- **Generative-agents-style memory (§3)**: the recency/importance/relevance retrieval formula, run as a real per-creature short event log (no LLM needed, just the scoring function) is trivial to keep for 5 creatures and was never realistic for thousands.
- **The fly connectome (§4)**: this is the big one. Feasibility tiers B and D — "one creature runs a small real circuit" or even "a creature runs the full 139k-neuron graph" — assumed *one* specimen because that's all that was affordable. At a hard cap of 5, **running an actual small-to-medium real circuit per creature, all five at once, stops being a stretch.** Tier F (one shared connectome as a command source) is still the cheapest and most proven route, but tier B (real neurons, one circuit, per individual) is now realistic for the *entire* population, not a single showcase specimen — closer to "each chao has a real brain" as originally asked, rather than the compromise version.
- **Rendering (§5)**: 5 detailed low-poly PS1/PS2-style creature meshes is a completely different, much smaller task than thousands of point sprites — full per-creature geometry, individual animation, and the whole retro shader pipeline are all in reach without a performance fight.

The one structural idea to carry forward from here: a fast, visible mood value and a slow, permanent trait value, kept as two separate quantities with two very different time constants. That split is the emotion ball versus evolution, and everything in §1a's drive system is a richer version of it.

---

## 2. Academic paper 1 — emergent psychosocial NPCs

**Bailey & Katchabaw, "An Emergent Framework for Realistic Psychosocial Behaviour in Non Player Characters," FuturePlay 2008.**

Core idea: bottom-up, not scripted. Every character is a **psychosocial object** with:
- Internal state: a personality model (they use *agreeableness* + *dominance*), an emotion set (Ekman's six: anger, fear, disgust, surprise, sadness, joy), needs/values.
- Social state: ties to other characters (strength + polarity, symmetric or not), group memberships, a "social influence modifier" (peer pressure / groupthink knob).

Mechanism: a generalized **Actor/Reactor stimulus-response system**.
- An **Actor** broadcasts a stimulus: type, magnitude, propagation method, and a **fall-off radius**.
- **Reactors** in range run it through a Relevance Filter → Stimulus Dispatcher → Decision-Maker, which consults their own traits to pick a response (or ignore it).
- Chain reactions are expected and desired: a reaction can itself become a new stimulus.

Their test scenarios (worth reproducing as a sanity check for any system built this way):
- **The Angry Individual**: one hostile, dominant character dropped into a happy friend group. Agreeable/less-dominant characters turned sad near him; disagreeable/less-dominant ones turned fearful; equally-dominant ones turned angry back. Everyone reverted to happy near a friend.
- **A Friend in Need**: one sad character surrounded by happy ones was cheered up almost immediately by proximity; the reverse case (one happy character in a low-mood crowd) showed joy visibly *diffusing outward* from the source over several simulated moments, radius by radius.

At 5 creatures this paper can be implemented as written: per-creature dominance and agreeableness fixed at birth, ten per-pair ties with strength and polarity, and stimuli with a fall-off radius. The decision step (relevance filter, then a trait-consulting response choice) is the part that makes the "angry individual" scenario work; a plain mood-averaging contagion cannot produce "equally dominant ones turned angry back," so build the decision step, not just diffusion.

---

## 3. Academic paper 2 — Generative Agents (Smallville)

**Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," arXiv 2304.03442 (2023).**

25 LLM-backed agents in a Sims-like sandbox. Three-part architecture:

1. **Memory stream** — every experience logged as a natural-language sentence.
2. **Retrieval function** — scores memories for what to surface right now:

   ```
   score = recency + importance + relevance
   recency    = 0.99 ^ (hours since last accessed)      — exponential decay
   importance = 1–10, scored by the LLM at creation time  (mundane=1, poignant=10)
   relevance  = cosine similarity between memory embedding and current query
   ```
   All three weighted equally (α = 1.0); top-scoring memories that fit the context window are pulled.
3. **Reflection** — triggered when the sum of recent importance scores crosses **150 points** (roughly 2–3×/day). The agent asks itself "what are the 3 most salient high-level questions I can answer about the subjects in these statements?", retrieves memories against those questions, and synthesizes higher-level insights that cite their source memories. This builds a **tree**: raw observations at the leaves, reflections on reflections above them.
4. **Planning** — recursive: a one-paragraph day sketch → hourly blocks → 5–15 minute actions, replanned when something observed doesn't fit.

Emergent results: a single seeded intent ("Isabella wants to throw a Valentine's Day party") reached 13 of 25 agents (52%) through organic conversation over two in-sim days; a piece of election gossip spread from 1 agent to 8 (32%); the overall relationship-graph density rose from 0.167 to 0.74; the party happened on the day, including agents who forgot invitations or double-booked themselves — failure was part of the realism.

**Reality check:** this whole system runs *on* an LLM (each reflection/plan step is a real language-model call). Not something to run per-creature, client-side, for a garden of thousands. What's portable without an LLM: the **retrieval formula itself** (recency/importance/relevance scoring) as a lightweight per-creature ranking function over a short event log, and the general shape — accumulate → periodically synthesize into a durable trait → let the trait steer behavior — which is the same fast-state-into-slow-trait shape described at the end of §1a.

*Review:* two holes in "portable without an LLM." Relevance in the paper is embedding cosine similarity; with no LLM there are no embeddings, so the term needs a replacement (likely a tag match between event type and the creature's current dominant drive) or should be dropped. Importance likewise needs a heuristic, e.g. magnitude of drive change at the moment of the event. Reflection has no cheap analogue and should be skipped. Also: the LLM rejection above was written when the population was thousands. At 5 creatures and one call every few minutes, scale is no longer the reason to reject it. The reason now is that an ambient toy should not depend on a network or pay latency per decision. Either argument lands in the same place, but the doc should make the current one.

The behavioral payoff also needs to be concrete or the whole memory system is YAGNI. Two uses that justify it: a creature remembers where food appeared and heads there when hungry; a creature remembers who bullied it and avoids them (which is the per-pair tie from §2, stored as a memory rather than a matrix).

---

## 4. The fly connectome scene

### What actually exists
- **MaleCNS** (Google/Janelia, complete male fly CNS, released 2024–2026): **166,000 neurons, ~125 million synapses**. A genuine large scientific dataset — distributed via Neuroglancer/neuPrint/Codex, meant for research browsing, not embedding in a page.
- **FlyWire/FAFB** (adult female brain): the earlier, more widely-used dataset most hobby projects actually build on.

### What the browser demos actually run
People are *not* simulating the full 125M-synapse graph for fun. The pattern, confirmed by pulling one real project's data (`snedea/flybrain`, FlyWire FAFB v783):
- Pruned to **139,255 neurons / ~2.7 million synapses**.
- Simulated as **leaky integrate-and-fire (LIF)** neurons, in a **Web Worker**, rendered with **WebGL** (activity visualization across the whole network, color-grouped by region).
- Ready-to-simulate connectivity file: **12.4MB gzipped** — for *one* fly. That's most of an Artifact's page budget, and it's one instance, not one per creature.
- Inputs (food, touch by body region, light level, temperature) excite specific real sensory neurons; behavior (seeking, startle, phototaxis) **emerges from signal propagation through the real wiring** — nothing scripted.
- Important caveat: the connectome is a **fixed architecture, not a learner**. Several of the flashiest demos (Beat Saber, a Chrome-Dino player, a Pong paddle) bolt a *separate* trained readout (PPO/REINFORCE/CEM) on top of the fixed connectome to turn raw spike patterns into competent actions — the "brain" provides the substrate, not the competence.

### The wider hobby ecosystem (from a curated list of ~60 real projects)
Real, working precedent at every scale:
- **Full 139k neurons, real-time, one instance**: `webgpu-fly` (WebGPU/WASM), `FastFly` (CUDA), `Connectome OS` (Rust runtime).
- **Small curated real circuits, many instances feasible**: `Fly Dino` (an 80-neuron real circuit drives a Chrome Dino clone), `gnat` desktop pet (668-neuron subset), `Swat` (a 6,000-neuron real escape circuit drives evasion in an arcade game), `CHIMERA` (the *larval* fly connectome, only 1,373 neurons, driving a MuJoCo body).
- **Desktop-pet / ambient framing** (closest to this project's tone): `DesktopFly`, `gnat`, `Infinite Sugar` (a browser terrarium art piece — FlyWire-based fly, sweet-sensing, ambient).
- Research frameworks if this ever gets serious: `flybody`/`NeuroMechFly` (MuJoCo fly body + RL), `flyvis` (PyTorch connectome-constrained visual system model), `train-your-fly` (PyTorch Geometric FlyWire graph toolkit).

### The other workflow — a real one, straight from the source (r/robots thread)

The actual post (pasted directly by the user, since Reddit isn't fetchable from here):

> This fruit fly has been dead for years, but today, its brain moved a robot around my apartment. I downloaded **MaleCNS**, a reconstructed connectome of an adult male fruit fly containing **166,700 neurons and 25.6 million connections**, and got the whole thing running live on my MacBook. Then I used Codex to modify the desktop app for my quadruped robot. The setup is basically: robot desktop app ↔ Wi-Fi ↔ robot, and now: digital fly brain → desktop app → Wi-Fi → robot. The fly brain continuously runs on my MacBook. I read neural activity from it, translate that into high-level commands like FORWARD / LEFT / RIGHT, and send those commands over Wi-Fi to the physical robot. So the robot's legs are still controlled by its normal firmware — the fly brain isn't individually controlling 12 servos. It's producing the high-level decisions that get translated into robot movements.

This is a materially different — and much more buildable — pattern than the MuJoCo/NeuroMechFly pipeline in the previous writeup, and it sidesteps that pipeline's whole problem:

1. **Dataset**: MaleCNS, **166,700 neurons / 25.6 million connections** (note: Google's own blog post quotes 166,000 neurons but ~125 million synapses for the same release — likely a different connection-strength threshold or counting method between "raw synapses" and "connections used for simulation"; worth not treating either number as gospel). *Review:* nothing in this project needs MaleCNS. Every small-circuit tool and demo cited here (Fly Dino, gnat, flybrain, train-your-fly) is built on FlyWire. Pick FlyWire and stop tracking the MaleCNS count discrepancy.
2. **Run it live**, continuously, as its own process — same LIF-style simulation as the browser demos, just on a MacBook instead of in a Web Worker.
3. **Read out activity, don't wire up muscles.** Instead of trying to map brain regions to servos/muscles (the part that made the MuJoCo pipeline shaky — "had to make educated guesses"), this workflow reads aggregate neural activity and **classifies it into a small, discrete command vocabulary**: FORWARD / LEFT / RIGHT (presumably plus STOP/idle).
4. **The body stays exactly what it already was.** The robot's own firmware still walks it — the connectome never touches low-level motor control, only feeds it high-level intent. The existing "desktop app ↔ Wi-Fi ↔ robot" link didn't change; a fly brain was just spliced in as the new *source* of commands, in place of whatever was issuing FORWARD/LEFT/RIGHT before (presumably a human with a controller).
5. **Codex was used to adapt the existing desktop app** to consume this new command source — glue code, not a rewrite.

This is a much better fit for a browser garden than the MuJoCo route: **it decouples the brain from the body entirely.** You don't need a physically simulated fly, muscle-mapping guesses, or a research-grade physics engine — you need one running connectome instance producing a thin stream of discrete decisions, which then drives whatever movement system already exists (in this case, Quiet Garden's existing force-based physics). That reframes the whole "give each chao one of these" question: instead of simulating a full brain *per creature* (computationally impossible at thousands of instances), a handful of shared connectome instances could each produce a slow trickle of high-level commands (seek food / flee / idle / approach) that bias many creatures' behavior — closer to how the robot demo actually works than anything in the feasibility table below originally assumed.

### Feasibility tiers for "give each chao one of these"

*Superseded 2026-09-16.* This table was written for a browser deployment with a page-size budget. The project is now a native app (§8), which makes tier D, one full brain per creature, the plan for all five. Kept for the record.
| Tier | What it is | Real-connectome fidelity | Scales to a garden of thousands? |
|---|---|---|---|
| A | Current: hand-authored force rules | None | Yes, trivially |
| B | One "wild-caught" creature runs a small real circuit (Fly Dino/gnat scale, tens–hundreds of real neurons) | Real, but a fragment | No — one specimen |
| C | Every creature shares one small real circuit (same weights, per-creature activation state) | Real, reduced | Yes |
| D | One creature runs the full 139k-neuron FAFB graph, WASM/WebGPU | Real, full | No — one specimen, and heavy |
| E | Skip neurons; psychosocial-paper-style personality/emotion instead | None (but honest) | Yes |
| F | **One (or a few) full/large connectome instance(s) run continuously, read out as discrete high-level commands** (the robot-demo pattern) that bias many creatures' existing physics | Real, full-scale, decoupled from body | Yes — the connectome is shared infrastructure, not per-creature cost |

*Review:* the fidelity column treats "real" as binary. An 80-neuron fragment and the 139k graph both read "Real." Relabel honestly: a fragment is real wiring with hand-chosen inputs and outputs, and the hand-choosing is where the behavior actually comes from.

### Tier B, checked: what §1a and §6 assume but never verify

The 5-creature section calls five real circuits per creature "realistic" and §6 calls it "the cheapest brain upgrade." Neither is established. Three unknowns, any of which could make tier B the most expensive option:

1. **Which circuit, extracted how.** An 80-neuron fragment has no sensory periphery of its own. Someone has to pick which neurons receive "food nearby" or "another creature approaching" and which neurons are read as output. That is exactly the "educated guesses" problem this doc criticizes in the MuJoCo pipeline. The robot demo avoided it by running the whole brain, where real sensory neurons exist.
2. **Readout.** The robot post says activity was "translated" into FORWARD/LEFT/RIGHT but not how. The flashier demos (Beat Saber, Dino, Pong) bolt a *trained* readout on top. If tier B needs a trained decoder per circuit, that is a real ML task not in any scope estimate here.
3. **Five copies of the same weights with different activation state is tier C, not tier B.** The 5-creature section conflates them. Tier B as written (a different circuit per creature) multiplies unknowns 1 and 2 by five.

**Pick the circuit by function, not by which demo used it.** The hobby projects chose subsets by size or convenience. The fly neuroscience literature names small, well-characterized circuits with legible outputs, and two of them are exactly what a garden creature needs:

- **Central complex ring attractor** (Seelig & Jayaraman 2015; ellipsoid body E-PG/P-EN loop, low hundreds of neurons): maintains a heading bump that rotates with the animal's turns. The bump position *is* the readout: decode its angle, compare to the direction of the current goal, and you have turn-left/turn-right with no trained decoder. This is the honest answer to unknown 2.
- **Mushroom body** (Kenyon cells plus a handful of output neurons and dopaminergic reinforcement neurons): associative learning. Pairing "this creature fed me" or "this spot had fruit" with reward would give *learned* food and person preference from real plasticity rules, which is the one thing a fixed connectome cannot otherwise do and the thing that would make feeding matter.

**Falsification test, required before shipping the claim.** Run the same garden with a shuffled-weights network of identical size in place of the real circuit. If a viewer cannot tell the difference over ten minutes, the connectome is a label, not a mechanism. Keep the feature if it looks good; drop the "real fly brain" claim.

**Legibility.** Nothing here asks how a viewer would know *why* a fly-brained creature did something. Without a visible readout (the emotion ball, a heading indicator, a tiny activity strip when the creature is selected) real neurons and noise are indistinguishable to the audience. See §7 on aliveness.

---

## 5. What a PS1/PS2-era look would actually take

### Correction first: Chao Garden was never on a PlayStation

*Review.* Sonic Adventure (1998) and Sonic Adventure 2 (2001) are **Dreamcast** games, later ported to GameCube (SA2 Battle 2001, SADX 2003) and PC. Neither shipped on PS1 or PS2. The Dreamcast's PowerVR2 had a real z-buffer, **perspective-correct texturing**, 640×480 output, bilinear filtering, and smooth Gouraud shading. It had **no vertex wobble and no affine warping.** Every signature PS1 artifact recipe below (vertex snap, affine UV hack, 320×240 internal resolution) would make the garden look *less* like the reference, not more.

The look the reference actually has is sixth-generation: low poly counts, small textures with bilinear filtering, distance fog, flat storybook palettes, clean geometry, simple specular. That is roughly what the "PS2" paragraph below describes, minus the "shiny plastic," and it is also a much smaller shader job (fog, low res textures, low poly, a mild posterize) than the PS1 pipeline.

If PS1 jitter and dither are wanted anyway as a deliberate stylistic choice, that is a legitimate call. It just needs to be made knowingly, and stated, rather than inherited from a wrong assumption about "the era." Recipe items 1 and 3 below (vertex snap, affine warp) are PS1-only; items 2, 4, 6, 7 apply to both.

### Why the era looks the way it does
- **PS1 vertex "wobble"**: not the fixed-point math itself (a common myth) but **no sub-pixel rasterization precision** — vertices snap to the pixel grid every frame, so as geometry moves/rotates it visibly shivers between grid positions. More pronounced at the era's native output resolution, 320×240.
- **PS1 texture warping**: **affine texture mapping** — texture coordinates interpolated linearly in screen space instead of perspective-correct. Distorts badly on large flat surfaces at steep angles; the reason old-school level design keeps big floors chopped into extra triangles.
- **PS1 had no depth buffer** — primitives were sorted back-to-front by hand (painter's algorithm via ordering tables), causing occasional flicker/pop where sorting was ambiguous. (Usually skipped in modern homages — a real z-buffer looks fine and nobody misses the flicker.)
- **PS2** fixed the vertex wobble (real floating-point transforms) and pushed polygon counts and texture/lighting quality way up (specular highlights, environment mapping, particle effects), but kept comparatively low native resolution, heavy use of fog for draw-distance, and a distinct "shiny plastic" look. A PS2 aesthetic is really "PS1 without the wobble, plus shine and fog."

### Concrete browser recipe (sourced from working Three.js implementations)

**1. Vertex snapping** (the signature PS1 jitter) — cheap, portable, in the vertex shader:
```glsl
vec4 pos = projectionMatrix * mvPosition;
vec2 resolution = vec2(320.0, 240.0);   // lower = more jitter
pos.xyz /= pos.w;
pos.xy = floor(resolution * pos.xy) / resolution;
pos.xyz *= pos.w;
gl_Position = pos;
```
*Review:* if kept, make `resolution` a live knob and default it well above 320×240. The era's wobble was tolerable partly because cameras were fixed or slow; under a user-driven orbit camera, full-strength snap can be nauseating. Also honor `prefers-reduced-motion` by disabling snap, dither, and any scanline pass.

**2. Low internal render resolution + nearest-neighbor upscale** — the other half of the signature look:
- Render to an off-screen target at ~320×240 (or 2–4× that), then blit to the full canvas with `NEAREST` filtering (or, in p5.js, `createGraphics(320,240)` → draw the 3D scene into it → `image()` it scaled up with `noSmooth()`).
- CSS `image-rendering: pixelated` on the canvas reinforces crisp scaling.

**3. Affine texture warping** — the annoying one. WebGL2/GLSL ES 3.00 has **no portable `noperspective` interpolation qualifier** (that's a desktop-GL feature some native PS1-style renderers rely on); the browser-compatible trick multiplies UVs by a distance-derived "affine factor" in the vertex shader and divides it back out in the fragment shader, faking the perspective-incorrect interpolation:
```glsl
// vertex
float affine = dist + (pos.w * 8.0) / dist * 0.5;
vUv = uv * affine;
vAffine = affine;
// fragment
vec2 uv = vUv / vAffine;
```
Commonly the hardest piece to get looking right, and the first thing hobby projects skip.

*Review:* cut it. It is only visible on large flat surfaces at steep angles, and the era's own fix (noted above) was to chop big floors into more triangles. Subdivide the ground mesh and skip the shader hack. Same look, zero fragile code. Also moot if the Dreamcast correction is accepted.

**4. Dithering / posterization** — reduce color depth (PS1 was ~15-bit color) via an 8×8 Bayer-matrix ordered dither in a post-process fragment shader; cheap, big visual payoff, standard shadertoy-grade recipe.

**5. Flat/cheap shading** — per-face lighting instead of smooth Gouraud/specular (GLSL `flat` qualifier, or compute lighting per-face in the vertex shader off the face normal).

**6. Fog** — linear or exponential distance fog blending to the background color; doubles as free draw-distance culling and is very on-era (both PS1 and PS2 leaned on it).

**7. Low-poly meshes, low-res/no-mipmap textures** — nearest-neighbor texture filtering, small textures (64–256px), modest triangle budgets.

**8. Optional bonus (more "CRT" than "PS1 GPU," but usually paired anyway)**: scanlines, slight chromatic aberration, vignette, capped low frame-rate.

### p5.js vs. raw WebGL2 vs. Three.js, for this specific job
- **Raw WebGL2**: a real 3D garden implies meshes, a camera, depth, and per-face lighting — which means hand-rolling matrix math, mesh generation, and a render pipeline that a library already solves well. This is the one place reaching for a library is the lazy choice, not the over-engineered one.
- **Three.js**: the natural fit for a real low-poly 3D diorama — has camera/mesh/material plumbing built in, `EffectComposer` for the post-process passes (pixelation, dithering), and is exactly what all the sourced PS1-shader recipes above are written against. Loadable from cdnjs as a pinned UMD build. *Review:* verify this before assuming a one-line script tag. Three.js dropped its UMD build around r160 and ships ESM only since; cdnjs still hosts older UMD versions, but `EffectComposer` and the other post-process passes live in `examples/jsm/` and need an import map (jsdelivr `three@<ver>/build/three.module.js` plus `three/addons/`). Either pin a pre-r160 UMD release and hand-write the post-process passes, or use the ESM import-map route. Decide once.
- **p5.js (WEBGL mode)**: lighter-weight, sketch-style API, and notably makes the "low internal resolution + nearest-neighbor upscale" trick a one-liner (`createGraphics` + `noSmooth()`). Custom shaders are supported (`createShader`/`shader()`), so vertex snapping and dithering are still reachable, but there's a much thinner trail of existing PS1-recipe code written against p5 specifically compared to Three.js — more assembly required.

### Correction: the real Chao Garden is not top-down — it's a free 3D camera

Worth stating plainly, because a flat top-down 2D canvas is the wrong reference. Checked directly:

- **Camera**: full third-person 3D, player-controlled (you walk your human character around the garden), and separately steerable — right stick rotates it, D-pad pans it up/down, a button freezes it, L+R together resets it to default. Nothing about it is fixed or top-down; it's much closer to a small open 3D playground than a diorama viewed from above.
- **Real asset scale, confirmed from an actual ripped model**: the entire Station Square garden — full terrain geometry *and* every texture (marble floor, water, grass, rock, flowers, building facades, nursery hut) — is a **189KB download** as OBJ/MTL. That's the real budget a PS1/PS2-era environment lived inside, and it's a genuinely useful target number for how little geometry/texture data this needs, not just an aesthetic to imitate.
- **Each garden is a distinct, small biome**, not one generic green lawn:
  - *Neutral Garden*: green, Green-Hill-Zone-like grassy park, a waterfall, a cave mouth leading to the race stadium.
  - *Hero Garden*: yellow sky with pink clouds, two small floating islands (one with a bell), a pool, a small fountain — a bright, storybook-sky palette, distinctly not grounded/realistic.
  - *Dark Garden*: grey rocky mountainside, twisted dead trees, gravestones, red-tinted water, a black fence ringing parts of it — genuinely macabre, not just "the same garden with darker lighting."
- **The Chao model itself**: soft round blob body, oversized simple eyes, small nub limbs (or fins/wings/etc. once animal parts and evolution kick in), and the floating "emotion ball" above the head as its own small separate geometry that changes shape with mood — worth remembering as its own mesh/element when modeling a creature, not just a texture on a sphere.

This matters beyond aesthetics: a faithful version needs an actual 3D scene with a real (if simple) camera rig, not a camera-less flat canvas with a shader over it.

**Deliberate deviation from the reference**: no playable human character. The real garden's camera is a third-person follow-cam tied to whichever character you're walking around as; this version drops the avatar entirely. The garden stays a real, walkable-looking 3D space (§ above), but navigation becomes an **orbit/pan camera** the viewer drives directly (drag to rotate, scroll to zoom — the standard free-look scheme, no character in frame at all), and interaction is **click-to-select on the creatures themselves** (raycast from the mouse into the 3D scene) rather than walking a body up to one and opening a menu. Feeding, petting, or whatever else stays interactive, but the verb is "click the chao," not "walk to the chao."

### Scope reality check
The build is a real 3D diorama from scratch (camera, ground plane, low-poly creature meshes, lighting, fog, optional post-process pipeline). There is no 2D "lite" tier; that option existed only as a filter over an earlier flat prototype that has since been dropped, and per the correction above it was never a step toward the reference anyway.

**The build**: Three.js as a real 3D scene — a small garden (terrain, water, a couple of landmark props, at real-precedent asset budgets: think tens to a couple hundred KB of geometry+texture, not megabytes), an orbit camera the user can actually rotate, and low-poly creature meshes (body + separate emotion-ball mesh) with fog, low-res textures, and whichever era treatment §5's correction settles on. With the 5-creature cap (§1a), the creature side of this is very affordable; the environment/camera side is the real new work.

---

## 6. Where this leaves the four inputs together

*Superseded 2026-09-16 by §8.* The four builds below assumed a browser page with the fly brain as an optional layer. The fly brain is now the core and the platform is native. The sequencing argument in the review note at the end of this section still holds in spirit (prove the look, then the minimal body layer, then deepen) and is restated for the new architecture in §8. Kept for the record.

Updated for the settled constraints: **5 creatures max**, a **real 3D scene with an orbit/pan camera** (no avatar, click-to-select creatures directly), PS1/PS2-era rendering. At this population size almost everything upstream in this doc is individually affordable — the real choice is which systems to layer, not what's feasible. Four coherent builds, roughly increasing in scope:

**1. Diorama first — get the look right, keep the brains simple**
Build the actual thing that changed today: Three.js scene, small garden environment at real-precedent asset budgets (§5), orbit/pan camera, click-to-select on 5 low-poly creature meshes (body + separate emotion-ball mesh), full PS1/PS2 shader pipeline (vertex snap, affine UV, dither, fog). Behavior is the minimum that makes five creatures look alive: Reynolds steering (wander, seek, separate) plus a fast mood and a slow trait per creature. Lowest risk: the visual style just got corrected and hasn't been proven out yet, so this validates it before building anything on top.

**2. Chao-fidelity brains, same diorama**
Same 3D scene as (1), but replace the abstract kinship matrix with the actual sourced Chao mechanics (§1a): stat-influence sliders, real evolution, alignment, the multi-channel emotion/drive system (Hunger, Boredom, Energy, Joy, Fear, etc.), personality traits that modulate decay rates. Click-to-feed / click-to-pet becomes the real interaction verb, driving alignment and stat-influence the way the source game does. This is the deepest fidelity to the actual inspiration, and only realistic because the cap is 5, not thousands.

**3. Add real fly brains — the actual "each chao gets one" answer**
Layer onto either (1) or (2): give each of the 5 creatures a real, small-to-medium connectome-derived circuit (§4 tier B — tens to low hundreds of real neurons, not the full 139k graph) whose output steers its behavior, following the robot-demo pattern (§4: run it live, decode activity into a handful of discrete high-level commands, feed those into the existing movement/interaction system rather than trying to wire it to "muscles"). Five real brains running at once is the thing that was never affordable before the population cap dropped to 5 — this is the option that actually delivers what was originally asked for.

**4. Everything**
(1) + (2) + (3), plus the two cheap remaining pieces: per-pair social ties and mood contagion from the psychosocial paper (§2 — now genuinely per-relationship instead of aggregate, since 5 creatures means at most 10 pairs to track) and a real per-creature memory log scored by the recency/importance/relevance formula (§3) to decide what each one "remembers" wanting. Every research thread represented, all at a population size where none of it requires a compromise — but it's a real multi-system build, not an afternoon.

Recommended sequencing if unsure: **1 → 3 → 2 → the rest of 4**. Prove the 3D look works and is fun to look at before investing in either brain system; add real fly brains next since that's the most novel/differentiating piece and the cheapest of the two brain upgrades to bolt onto simple movement; save full Chao-mechanic fidelity (stat sliders, evolution, alignment) for after, since it's the most content/tuning-heavy piece and least dependent on anything else being done first.

*Review: revised sequencing, **1 → minimal 2 → 3 → rest.*** Three problems with the order above. First, option 1 with the kinship matrix ported is a screensaver: there is nothing to click on, and the emotion-ball mesh has no state to display. The smallest Chao layer (hunger, one fruit, Joy and Fear fast reactions, the emotion ball driven by them) is what makes click-to-select mean anything, and it is a day of work, not the full §1a transcription. Second, "cheapest brain upgrade" for fly brains is unsupported; per §4 "Tier B, checked," it may be the most expensive because of the readout problem. Third, "least dependent on anything else" for Chao mechanics is backwards: click-to-feed needs the raycast UI from option 1, and the emotion ball needs a drive system behind it. They are coupled, so build them together.

Also: **keep Chao structures, rescale Chao numbers.** Opposed sliders, slow and fast drive tiers, personality modulating decay rates. Those are the design. 3,266 max points and 20 animals per slider are tuned for hours of grinding toward a race payoff this garden does not have. Transcribing them is wiki fidelity, not design. Pick a session length first (§7), then scale every rate to it.

Option 1 also understates the movement work: terrain following, facing direction, water regions, idle animation, and touch handling for orbit plus raycast if this ships on phones. Lazy version: flat ground plane, props as decoration with no collision, creatures move in XZ, height from a lookup, orientation from velocity. Use Reynolds steering (seek, flee, wander, arrive, separation) for movement; it is the standard middle layer and every drive maps to a steering target.

---

## 7. Review findings: gaps, blindspots, open decisions

Added 2026-09-16 after a read-through of §1–§6. Three groups: things the doc looked at but left holes in, whole domains it never looked at, and decisions that have to be made before any numbers downstream mean anything.

### 7a. Gaps in what was researched

- **No goal statement.** The header says no decisions; §6 lists settled constraints. Nowhere does it say what this is for, who looks at it, or how long a session lasts. Every rate in the doc (evolution, alignment drift, memory decay) depends on session length.
- **No non-goals.** Races, karate, breeding, naming, death, save transfer are all silently in or out. Breeding in particular conflicts with a hard cap of 5 and is a design smell at that population. Recommend: out.
- **No persistence model.** Every slow system is meaningless if a reload resets to zero. Chao Garden lived on a memory card. Options: localStorage snapshot; or deterministic seed plus "advance the sim by wall-clock since last visit" on load. See 7b on background tabs.
- **Tier B unverified.** Covered in §4 "Tier B, checked."
- **Memory retrieval holes.** Covered in §3 review note.
- **No performance budget.** Five LIF circuits at what tick rate, main thread or Worker, what internal resolution, what target device. One line each is enough.
- **No success test.** Proposed: leave it open ten minutes. Did something happen you didn't script and wanted to tell someone about? That is the Chao Garden test. For fly brains, the shuffled-weights control in §4.
- **No tuning plan.** Ten drives across five creatures with contagion cannot be tuned by watching in real time. Needs a headless fast-forward sim, deterministic seeds, and plots of drive values over simulated hours. Every emergent-behavior project dies here.
- **No audio.** Chao Garden's ambience is half sound. Zero mentions anywhere above.
- **Style.** The doc runs on em dashes, roughly a hundred. House rules forbid them. Run through chrisnizer before it leaves the machine.

### 7b. Blindspots: domains never looked at

- **Wrong hardware.** Chao Garden is Dreamcast, not PlayStation. See §5 correction. This undercuts most of §5's recipe.
- **Creatures (1996).** Steve Grand's Norns had real neural-network brains, a biochemistry of drives, and genetics, running per-creature on 1996 consumer hardware, shipped as an ambient pet world. It is the direct precedent for "each creature has a real brain" and is far better documented than any fly-connectome hobby project (Grand's *Creation: Life and How to Make It*; the Creatures Wiki brain-lobe and biochemistry pages). Also unread: Black & White's creature (learns from the player), Nintendogs and Tamagotchi (wall-clock time), Viva Piñata (garden ecosystem), and The Sims needs system, the canonical shipped version of "drives with decay and action selection."
- **Utility AI and steering behaviors.** The doc jumps from particle-life forces straight to connectomes. The standard middle layer for creature movement is Reynolds steering; the standard frame for "drives pick actions" is utility AI with response curves (Dave Mark, *Behavioral Mathematics for Game AI*). §1a's drive system *is* utility AI. Reinventing it from a datamined wiki will cost tuning time.
- **Aliveness comes from legibility and projection, not fidelity.** The doc's implicit bet is that realer brains feel more alive. The design literature says the opposite: viewers project intent onto anything with readable state and slight unpredictability (apophenia; the Eliza effect). The Sims thought bubbles, the Chao emotion ball, Creatures' brain viewer all exist to make internal state readable. Design for the readout first; the mechanism behind it is secondary to how it feels.
- **Circuit selection by function.** Covered in §4: ring attractor for heading, mushroom body for learned preference.
- **Asset pipeline and IP.** Nobody asks who makes the meshes or how (Blender export, procedural geometry in code, or squash-and-stretch on primitives with separate eye meshes). The 189KB garden model cited in §5 is a ripped Sega asset and cannot be used for anything but a size reference. "Chao" is a Sega trademark; the doc uses it as a common noun throughout. Fine for a private toy, not for anything published. Picking a different creature resolves this; see 7c.
- **Time when the tab is closed.** Browsers freeze `requestAnimationFrame` and throttle timers to about once a minute in background tabs; Workers are throttled too. An "ambient garden" that only lives while focused is a screensaver. Tamagotchi and Nintendogs run on wall clock and catch up. The Dreamcast solved this with the VMU Chao Adventure minigame that ran while you were away. Decide: frozen, catch-up on focus, or coarse background tick.
- **Reduced motion and color.** Dither, wobble, scanlines, and capped framerate are migraine triggers; honor `prefers-reduced-motion`. Warm-versus-cool tint as the mood signal fails for common color blindness; shape and motion should carry mood too, which the emotion ball already does.
- **The LLM rejection predates the 5-cap.** Covered in §3 review note.

### 7c. Decisions (status as of 2026-09-16)

Settled, recorded in the Goal block at the top: goal and audience (ambient desktop app, ducks driven by AI with emergent behavior and personality, fly brain as hook, aesthetic and gameplay for stickiness); platform (native, Python brain server plus Godot or Tauri diorama, §8); creature (microducks); non-goals (breeding, races, karate, permanent death, naming UI, multiplayer, win state, vision).

Still open:

1. ~~Goal and audience.~~ Settled.
2. **Session length and offline time.** Days, since it is a desktop app that can stay resident. Still to decide: does the brain keep running while the window is closed (menu-bar process, like the robot demo's always-on MacBook brain) or catch up on launch? Recommend catch-up on launch first; always-on is a later upgrade.
3. **Era.** Dreamcast-faithful (fog, low poly, bilinear, clean) or deliberate PS1 jitter as a stylistic choice? Recommend Dreamcast-faithful: it is the actual reference and half the shader work.
4. ~~Non-goals.~~ Settled.
5. ~~Creature identity.~~ **Microducks**, settled. The original case for them, kept for the record: In favor: a duck is a round body, a bill, two nub feet and stub wings, which is the same low-poly budget as a Chao blob and fits any era in §5; ducks give a natural reason for the water regions Chao Gardens already have (swim is a real Chao stat and a real duck behavior); waddle and paddle are two of the easiest procedural animations to make read; ducklings imprint, which is a legible, sourced hook for the alignment-by-treatment mechanic in §1a (who fed you shapes who you follow); and the mushroom-body learning circuit in §4 maps cleanly to "learned who feeds me." Against: nothing structural. The Chao evolution-shape system (run vs. power, swim vs. fly body morphs) would need duck-flavored equivalents (diver vs. dabbler, flyer vs. runner), which is a naming exercise, not a mechanics change. Nothing in §1–§6 depends on the creature being a Chao specifically; every mechanic transfers.
6. **Fly brain: keep the claim or keep the feature?** Decide after the shuffled-weights control, not before. The brain is now the core mechanism (§8), so the honest fallback if the control wins is "connectome-derived network," not "real fly brain."
7. ~~Three.js loading route.~~ Moot if the diorama is Godot. Still applies if the Tauri route is taken; see §5.
8. **Renderer: Godot 4 or Three.js inside Tauri.** Recommend Godot: orbit camera, raycast, fog, per-material filtering, save files, and Mac/Windows export are built in, and the project is greenfield anyway. Tauri only if staying in web tech is worth more than those.
9. **Brain compute path.** PyTorch sparse on MPS (Mac), CUDA (Windows/Linux), or a Rust runtime (Connectome OS precedent). Recommend PyTorch first because every FlyWire data tool is Python; move the hot loop to Rust only if five brains at 1 kHz don't hold real time.
10. **Sensory set.** Recommend the minimum that produces readable behavior: gustatory sugar neurons (food contact), olfactory receptor neurons (food odor gradient), mechanosensory (pet, bump), looming detectors (something approaching fast). No vision.

### 7d. Lessons from the dropped prototype

A WebGL2 particle-life sim (thousands of point sprites, a random kinds-by-kinds kinship matrix, one blended mood scalar) was built early, reviewed, and dropped as not the vision. Nothing from it is ported. Three mistakes it made are worth not repeating in the greenfield build:

- **Rates must be per-second, not per-frame.** The prototype's mood decay, trait drift, and friction were all per-frame, so a 120Hz display ran it twice as fast. Use a fixed-timestep accumulator or a `dt` multiplier from the first commit. Every tuning number in §1a depends on it.
- **Mood contagion needs valence, not just diffusion.** Averaging toward neighbors' mood can only homogenize. The psychosocial paper's "angry individual" case (agreeable neighbors turn sad, equally dominant ones turn angry back) requires per-pair polarity and a decision step (§2).
- **Seed the randomness.** Without a seeded PRNG and the seed in the URL hash, no run is reproducible and the tuning plan in 7a is impossible. First thing in the new build.

Also confirmed by the prototype: a slow "lifetime" trait defined by a neighbor-count threshold saturates for everyone in seconds. Define slow traits by accumulated events (fed, petted, bullied), not by ambient crowding.

---

## 8. Direction: native app, one full fly brain per duck

Decided 2026-09-16. Replaces the §4 tier table and the §6 builds.

### 8a. What it is

Five microducks, each a puppet driven by its own running copy of the FlyWire adult fly brain. The duck body is the robot in the r/robots post (§4); the garden is the apartment. The brain does not know it is a duck, and the project says so. The product is the emergent behavior and the visible personality differences; the fly brain is the hook and the mechanism; the Dreamcast look and the light gameplay are what make it sticky.

### 8b. Why native

The browser forced a few-hundred-neuron fragment (§4 tier B) and a hand-built readout. A desktop app removes both ceilings. FlyWire v783 is 139k neurons and 2.7M synapses, about 12 MB compressed, and one instance runs in real time on an Apple Silicon GPU (`webgpu-fly`, `FastFly`, Connectome OS precedents). Five instances is the same sparse multiply five times per millisecond. Every duck carries a whole brain.

### 8c. Why the whole brain matters more than its size

The readout problem in §4 "Tier B, checked" mostly dissolves at whole-brain scale, because the inputs and outputs are annotated rather than guessed:

- **Output: descending neurons.** FlyWire annotates roughly 1,300 descending neurons carrying commands from brain to ventral nerve cord. Several are characterized: DNa02 (turning), DNp09 (forward walking), the giant fiber (escape), the moonwalker descending neurons (backing up). Reading those populations is a principled motor output. Steering takes them as forces.
- **Input: real sensory neurons.** Gustatory sugar-sensing neurons (Gr5a, Gr64f) fire on food contact. Olfactory receptor neurons take a food odor gradient. Mechanosensory neurons take a pet or a bump. Looming detectors (LPLC2 and friends) take something approaching fast. *Revised 2026-09-16:* **vision is in.** A per-duck view (raycast fan, rendered camera, or the real microduck camera) is resampled to a 721-column hex lattice and injected into the lamina per optic-lobe column, the way flyvis does. Looming then comes out of real wiring instead of a scalar. Design and risk in `ARCHITECTURE.md` §3.4.
- **Body and drives live outside the brain.** The connectome has no metabolism. Hunger, sleep pressure, and fatigue are a thin physiology layer that does what the real body does: a starved fly's sugar neurons fire harder because dopamine turns up their gain. Hunger scales sensory gain and neuromodulatory input; the brain decides what to do about it. This is where §1a's slow drives land.
- **Personality is parameters.** All five ducks share one wiring diagram. They differ by neuromodulator tone, sensory gain, and synaptic noise set at hatch. §1a's Aggressiveness and Curiosity modulating decay rates maps onto this directly. Goal #1 depends on these parameters producing visibly different ducks; that is the first thing to test.
- **Learning is real.** The mushroom body's plasticity rule is characterized: dopamine-gated depression at Kenyon cell to output neuron synapses. Implement that one rule and a duck that finds bread near the reeds genuinely learns to prefer the reeds, in the circuit that does this in flies. This is the payoff the browser version could never have, and it is what makes feeding matter.
- **Environment drives behavior.** Water versus land, day and night, temperature, weather, and food placement all enter as sensory scalars or gain changes. Goal #2 requires behavior to be readable as a response to these, so every environmental input should have a visible cause in the garden.

### 8d. Architecture

Two processes, mirroring the robot demo (brain → app → body).

- **Brain server, Python.** Every FlyWire download and annotation tool is Python (Codex exports, `fafbseg`, `navis`, `train-your-fly`), so the brain is Python. PyTorch sparse on MPS (Mac) or CUDA, 1 ms timestep, five brain instances, the physiology layer, the plasticity rule, and the shuffled-weights control behind a flag. Receives sensory scalars per duck; sends descending-neuron population activity back at 50 Hz over a local WebSocket. If five brains at 1 kHz do not hold real time in PyTorch, the hot loop moves to Rust; nothing else changes.
- **Body and viewer.** *Revised 2026-09-16:* the body speaks microduck's own JSON-RPC (`robot.move`, `robot.head`, `robot.do`), so a 2D stub, the MuJoCo microduck sim, and physical microducks are interchangeable behind one contract. The viewer (2D debug, Godot garden, or the companion view for a real room) is decision 2 in `ARCHITECTURE.md` §6.2. Godot details from the earlier draft still apply if that route is chosen.
- **Mood indicator.** Driven by the physiology layer plus recent descending activity (escape firing reads as fear, feeding-circuit firing reads as joy), not by raw spikes. §7b's legibility finding applies harder now that a brain, not a rule, decides.
- **Persistence.** Godot saves the physiology state, plasticity weights, and relationships per duck. On launch, elapsed wall-clock time is simulated forward in coarse steps with rendering off, capped at a few simulated days. Always-on background brain (menu-bar process) is a later upgrade, not the first version.
- **Tuning.** The brain server runs headless with time scaled up and logs drive values, descending activity, and plasticity weights per simulated hour as CSV. Seeded PRNG throughout (§7d).

### 8e. Sequencing

1. **One brain, one duck, no garden.** Get FlyWire running in the Python server, feed it sugar and looming, read DNa02/DNp09/giant fiber, and confirm it turns toward food and startles from looming in a bare Godot scene. This is the whole risk of the project in one step. Run the shuffled-weights control here, first.
2. **Five ducks, five parameter sets.** Same bare scene. Confirm the personality parameters produce visibly different ducks over ten minutes. If they do not, goal #1 is in trouble and it is cheaper to know now.
3. **Physiology and environment.** Hunger gain, sleep, water and land, day and night, food placement. Confirm behavior reads as response to environment.
4. **Mushroom-body plasticity.** One rule. Confirm a duck learns a food location.
5. **The garden and the look.** Terrain, water, props, fog, filtering, the mood indicator, the click verbs. Dreamcast-faithful unless the era decision in §7c goes the other way.
6. **Persistence and catch-up.** Then the sticky loop exists.

### 8f. Honesty and risk

The 139k model is a caricature: uniform leaky integrate-and-fire neurons, synapse counts as weights, transmitter signs predicted from imagery, no neuromodulation beyond what the physiology layer injects. Emergent behavior will be crude. It moves, it startles, it seeks sugar, it learns one association. It will not act like a duck, and the five will differ because their parameters differ, not because they have lives. The shuffled-weights control decides whether the label "real fly brain" stays on; the fallback label is "connectome-derived network." Goal #1 is the real bet, and step 2 of the sequence is where it pays off or does not.

---

Sources consulted: Bailey & Katchabaw, *FuturePlay 2008* (csd.uwo.ca); Park et al., *arXiv 2304.03442*; Google Research blog on the MaleCNS connectome; `snedea/flybrain` (GitHub); `cobanov/awesome-fly` curated list (GitHub); r/robots post on running MaleCNS live to drive a quadruped robot (quoted directly, §4); RoboHorizon magazine writeup of the EON FlyWire→NeuroMechFly→MuJoCo pipeline; Roman Liutikov, "PS1 style graphics in Three.js"; David Colson, "Building a PS1 style retro 3D renderer"; Codrops PS1 jitter shader tutorial; three.js forum thread on affine texture mapping.

Added in review (from memory, not re-fetched; verify before citing): Seelig & Jayaraman, "Neural dynamics for landmark orientation and angular path integration," *Nature* 2015 (ring attractor); Steve Grand, *Creation: Life and How to Make It* (2000) and the Creatures Wiki; Dave Mark, *Behavioral Mathematics for Game AI* (2009); Craig Reynolds, "Steering Behaviors for Autonomous Characters," GDC 1999; Sonic Adventure / SA2 platform history (Dreamcast 1998/2001, GameCube ports 2001/2003); three.js r160 release notes (UMD build removal).
