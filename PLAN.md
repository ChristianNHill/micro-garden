# Quiet Garden: gated build plan

## Context

Goal (RESEARCH.md goal block): show five microducks driven by real AI with emergent behavior and distinct personalities, responding to environment and bodily needs. Fly brain is the hook; aesthetics and gameplay are last. Chris drives architectural decisions; each gate proves one small piece works correctly; a low-fidelity 2D body is fine until the system is proven.

Locked decisions (ARCHITECTURE.md §0, §6): world simulation lives in Python beside the brain; body contract is microduck's JSON-RPC (`robot.move`, `robot.head`, `robot.do`, `robot.stop`, `robot.relax`, `robot.init`) over NDJSON on a Unix socket, with a separate raw float32 channel for sensory frames; final room is both (Godot garden as a skin over the MuJoCo microduck body now, physical room later), with the rule that no Godot work starts until the MuJoCo gates pass; compute path measured in Gate 1; vision in; personalities are silly Chao-style presets; persistence is catch-up on launch.

Repo today: `RESEARCH.md`, `ARCHITECTURE.md`, no code, not a git repo. First action after approval: `git init`, copy this plan to `PLAN.md`, commit.

## Working rules for every gate

- One gate = one runnable check in `gates/gate_NN_*.py` that exits non-zero on failure. That check is the definition of done. No frameworks beyond `pytest` for the few unit checks that need parametrization.
- Every gate is headless-first. The 2D pygame view is a debug window, never the check.
- Seeded RNG everywhere (`numpy.random.default_rng(seed)`), fixed 1 ms brain step, fixed 20 ms body step. Same seed, same spikes.
- Ponytail rules: stdlib and numpy before dependencies; no abstractions until a second body exists; `# ponytail:` comments on deliberate ceilings.
- A gate ends with a decision note for Chris where marked **Decision**. Nothing in the next gate that depends on that decision starts before it is made.
- Sizes are S (hours), M (a day or two), L (several days).

## Stack

Python 3.12 via `uv`. `numpy`, `torch` (MPS), `pygame` (2D stub only), `pyarrow` or `pandas` for the FlyWire CSV load, `mujoco` only via the microduck sim (their container, not ours). Godot 4 arrives in Phase D. No web stack.

Layout:

```
brain/      connectome load, LIF step, encoder, decoder, plasticity, physiology, personality presets
body/       contract (JSON-RPC NDJSON server/client), stub2d/, mujoco_adapter/, sensory frame codec
world/      odor/temperature fields, food, day-night, contact, room tracker (later)
viewer/     pygame debug view; godot/ later
gates/      gate_00_data.py ... one runnable check per gate
data/       gitignored FlyWire download
```

---

## Phase A: brain on a bench (no body)

### Gate 0: data and named neuron sets (S, plus a Chris action)
Verified 2026-09-16: two data routes. Codex "Download Data" (`connections.csv.gz`, `neurons.csv.gz`, `classification.csv.gz`, `visual_neuron_types.csv.gz`) requires sign-in and accepting terms, which **Chris does by hand**; Claude does not create accounts or accept terms. Zenodo 10.5281/zenodo.10676866 is CC BY 4.0 with no sign-in but is 852 MB connections plus 9.5 GB synapses and no cell types; cell types come from `flyconnectome/flywire_annotations` on GitHub. **Optic-lobe column assignments are not in either export.** They live in the Codex "Visual Columns Mapping Challenge" download (sign-in) or Matsliah et al. 2024 Supplementary Data 2.
Build: `brain/data.py` loads whichever files land in `data/`; builds a sparse CSR matrix signed by predicted transmitter (GABA −1, everything else +1, the snedea/flybrain convention) with weight = synapse count; resolves the named sets from ARCHITECTURE.md §3.6 by classification substring, seeded from flybrain's `build_connectome.py` group map.
Check: `gates/gate_00_data.py` prints neuron and synapse counts and the size of each named set; fails if any set is empty; prints whether a column-assignment file is present and how many columnar neurons it maps (expect about 23k across 796 columns).
**Decision**: Chris reviews named-set counts and license. If the column file is absent or messy, Gate 6 is pre-decided as the flyvis front-end route.
Review (2026-09-17):
- **Named sets:** counts are printed by the gate. The grooming DNs were a guess (DNg12). They are now the DN types that receive the most grooming JO-F and head/eye bristle input (DNg20, DNg84, DNg15, DNge133). These are data-derived; FlyWire has no aBN/aDN labels.
- **Column file:** never downloaded. Gate 6 went the flyvis route (Chris, 2026-09-17), so it is optional. If it lands in `data/` later, the gate reports it.
- **Licenses:** recorded in `ATTRIBUTION.md`. Zenodo connectivity is CC BY 4.0 (attribution required). The flywire_annotations repository has no license file, only a request to cite four papers. That needs a decision before shipping an app: ask the authors, or rebuild types from a licensed source.

### Gate 1: one brain ticks, five brains in real time (M)
Build: `brain/lif.py` with two backends behind one function signature: `torch` sparse on MPS, and an event-driven numpy loop (only spiking neurons propagate). Starting LIF parameters from snedea/flybrain (verified: leak V×0.95 per tick, threshold 1.0, reset 0, refractory 3 ticks, weights normalized to 0.15 × w/max|w|). Note flybrain ticks at **10 Hz (dt = 100 ms)**, not 1 kHz; its behaviors live at that timescale. Timestep is a Gate 1 parameter, not inherited. Batched state for N instances.
Check: `gates/gate_01_tick.py` (a) same seed gives identical spike trains across two runs and across both backends within tolerance; (b) benchmarks 1 and 5 instances at dt = 1 ms and dt = 10 ms for 10 simulated seconds on each backend and prints wall-clock ratio and mean firing rate (sanity: not silent, not saturated).
**Decision**: compute backend and dt. Pick the backend and step that hold 5× real time with margin and a plausible firing regime; delete the other backend.

### Gate 2: sugar in, feeding out (M)
Build: `brain/encoder.py` injects current into named input sets from a dict of scalars; `brain/decoder.py` reads DNa02, DNp09, giant fiber, moonwalker rates with a short window and smoothing, and emits an intent `{vx, vy, vyaw, escape}`. Shuffled-weights control: same in-degree, permuted targets, behind a flag.
Check: `gates/gate_02_sugar.py` steps one brain with sugar GRN input on and off and asserts the proboscis motor neuron rate rises with sugar (changed 2026-09-16: DNp09 is visually driven and sugar never reaches it); steps looming into LPLC2 (scalar for now) and asserts a giant fiber spike within 50 ms; runs the same on the shuffled brain and prints both. Does not assert the shuffled brain fails.
**Decision**: is the real-versus-shuffled difference visible in the numbers? Record the answer; it decides the label later, not the build.

---

## Phase B: 2D body and world in Python

### Gate 3: body contract and 2D stub (M)
Build: `body/contract.py` JSON-RPC 2.0 NDJSON server over a Unix socket implementing the microduck methods with the same parameter names and units (radians, trunk frame). `body/frames.py` sensory frame as a fixed-layout float32 record over a UDP socket, with a monotonic timestamp. `body/stub2d/` kinematic circles on a plane that integrate `robot.move`. `world/fields.py` odor as a 64×64 diffusing grid from food sources, temperature as a static map with shade and sun, food dishes, contact by radius. `viewer/debug2d.py` pygame window drawing world state, heading lines, drive bars.
Check: `gates/gate_03_stub.py` drives 5 stub ducks with a scripted random walker over the contract for 60 simulated seconds and asserts the final positions are identical across two seeded runs; asserts odor gradient at any point points toward the nearest food.

### Gate 4: closed loop, one duck finds food by smell (M)
Build: `brain/server.py` main loop: receive sensory frame → encode (olfaction from odor field, gustation on contact) → step 20 brain ticks → decode → send `robot.move`. Eating resets hunger and removes food.
Known risk from Gate 2 (2026-09-16): at syn gain 0.005, full-strength food odor activates about 11k neurons and 545 DNs including MDN, the same DN set bristle input gives, so odor is not yet a specific signal. Expect to lower odor level or gain, or make glutamate inhibitory, before this gate passes. Forward drive to food comes from DNp09, whose top inputs are visual (LC9, LC31a).
Outcome (2026-09-16): after model work (adaptation, ORN release asymmetry, odor contrast gain, odor-steering DNs) the real brain found food in 20/20 episodes, median 25.2 s; shuffled 0/20. Details in `gates/gate_04_seek.py`.
Check: `gates/gate_04_seek.py` runs 20 seeded episodes each for real and shuffled brains in the 2D stub, headless, and prints median time-to-food for both. Passes if the real brain's median is below a fixed bound (tune the bound once, commit it). Prints the shuffled result beside it.
**Decision**: the fly-brain claim. If real and shuffled are indistinguishable here, the label becomes "connectome-derived network" and the plan continues unchanged.

### Gate 4b: more senses (M) — added 2026-09-16 at Chris's request
Build: the world gains stink patches (a second diffusing smell), a pond (humid air around it, water on contact) and left/right touch; the sensory frame carries each directional sense per antenna. The encoder feeds each side's neurons: danger smell into geosmin/CO₂ ORNs, humidity into moist/dry-air neurons, temperature into heat/cold neurons, touch into bristles, and pond water into the sugar/water taste neurons, which FlyWire does not split. The decoder flips steering to "away" while DNp32 is active; a held-out screen found DNp32 fires for danger smell and nothing else.
Check: `gates/gate_04b_senses.py`, real beside shuffled:
- ducks spend less time near a stink patch
- ducks find the pond
- touching ducks move apart
- shade time is printed only; no descending neuron is heat-specific, so heat comfort waits for Gate 5
Open: aggression. aIPg and pC1d/e stayed silent for every input tried; attacking needs its own investigation, planned before Gate 8.

### Gate 4c: temperament (M) — added 2026-09-16 at Chris's request
Build: aggression as a mood. A per-duck aggressiveness knob scales a tonic input into pC1d/e, which drives aIPg. The trigger is hunger near food (food defense) or having just been headbutted (provoked, fading over about 10 s). While aIPg is active, the touch turn flips toward the other duck, and a touching aggressive duck lunges and headbutts; in the stub a headbutt pushes the victim back. Dishes hold several bites and a bite takes 0.5 s, so hunger falls gradually. A per-duck stink-affinity knob decides whether a duck bolts from stink or lingers in it.
Check: `gates/gate_04c_temperament.py`:
- a hungry aggressive duck attacks a meek one and holds the dish longer
- fed, it attacks far less
- a provoked duck attacks; an aggressive target retaliates and a meek one does not
- stink lovers stay near stink
- in-between knob values give in-between behavior (knobs are scales, not switches)
- the shuffled brain is printed beside the first scenario

Re-run with graded senses (2026-09-18): **passing, all eight.** The dial reads the first blow as a rate rather than counting blows, and `ATTACK_P` dropped from 0.1 a tick to 0.01: ten strikes a second is not a duck, and every setting above aggression 0.2 was landing inside the ~0.7 s the touch rate needs to pass `TOUCH_HZ`. First blow now falls at 20.0, 18.1, 15.2, 10.8, 9.1 s across the knob and the raw count grades too (0.0, 0.1, 0.25, 0.5, 0.6) where it had been pinned at 1.0. `AGGR_FULL_HZ` was also re-anchored 1.0 to 4.4, since graded senses quadrupled aIPg's rate. A full-aggression duck is now a milder bully, 0.6 headbutts in 20 s against 1.0.

### Gate 5: physiology and personality presets (M)
Build: `brain/physiology.py` hunger, thirst, fatigue, sleep pressure, temperature comfort; fast reactions Joy, Fear, Urge to Cry with decay rates; gain scaling and neuromodulator tone into the encoder. `brain/personality.py` the knobs and the label presets from ARCHITECTURE.md §2.4 with jitter. Sleep sends `robot.relax`.
Personality rules (Chris, 2026-09-16):
- Every knob is a continuous 0–1 scale, never a switch.
- Following the Chao games (RESEARCH.md §9), knobs mostly set how fast moods and drives rise and fade, and how strongly they feed the brain, rather than scripting behavior.
- Knobs so far: aggressiveness (Gate 4c; also how fast Anger and Fear fade), stink affinity (Gate 4c), water love (below), plus playfulness, music affinity, hoarding and vanity for Gate 8b.
- Reviewed with Chris 2026-09-16: build all Gate 5 knobs (timidity through carelessness) now; the label set is the Chao Doctor list plus Bully, Zoomer, Napper, Show-off, Scaredy and Loner (21 labels). Knob values are in ARCHITECTURE.md §2.4.

Water, drinking and swimming (Chris, 2026-09-16):
- **Drinking** happens in a thin shore band at the pond edge. Water taste there feeds the sugar/water taste neurons, and when the feeding neurons fire the body sends a `drink` action (stub skill, like `ground_pick`). Each sip lowers thirst; a thirsty duck already follows humid air (Gate 4b).
- **Swimming** is a body mode inside the pond: slower, floaty, no eating, drinking or headbutting, and the water cools the duck. The body reports being wet, fed to the brain as saturated humidity plus touch on every bristle. Flies do not swim, so this encoding is a design choice.
- **Why a duck swims:** temperature comfort (a hot duck cools off) and a per-duck water-love knob. Some ducks paddle, others only drink at the shore.
- **Physical robots cannot enter water:** in the room (Phase E) the pond is virtual, and swimming exists only in the sim and the Godot garden.
Outcome (2026-09-17): drives pass; the labels check now fails at ratio 1.47 against the 1.5 bar. Numbers and the changes that got there are in the gate file, including why the refactor is ruled out as the cause. Chris decided to leave it failing and re-judge after Gates 6 and 8. Two runs of the full gate were stopped by the harness for "low memory" with about 48% of memory free; long runs now go one garden at a time and run detached.
Check: `gates/gate_05_personality.py` asserts drinking lowers thirst at the shore; a duck with high water love swims more than a water-shy one in the shade, and a heat-intolerant duck swims more than a heat-tolerant one in the sun; and it runs five ducks with five distinct labels for 10 simulated minutes, headless, twice with different jitter seeds; computes per-duck behavioral signature (fraction of time moving, near food, near others, asleep; mean speed; startle count) and asserts pairwise distance between labels exceeds a threshold and same-label distance across seeds is below it. Prints the signature table.
Blind test (manual, recorded in the gate's docstring): Chris watches the 2D view for 5 minutes with labels hidden and guesses. Score noted.
**Decision**: the knob set and label list. Add or cut knobs based on which ones separated the ducks.

Re-run with graded senses (2026-09-18): **four of five pass**. The labels check, marginal or failing since it was written, now reads 2.29 (4.28 against 1.87) where it was 1.47 and then 1.52; graded senses sharpened the signatures. Water lovers pass more clearly (0.15 against 0.43 swimming, wanting a 0.15 gap). The shade-or-water knob came out inverted (0.21 for a lover against 0.34 for a water-shy duck). The first explanation, that cold-seeking walks a duck into the pond, was wrong: `temperature_at` depends only on distance from the tree, so cold-seeking does steer to shade. The fault was arithmetic: `swim_urge` was `clip(water_love + hot)` and heat alone pinned it at 1.00 for every duck, so the knob could not act. Heat now multiplies the taste for water, `clip(water_love * (1 + hot))`, leaving the shade case untouched and separating the sunny one 0.20 against 1.00 (Chris, 2026-09-18: the difference should come from whether a duck likes water).

### Gate 6: vision v1 (L)
Build: `body/stub2d/retina.py` 1D raycast fan per duck (luminance from world objects, food bright, ducks mid, background dark), resampled to a frontal subset of the 721-column hex lattice (flyvis convention, verified: `extent=15`, 721 ommatidia, MIT license), ON/OFF contrast with a short temporal filter. `brain/encoder.py` injects per column into lamina L1 (ON) and L2 (OFF) using the column-assignment file from Gate 0 (Codex challenge CSV or Matsliah 2024 Supplementary Data 2). If Gate 0 pre-decided the fallback, run flyvis as the visual front end instead and feed its output neurons into the central brain. Remove the LPLC2 scalar hack.
Check: `gates/gate_06_vision.py` presents a looming disc in the retina and asserts the giant fiber fires via the visual pathway with the scalar injection disabled; presents a static scene and asserts it does not. Prints LPLC2 population rate traces.
**Decision**: direct lamina injection worked, or fall back to flyvis as a visual front end feeding its outputs into the central brain. If Gate 0 showed the export has no column assignment, this decision moves up to Gate 0.
Decided 2026-09-17 (Chris): **flyvis front end.** Probe first:
- FlyWire includes the photoreceptors (R1-6, about 4.4k per eye, laid out on a near-flat retinotopic sheet), so the view was fed straight into them.
- Light drove about 10k neurons but never reached LPLC2 or the giant fiber, whether the disc loomed, stayed still or receded.
- Why: the photoreceptor synapse is histamine (inhibitory; FlyWire's predicted transmitter says acetylcholine), and early vision needs graded, tonically active, temporally filtered neurons that a plain spiking model lacks.
Route: flyvis runs retina to T4/T5; its per-column outputs drive the matching FlyWire neurons, placed by their positions; the real brain does the rest (LPLC2, LC9, and so on).
Built 2026-09-17: `body/stub2d/retina.py` renders the garden onto flyvis's 721-column hex lattice, one eye each side, objects as cylinders so looming falls out of the geometry; the frame carries it as `lum`. `brain/vision.py` steps flyvis in lockstep with the body and drives every FlyWire cell of a type flyvis also models (50 types, about 55k cells), column by column. Two model changes came out of it:
- T4/T5 alone cannot reach LPLC2. They are a quarter of its excitatory input, and at SYN_GAIN 0.01 no tick's worth of them crosses threshold. Driving all 50 shared types fixes that.
- Poisson spikes cannot either. Optic-lobe cells are graded in the fly, which is why the photoreceptor route failed, so `LIF.step` gained a `graded` argument: a driven cell releases in proportion to flyvis's activity instead of spiking, with a tonic release at rest so hyperpolarisation disinhibits.
Two more fixes came from asking why a blank retina was not reading as zero: graded cells release all the time, so an empty grey world pushed the whole brain and fired the giant fiber about 20 times per condition (`LIF.calibrate` now takes the resting release as the zero point); and flyvis's `steady_state` and a grey image fed through its stimulus land about 0.03 apart, so rest is measured down the path the eye actually uses.
Outcome (2026-09-17): **three checks pass, the fourth fails.** A blank retina drives nothing at all (0 giant fiber spikes), a static scene 4, a looming disc 9; LPLC2 reads blank 0.00 Hz, static 0.04, looming 0.07. But the same disc receding reads 0.09 Hz, above looming, in every setting tried, so the check "looming beats the same disc receding" was added and it fails. A looming detector that prefers retreat is a motion detector. Stimulus clipping is ruled out: centring the disc on each eye's axis does not help. Numbers in `gates/gate_06_vision.py`. Not yet wired into `brain/server.py`: the closed loop still uses the Gate 2 LPLC2 scalar until this passes.
Fixed and **passing** (2026-09-17). Three fixes, each found by asking why a number that should have been zero, or symmetric, was not: `LIF.calibrate` takes the resting graded release as the zero point, so a blank retina drives nothing; rest is measured down the path the eye actually uses, since flyvis's `steady_state` and a grey image through its stimulus land about 0.03 apart; and the release was clipped lopsided, letting a cell add 0.9 but withhold only 0.1, which threw away nearly all of the disinhibiting half of graded transmission and made LPLC2 prefer a retreat. Centring the tonic at 0.5 flips it: the looming-to-receding ratio runs 0.78, 1.07, 1.14 as the tonic goes 0.1, 0.3, 0.5, then 1.43 and 1.90 as the gain goes 4 and 8, while raising the gain at tonic 0.1 had done nothing for that ratio.
Final numbers: blank 0.00 Hz and 0 giant fiber spikes, static 0.04, looming 0.06 with 11 giant fiber spikes, receding 0.03. All eight robustness cases pass (disc radius 0.15-0.4 m, closest approach 0.12-0.3 m, eye axis 35-70 degrees, half-speed approach), looming beating a retreat by 1.5 to 2.6 and a static scene by 1.4 to 3.7. Ruled out along the way: clipping, hex sign conventions, noise, direction labels, the column map and the gain.
Checks read LPLC2's 210 cells rather than the giant fiber's two, which cannot support a ratio at 0 to 11 spikes and takes half its drive from LC4, a type flyvis does not model; the giant fiber is reported and only asked to fire at all.
Left standing: the pathway is quiet (LPLC2 peaks near 0.06 Hz, and Tm5f, a quarter of its excitation, never fires), so `SYN_GAIN`, the threshold and the adaptation, set at Gates 1 and 4 against olfaction, may still be wrong for the optic lobe. The Codex Visual Columns file is not needed: randomising the column map halves LPLC2's rate but does not move the selectivity.
Wired into `brain/server.py` (2026-09-17) behind `eyes=`, **on by default**, and the timidity gain in `brain/physiology.py` now has a use: it was dead code, because `sense_levels()` never produced an `LPLC2` key, so the ducks had never had visual input in the closed loop at all. Also fixed: a duck with no frame yet read `lum` as all zeros, pitch black in both eyes, the largest transient the visual system can be given; `frames.blank()` now returns grey.
**The eye adapts** (Chris, 2026-09-17). A lone disc on grey and the demo garden differed 120-fold in raw drive, so a fixed gain could not serve both; the eye now normalises by its own drive, one scale per duck shared across both eyes so left/right stays comparable, over 3 s so adaptation tracks the scene rather than cancelling the loom being watched. `VIS_FLOOR` is the ceiling on its own gain: a featureless field has a drive RMS of 1e-5 against 0.047 for the garden, and without a floor that residue normalises up to full scale and a blank retina fires the giant fiber. Adapted scales are now 0.056 and 0.047, a 1.2-fold spread.
Adaptation did not, by itself, fix the closed loop: Gate 4 went 8/20 without it and 6/20 with it. The cause is elsewhere. Vision and odor write to the same steering neurons, and with eyes open the left/right asymmetry that odor steering reads fell from 0.48 to 0.15 Hz, because vision adds its own asymmetry pointing at whatever is visually salient. Nothing yet tells a duck that food looks like anything, so the eye is a distractor. Escape is not involved: startles were 0 with eyes both open and shut.
Outcome: `VIS_GAIN` 0.05, four times weaker than Gate 6 alone would want. Gate 4 is back to 20/20 (median 29.6 s against 23.5 s blind, bound 40 s, shuffled 0/20) and Gate 6 passes seven of eight robustness cases, looming beating a retreat by 8 to 11 times. The exception is a static disc with the eyes 35 degrees forward, which drives LPLC2 as hard as a looming one; the committed geometry is 55 degrees. This is a compromise between two gates, not a principled value, and it is narrow: 0.025 is too weak for Gate 6 to see anything. Vision costs about 12 ms a step, taking the five-duck loop from 13.8 to 26.0 ms against a 20 ms budget, so the live view runs at 0.77x real time.
Re-run with eyes open, 2026-09-17: **Gates 4, 4b and 4c all pass.** Gate 4 20/20 (29.6 s), Gate 4b all three senses (danger 0.12 against 0.20 with no patch, pond 20/20, touch 0.08 against 0.40), Gate 4c all eight checks with both dials still grading smoothly (aggressiveness 0.0, 0.1, 0.75, 1.35, 1.55 headbutts; stink affinity 0.12 to 0.80).
**Gate 5 is now the other way round.** Its labels check, the one left failing at 1.47, **passes at 2.01** (same label 2.10, different labels 4.23) on the same 4 gardens, so that is more signal and not more averaging: timidity, sociability and curiosity finally have something to react to. But both swimming checks now fail on margin, not direction: water love 0.1 against 0.9 swims 0.55 against 0.68 (needs +0.15), heat tolerance 0.05 against 0.95 swims 0.35 against 0.30 (needs +0.10), where blind the water-love pair read 0.19 against 0.37. Eyes roughly triple how much every duck swims, so the measure saturates and squeezes the knobs together.
The likely reason is that **the pond is not drawn into the retina at all** (`body/stub2d/retina.py` renders dishes, ducks and the tree, on the grounds that water lies flat). A duck cannot see water, so it wanders in rather than choosing it, and a knob about wanting water cannot show through.
**Next decision for Chris**: draw the pond into the retina (a real pond is a bright reflective surface, so this is defensible and would let water love act through sight) and re-run Gate 5; or accept the swim checks failing while the labels check passes; or lower the swim bars, which is the least honest of the three. Separately, Gate 7 remains the durable fix for the visual gain, which is still a compromise between two gates.

### Gate 7: learning (M)
Build: `brain/plasticity.py` dopamine-gated depression at Kenyon cell → MBON synapses; PAM driven by sugar and petting events, PPL1 by startle. Per-duck plasticity weights are the only mutable synapses.
Check: `gates/gate_07_learn.py` pairs an odor with sugar for N trials then measures approach bias toward that odor versus a control odor; asserts bias grows and that the control odor is unchanged; logs a forgetting curve over the following minutes.
Built and **passing** 2026-09-18. Odor A's own synapses end at 0.526 of baseline against odor B's 0.660, on a brain that saw the same odors as a yoked control and differed only in getting dopamine; the control stays at 1.000 and everything recovers to 0.983 over five minutes.
Two changes were needed first, and both are about the *code* rather than the rule:
- **Sparse.** 28% of Kenyon cells answered any odor and two odors shared 79% of their cells. `plasticity.sparsen` raises their threshold until about 5% answer, the figure measured in the fly (Turner et al. 2008).
- **Repeatable.** Even sparsened, the same odor twice lit different cells (overlap 0.59) about as often as two different odors did (0.44), because `encode` fires a random subset of an odor's receptors each tick. `encoder.graded` releases steadily at the identical mean current, and the same odor then gives the same cells every time (1.00) while two odors still differ (0.48). This is the same move that fixed vision at Gate 6.
The connectome was never the limit: it drives 826 Kenyon cells from odor A alone and 532 from B alone, correlation 0.21.
Two departures from the plan's wording, both recorded in the gate: the readout is the synapses rather than a walk toward a smell (the 2D world carries one food odor field, so two odors cannot be put in two places until Gate 8), and the control is a yoked second brain rather than the same brain before and after, because spike-frequency adaptation drags every rate down over a long run and dragged the unpaired odor 42% on the first attempt.
**Open**: the MBON rates do not yet show the learning (odor A 0.47 against its control's 0.52 Hz), so it is not visible in the signal that would steer a duck. `sparsen` and `encoder.graded` are used only by Gate 7; wiring them into `brain/server.py` would change every other gate and needs the suite re-run.

### Gate 8: social and legibility (M)
Build: other ducks emit odor into the field; contact and looming between ducks; petting verb (touch plus dopamine pulse); mood indicator derived from physiology plus recent descending activity; one-line event toasts in the debug view; selection readout.
Check: `gates/gate_08_social.py` runs the acceptance scenarios from ARCHITECTURE.md §2.8 headless: a Bully and a Scaredy at one dish (asserts displacement); two ducks startled together repeatedly (asserts increased mean separation afterward); a hand that feeds one duck (asserts that duck follows the hand). Prints per-scenario metrics.

### Gate 8b: toys, music, shiny rocks, hats (L) — added 2026-09-16 at Chris's request
Whether a duck likes any of these depends on its bodily needs (Gate 5), its personality (Gate 5) and what it has learned (Gate 7). Nothing is scripted as "likes X".
Build:
- **Ball**: a pushable object in the world. Seen through the retina and felt as touch. Play is gated by energy and boredom and by a playfulness knob; it maps to microduck's kick skills and `BallPlay`.
- **Music**: a sound source in the world, heard through the Johnston's organ neurons, level falling off with distance. Dancing maps to microduck's `Dance`. The music-affinity knob and dopamine pairing decide whether a duck approaches or leaves.
- **Shiny rocks**: bright objects in the retina. A duck picks one up with `ground_pick` and drops it at its nest; hoarding and curiosity knobs.
- **Hats**: a player verb. Head touch goes into the bristle neurons, and the grooming DNs decide whether the duck keeps the hat or preens it off. Comfort, mood and a vanity knob.
Check: `gates/gate_08b_toys.py`, headless:
- a rested, bored, playful duck interacts with the ball more than a hungry or tired one
- after music is paired with food, a duck approaches it, and an unpaired duck does not
- a hoarder's nest ends with more rocks than a non-hoarder's
- a hat stays on a duck with a high vanity knob longer than on one with a low knob
- the shuffled brain is printed beside each
Earlier sensory hookups (music as Johnston's organ input, danger odor, bitter taste, touch) can land before Gate 5 as small steps if Chris asks.

Built and **passing** 2026-09-18, music and hats only. A duck that likes music ends up 1.79 m from the speaker against 2.28 m for one that does not (shuffled 1.90 m), and a hat stays on 100% of the time at vanity 0.95 against 28% at 0.05, grading in between. Music is heard through the 1,103 Johnston's organ neurons with the sound falling off across the garden; a hat is felt on all 1,417 bristles and the grooming neurons answer the itch. Both knobs were defined at Gate 5 and had never been used.
**The ball and the shiny rocks wait on vision.** Both are things a duck has to see, and vision runs at a fifth of the gain Gate 6 alone would want so that it does not drown the nose, which is the same reason Gate 8's hand is printed rather than asserted. Music and hats ride hearing and touch, which carry.
Music steering is explicit in the decoder rather than emergent: a duck turns toward or away from the louder ear according to its affinity. The plan puts the decision with the knob, but no sound-responsive descending neuron was found by screening the way `odor_steer` and `danger_valence` were, and that screen is the more faithful route.

### Gate 9: persistence (S)
Build: `brain/save.py` serializes physiology, plasticity weights, personality, world objects (including hats worn, rocks at nests), seed; on load, simulates elapsed wall-clock forward in coarse steps, capped at three simulated days.
Check: `gates/gate_09_persist.py` saves, advances 10 simulated minutes, loads from the save with a faked 10-minute gap, and asserts drive values match within tolerance; asserts a three-day gap loads in under 10 seconds.

**Phase B exit review with Chris**: the system is proven in 2D or it is not. Nothing in Phase C starts until this review.

---

## Phase C: MuJoCo microduck body

### Gate 10: one simulated microduck walks to a dish (M, setup risk)
Verified 2026-09-16: the multi-duck command is `scripts/duck-sim boot N` (not `up N` as ARCHITECTURE.md §1 says; fix that). It runs one container per duck and needs `sudo`, `systemd-nspawn`, and `mmdebstrap`, which are Linux tools; the docs say Apple Silicon gets a native arm64 container, but the exact Mac path is the first thing to confirm, and a Linux VM is the fallback. Needs the `microduck_rl` venv and `libonnxruntime`. Per-duck `robotd` sockets at `~/.cache/duck-sim/duck-a.sock` (and `duck-a-tof.sock`); MuJoCo body on TCP `7801+n`; ducks do not hot-join, so boot with 5 from the start.
Build: `body/mujoco_adapter/` connects the brain server's contract client to each duck's Unix socket and sends `robot.move` and `robot.head`; reads duck poses from the MuJoCo TCP port to feed `world/fields.py` for odor and contact (Python still owns the environment layer, per decision 1).
Check: `gates/gate_10_sim_one.py` drives one sim duck with the real brain toward a food position and asserts it arrives within a time bound; asserts pose feedback and odor synthesis agree with the 2D stub's semantics for the same scenario.

### Gate 11: sim camera into the retina, possession (M)
Build: per-duck camera from the sim (verified: opt-in via `--cameras` / `DUCK_SIM_CAMERAS`; frames on TCP `7901+n`; the docs call N cameras at 30 fps "the scaling wall," so render small and at 15 fps, which is all a 721-column retina needs) → the Gate 6 resampler. Possession view in the debug window: ride along (show POV, hex retina, DN bars, intent arrow) and take the wheel (mute motor output, WASD to `robot.move`).
Check: `gates/gate_11_sim_vision.py` places an object approaching a sim duck and asserts the giant fiber fires from camera input; asserts take-the-wheel mutes the brain's intents while sensory frames keep flowing.

### Gate 12: five sim ducks, skills, full scenarios (L)
Build: `duck-sim up 5`; skill mapping in the decoder: escape → `Startle` skill plus backward `vx`, boredom with high energy → `Zoomies`, feeding → `ground_pick`, grooming circuits → `Preen`/`Ruffle`; each label's signature skill. Re-run Gates 4, 5, 7, 8 as MuJoCo variants.
Check: `gates/gate_12_sim_five.py` re-runs the Phase B checks against the MuJoCo body and prints the same metric tables side by side with the 2D results.
**Decision**: Godot garden go, or physical room only (Godot dropped, companion view is the 2D debug view grown up). Also: buy robots yes/no/when.

---

## Phase D: Godot garden (only after Gate 12 and a "go")

### Gate 13: Godot viewer over the world snapshot (L)
Build: Godot 4 project under `viewer/godot/` that subscribes to the world snapshot and state frames; ground plane with heightmap lookup, pond, five microduck-proportioned low-poly ducks with procedural waddle and paddle and squash-stretch, separate mood-indicator mesh, orbit camera with pitch limits and idle drift, click-to-feed on ground and click-to-pet on duck sent back as player actions.
Check: `gates/gate_13_godot.py` runs the brain and MuJoCo body headless, launches Godot, and asserts via its log that it received snapshots at ≥ 30 Hz for 60 s and that a scripted click produced a food object in the world. Manual: silhouettes match labels (ARCHITECTURE.md §2.4).

### Gate 14: the look (M)
Direction (Chris, 2026-09-16): a blend of Dreamcast low-poly forms with a halftone print-diorama finish (ARCHITECTURE.md §4, RESEARCH.md §12).
Spike first, before any other Godot work: one duck on a patch of garden through the print shader, judged by Chris for shimmer, silhouette readability and palette.
Build:
- Dreamcast pass: small textures, flat Lambert, vertex colors, fog to the paper color, moving sun, under 300 KB assets.
- Print pass: limited ink palette, halftone/dither locked so it does not crawl, outline pass, cream paper surround, isometric-ish drifting camera.
- Possession view: full-frame with fisheye and overlays.
- Reduced-motion switch.
- Ambient pond loop and event quacks.
Check: manual review against the visual brief (ARCHITECTURE.md §4) plus an asset-size assertion in `gates/gate_14_look.py`.

---

## Phase E: the room (when robots exist)

### Gate 15: one real microduck over LAN (M)
Verified 2026-09-16: scripted LAN teleop is **not implemented yet** upstream. The WebRTC teleop data channel and the "WebSocket surface for server-side programs" are both designed and deferred. The only working route today is an SSH tunnel or `socat` to `/run/robotd.sock` on the robot, speaking JSON-RPC NDJSON. Likewise no non-browser camera read exists yet; on-robot `v4l2-ctl --stream-mmap` works, so the first version pipes frames over SSH from a small on-robot grabber. Camera is a Pi Camera v2 (IMX219, roughly 62° horizontal), 720p at 30 fps tested. ToF is a VL53L5CX/8CX 8×8-zone sensor with 45° field and 4 m range, and "most ducks have no sensor fitted," so petting needs a camera or contact proxy rather than ToF.
Build: `body/real_adapter/` with the SSH tunnel to `robotd.sock`; on-robot frame grabber; laptop-side decode into the retina; petting as a visual hand-approach event until ToF hardware is confirmed. Revisit when upstream ships the WebSocket surface.
Check: brain drives one robot to a marked dish on the floor; possession shows its real camera with overlays.

### Gate 16: overhead tracker (M)
Build: one webcam, ArUco tags on duck heads and dishes; `world/tracker.py` publishes positions at 30 Hz; odor synthesized from them; god-view drawn in the companion viewer.
Check: tracker positions vs. tape-measured positions within tolerance; odor gradient points to the real dish.

### Gate 17: five ducks in the room (L)
Re-run Gate 8 scenarios on the floor. Record the blind personality test with a visitor.

---

## Architectural decision points (Chris drives)

| After gate | Decision |
|---|---|
| 0 | Named sets and license acceptable; column assignment present or Gate 6 fallback pre-decided |
| 1 | Compute backend |
| 2, 4 | Fly-brain label: keep or downgrade |
| 5 | Knob set and label list (Chris reviews before the check is written) |
| 6 | Direct lamina injection or flyvis front end |
| 8b | Which toys stay, and their knobs |
| 9 | Phase B exit: system proven |
| 12 | Godot go/no-go; robots purchase |

## Verification, end to end

Every gate's check is the verification. At Phase B exit, `uv run python -m gates` runs Gates 0 through 9 in sequence headless and prints one table: gate, pass/fail, key metric, wall time. The same command with `--body mujoco` runs the Phase C variants. Manual checks (blind personality test, visual brief review) are recorded as dated notes in the gate file docstrings, not asserted.

## Corrections to ARCHITECTURE.md (apply on approval)

- §1: `duck-sim up 4` → `scripts/duck-sim boot N`; sim runs one container per duck, per-duck sockets, cameras opt-in.
- §3.4: column assignments are not in the standard FlyWire export; source is the Codex Visual Columns challenge CSV or Matsliah 2024 Supplementary Data 2. flyvis confirmed 721 columns, MIT.
- §3.5: LAN teleop today is SSH tunnel to `/run/robotd.sock` only; WebRTC teleop and WebSocket surface are deferred upstream. Real camera needs an on-robot grabber. Most ducks ship without ToF.
- §7: mark items verified; add "Codex sign-in and terms are a Chris action."

## First steps after approval

1. `git init`, `.gitignore` with `data/`, copy this file to `PLAN.md`, apply the ARCHITECTURE.md corrections above, commit all three docs.
2. `uv init`, add `numpy torch pandas pygame`.
3. Chris: sign in to Codex, accept terms, download the four v783 CSVs and the Visual Columns challenge CSV into `data/`. Meanwhile Claude builds Gate 0's loader against the GitHub `flywire_annotations` TSV so it runs the moment the files land.
4. Gate 0.
