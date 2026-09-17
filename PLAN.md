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
Outcome (2026-09-17): passes; numbers and the changes that got there are in the gate file. Two runs of the full gate were stopped by the harness for "low memory" with about 48% of memory free; long runs now go one garden at a time and run detached.
Check: `gates/gate_05_personality.py` asserts drinking lowers thirst at the shore; a duck with high water love swims more than a water-shy one in the shade, and a heat-intolerant duck swims more than a heat-tolerant one in the sun; and it runs five ducks with five distinct labels for 10 simulated minutes, headless, twice with different jitter seeds; computes per-duck behavioral signature (fraction of time moving, near food, near others, asleep; mean speed; startle count) and asserts pairwise distance between labels exceeds a threshold and same-label distance across seeds is below it. Prints the signature table.
Blind test (manual, recorded in the gate's docstring): Chris watches the 2D view for 5 minutes with labels hidden and guesses. Score noted.
**Decision**: the knob set and label list. Add or cut knobs based on which ones separated the ducks.

### Gate 6: vision v1 (L)
Build: `body/stub2d/retina.py` 1D raycast fan per duck (luminance from world objects, food bright, ducks mid, background dark), resampled to a frontal subset of the 721-column hex lattice (flyvis convention, verified: `extent=15`, 721 ommatidia, MIT license), ON/OFF contrast with a short temporal filter. `brain/encoder.py` injects per column into lamina L1 (ON) and L2 (OFF) using the column-assignment file from Gate 0 (Codex challenge CSV or Matsliah 2024 Supplementary Data 2). If Gate 0 pre-decided the fallback, run flyvis as the visual front end instead and feed its output neurons into the central brain. Remove the LPLC2 scalar hack.
Check: `gates/gate_06_vision.py` presents a looming disc in the retina and asserts the giant fiber fires via the visual pathway with the scalar injection disabled; presents a static scene and asserts it does not. Prints LPLC2 population rate traces.
**Decision**: direct lamina injection worked, or fall back to flyvis as a visual front end feeding its outputs into the central brain. If Gate 0 showed the export has no column assignment, this decision moves up to Gate 0.

### Gate 7: learning (M)
Build: `brain/plasticity.py` dopamine-gated depression at Kenyon cell → MBON synapses; PAM driven by sugar and petting events, PPL1 by startle. Per-duck plasticity weights are the only mutable synapses.
Check: `gates/gate_07_learn.py` pairs an odor with sugar for N trials then measures approach bias toward that odor versus a control odor; asserts bias grows and that the control odor is unchanged; logs a forgetting curve over the following minutes.

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
