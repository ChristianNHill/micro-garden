# Quiet Garden Architecture

Status: the design of record. Written 2026-09-16, ledger brought up to date 2026-09-19; PLAN.md has what each gate found. Companion to `RESEARCH.md` (goal block at its top and §8 are the settled direction; this file is the design). The three decisions in §6 are made (ledger below); the tradeoff tables are kept as the reasons.

---

## 0. Decisions ledger

| # | Decision | Status | Where |
|---|---|---|---|
| 1 | Where the world simulation lives | **Settled 2026-09-16: Python, beside the brain** (option A) | §6.1, `world/fields.py` |
| 2 | Renderer and final "room" (virtual garden, physical room, or both) | **Settled 2026-09-21: the virtual garden only.** It was "both" (2026-09-16); Chris dropped the physical room, so there are no real robots and the simulated microduck is the final body. The Godot garden draws either body | §4, §6.2 |
| 3 | Brain compute path | **Settled at Gate 1: PyTorch sparse on MPS**, 10 ms tick. Five ducks with eyes run at about 0.77x real time, which Phase C has to fix | §3.7, `brain/lif.py` |
| 4 | Personalities | **Settled: Chao-style labels over continuous knob presets; Chao Doctor list plus six of ours (2026-09-16)** | §2.4 |
| 5 | Vision | **Settled: yes, through a flyvis front end** (2026-09-17): direct photoreceptor injection never reached LPLC2. Possession is Gate 11 | §2.6, §3.4, PLAN.md Gate 6 |
| 6 | Transport between brain and body | **Settled 2026-09-16: microduck JSON-RPC over NDJSON for intents, raw float32 over UDP for sensory frames** (option A) | §6.3, `body/contract.py`, `body/frames.py` |
| 7 | Persistence | **Settled: catch-up on launch first, always-on brain later** | §3.8 |
| 8 | Body contract | **Settled: microduck's own JSON-RPC**, built at Gate 3 | §1, §3.2 |
| 9 | Look | **Settled 2026-09-21: a Chao garden.** It began (2026-09-16) as a Dreamcast low-poly garden printed as a five-ink halftone diorama. Seen running, Chris wanted the Sonic Adventure 2 garden's forms and colours, so the halftone stayed as a light screen of shading and the five inks and the paper went | §4 |
| 10 | What the ducks are to each other | **Settled 2026-09-21: explicit and stored**, beside the body's moods (`brain/social.py`). It was "not stored anywhere, emerging from plasticity"; that gives a duck a general liking for ducks and nothing about one duck in particular | §2.5 |

---

## 1. Microducks are real, and that changes the body

[pollen-robotics/microduck](https://github.com/pollen-robotics/microduck) is an open-source (Apache-2.0), commercially available biped: about 25 cm tall, 800 g, fifteen servos, a Rockchip RK3566, a camera streaming over WebRTC, a time-of-flight depth sensor, a speaker that quacks, and a 50 Hz control loop running PPO policies trained in MuJoCo and exported to ONNX. Everything is Rust daemons talking one JSON-RPC 2.0 contract over a Unix socket.

What matters for us, verbatim from their design docs:

- **Motion is a velocity intent.** `robot.move {vx, vy, vyaw}` in the trunk frame, radians, sent as a stream of notifications at up to 50 Hz. `robot.head {neck_pitch, head_pitch, head_yaw, head_roll}` for gaze. `robot.stop`, `robot.enable`, `robot.init` (stand up), `robot.relax` (drop torque) as answered requests.
- **The policy takes a 61-float observation**: gyro, projected gravity, joint positions, joint velocities, last action, and a 13-float command block (velocity, head, body pose). It outputs 14 joint actions. The mouth servo is a separate slot.
- **Skills are one call.** `robot.do <skill>` runs a named ONNX policy for a duration: `roulade`, `polite-bow`, `sit_toggle`, `ground_pick`, plus anything fetched from the Hugging Face hub and given a name.
- **Simulation is the same contract.** "One MuJoCo process, one window, N duck bodies in one scene, so ducks share physics and can bump into each other." `scripts/duck-sim boot N` runs one container per duck with per-duck `robotd` sockets; ducks do not hot-join, and cameras are opt-in (`--cameras`). The docs say Apple Silicon gets a native arm64 container; the exact Mac path is confirmed at Gate 10. "The same binaries, the same units."
- **Sensors we can read**: camera (WebRTC), ToF hand distance, IMU, servo state, and BLE presence of nearby ducks with RSSI for coarse distance. Their own roadmap proposes a YOLOv8n detector on the RK3566's NPU for bearing to objects.
- **They already have a behavior machine** with sixteen states (Chill, LookAround, Wander, TurnInPlace, Zoomies, Startle, Stretch, Ruffle, Preen, Sneeze, Dance, GroundPick, Nap, BallPlay, Petted, Held) driven by an energy/mood model. Their stated philosophy: "Presence, mood, and the shared beat are inputs to one brain, not modes beside it."
- **Teleop is refused over Bluetooth by design.** `robot.move` and `robot.head` only travel over the LAN/WebRTC path or on the robot itself. A laptop brain drives a physical duck over Wi-Fi, exactly like the r/robots MaleCNS demo in `RESEARCH.md` §4.

**Consequence.** The fly brain replaces their sixteen-state machine as the decision source, and their body (real or MuJoCo) becomes one of three interchangeable bodies behind a single contract. The "each chao gets a real brain" idea and the "fly brain drives a robot around my apartment" post are now the same project. Their skills become the ducks' expressive vocabulary for free: a fly brain that fires its escape circuit triggers `Startle`; a bored, high-energy duck triggers `Zoomies`.

---

## 2. Emergent gameplay architecture

The player never controls a duck (except by possession, §2.6, which is explicit and temporary). The player changes the world and the ducks respond through their brains. Five layers, each talking only to its neighbors.

### 2.1 Brain

One FlyWire connectome per duck. Input is current injected into named sensory neuron sets. Output is firing rate of named descending neuron populations plus a few internal circuits, decoded into:

- a continuous **velocity intent** (forward, sidestep, yaw), which is `robot.move`;
- a **gaze intent**, which is `robot.head`;
- occasional **discrete skills**, which are `robot.do`.

The brain never sees a position or a duck. It sees light, smells, tastes, touches, and its own body's state.

### 2.2 Body (physiology)

Hunger, fatigue, sleep pressure, temperature comfort, and the Chao-derived fast reactions (§2.4). Each drive rises on its own clock and is satisfied by a world event. The body acts on the brain two ways: it **scales sensory gain** (a hungry fly's sugar neurons fire harder; that is real dopaminergic modulation) and it **injects neuromodulator tone**. The body also constrains motion: tired ducks walk slower, sleeping ducks send `robot.relax`.

### 2.3 World

Food sources, water, shade, sun, day and night, placed objects, and other ducks. The world turns geometry into sensory signals: an odor field that diffuses from food, a temperature field, contact events, and a camera view per duck. Which process computes this depends on decision 1 (§6.1). In the physical room, reality computes most of it and an overhead tracker fills in what reality won't (§3.5).

### 2.4 Personality (sillier, Chao-style)

Verified in `RESEARCH.md` §1a: the original Chao model has three continuous traits, **Kindness, Aggressiveness, Curiosity**, fixed at birth, where Aggressiveness sets how fast Anger and Fear decay and Curiosity sets how fast Sorrow decays; slow drives (Hunger, Sleepiness, Tiredness, Boredom, Energy) sit under fast visible reactions (Joy, Urge to Cry, Fear, Dizziness). Later games surface discrete labels over that (Big Eater, Crybaby, Energetic, Carefree, Careful, Naive, Gentle, and others; the full list on chao-island.com could not be fetched today and should be verified).

Our version: **a label is a preset of knob values plus one signature skill plus one silhouette tweak.** Five ducks draw five distinct labels at hatch. Knobs are jittered (about ±0.1), so two ducks with the same label still differ.

Settled with Chris, 2026-09-16:
- Every knob is a continuous 0–1 scale, never a switch; Gate 4c checks that in-between settings give in-between behavior.
- Following the Chao games (`RESEARCH.md` §9), knobs mostly set how fast moods and drives rise and fade, and how strongly senses and moods reach the brain. The brain decides what to do.
- The label set mixes the Chao Doctor list with six of our own.

**Knobs**

| Knob | What it changes | Low ↔ high | Built in |
|---|---|---|---|
| Aggressiveness | pC1d/e mood from hunger at food or being provoked; Anger fades slower, Fear faster | meek ↔ defends and retaliates | Gate 4c |
| Stink affinity | bolts from stink ↔ lingers in it | squeamish ↔ loves stink | Gate 4c |
| Timidity | looming and escape sensitivity; Fear fades slower | bold ↔ skittish | Gate 5 |
| Curiosity | approaches new things, wanders wider; Sorrow fades faster | incurious ↔ nosy | Gate 5 (full at 6, 8b) |
| Sociability | other ducks' smell and touch attract or repel | loner ↔ clingy | Gate 5 |
| Kindness | tolerates others at food, preens near them, pushes the ball gently | pushy ↔ gentle | Gate 5, 8b |
| Appetite | hunger rise rate, food-sense gain | picky ↔ big eater | Gate 5 |
| Energy | fatigue rate, walking speed, zoomies | quiet ↔ energetic | Gate 5 |
| Sleepiness | sleep-pressure rate | night owl ↔ napper | Gate 5 |
| Heat tolerance | comfort band | shade seeker ↔ sunbather | Gate 5 |
| Water love | swims vs only drinks at the shore | water-shy ↔ paddler | Gate 5 |
| Boredom rate | how fast boredom builds without novelty | patient ↔ easily bored | Gate 5 |
| Chattiness | quack rate | silent ↔ chatty | Gate 5 |
| Carelessness | reacts less to danger and looming | careful ↔ careless | Gate 5 |
| Smarts | mushroom-body learning rate | slow ↔ quick learner | Gate 7 |
| Playfulness, music affinity, hoarding, vanity | ball, music, shiny rocks, hats | | Gate 8b |

**Labels** (unlisted knobs sit at 0.5; signature skills and silhouettes to be set in Gate 5)

| Label | Source | Distinctive knobs |
|---|---|---|
| Gentle | Chao | aggressiveness 0.05, kindness 0.9, sociability 0.7 |
| Naughty | Chao | aggressiveness 0.6, kindness 0.2, curiosity 0.7, playfulness 0.8 |
| Energetic | Chao | energy 0.9, sleepiness 0.2, playfulness 0.7 |
| Quiet | Chao | energy 0.3, chattiness 0.1, sociability 0.3 |
| Big eater | Chao | appetite 0.95, aggressiveness 0.5 |
| Chatty | Chao | chattiness 0.95, sociability 0.8 |
| Easily bored | Chao | boredom rate 0.9, curiosity 0.6 |
| Curious | Chao (was our Nosy) | curiosity 0.95, timidity 0.2 |
| Carefree | Chao | timidity 0.1, water love 0.8, stink affinity 0.5 |
| Careless | Chao | carelessness 0.9, timidity 0.1 |
| Smart | Chao | smarts 0.95 |
| Cry baby | Chao | timidity 0.9, aggressiveness 0.05, curiosity 0.2 (so Sorrow fades slowly) |
| Lonely | Chao | sociability 0.95; Sorrow builds when alone |
| Naive | Chao | smarts 0.2, timidity 0.1, curiosity 0.7 |
| No personality | Chao | everything 0.5 |
| Bully | ours | aggressiveness 0.95, kindness 0.1, appetite 0.9, sociability 0.8 (seeks company, so food fights come up) |
| Zoomer | ours | energy 0.95, boredom rate 0.8, playfulness 0.8 |
| Napper | ours | sleepiness 0.95, energy 0.3, heat tolerance 0.3 |
| Show-off | ours | vanity 0.9, playfulness 0.8, sociability 0.8 |
| Scaredy | ours | timidity 0.95, sociability 0.4, stink affinity 0.0 |
| Loner | ours | sociability 0.05, curiosity 0.6, timidity 0.3 |

Goal #1 (`RESEARCH.md`) depends on these presets producing visibly different ducks. Gate 5's check and Chris's blind test decide which knobs and labels survive.

### 2.5 Social

Other ducks are world objects that smell, touch, loom and make sounds. Two layers sit on that.

**What the fly gives.** Mushroom-body plasticity: a duck fed or petted while it smells other ducks comes to like that smell, and one startled among them comes to avoid it. `Plasticity.fondness()` reads that learned valence and the decoder acts on it. It is a liking for ducks in general, since the brain is given one duck smell.

**What is explicit** (`brain/social.py`, decided 2026-09-21 after the Chao gardens). A Chao garden is mostly who likes whom, who has been treated well, and who learned what by watching, and none of that is in a fly. So it is kept where the body's moods are kept, as slow quantities that events push about, and it reaches the legs through explicit turns in the decoder that give way to a pressing need like every other like:

- `bond[i, j]`, what duck i thinks of duck j, from a grudge to a friend, and not mutual. A shove makes a grudge. Time at ease beside a duck, and more for sleeping beside it, makes a friend. A song a duck enjoyed makes a friend of the singer. A hat taken under a vain duck's beak is held against the taker. Bonds fade over about three garden days.
- Witnesses, comfort and sharing. A kind duck that sees a shove holds it against the one who did it. A miserable or starving duck cries, a kind one goes to it, and being reached cheers it. A kind duck that is not starving leaves the food to one crying for it.
- Catching moods: an alarm nearby frightens, a whoop nearby cheers the sociable.
- A performance has an audience. A song, a dance or a tap on the drum is entertainment to a sociable duck, an imposition on a solitary one, and a provocation to an aggressive one.
- Skills grow with use (swimming, running), and dancing is learned by watching as well as by doing.

The body says what happened and to whom. The sensory frame carries the other duck's number for each event and which side things are on. It also carries each other duck's own smell at each antenna, so a friend or a grudge steers a duck from across the garden. What a duck makes of any of it is its knobs', which is where personality comes in.

### 2.6 Player

What the player can do, all through the garden's control calls (`garden.*`), so every viewer has the same verbs:

1. **Feed.** Shake fruit down from the tree, which is where food comes from, or hand a fruit to one duck (`garden.give`). Fruit is an orange, an apple or a banana, and each duck has a favourite.
2. **Pet.** A touch on the head paired with a dopamine reward. Petting and fruit from the hand earn a duck's trust in the hand, a clap loses a little, and a duck comes to a hand it trusts.
3. **Clap.** Startles every duck.
4. **The hand.** A mode in which the cursor is a glove that picks up, carries, puts down and throws: fruit, hats, balls, the drum, the music box, and ducks. A duck that trusts the hand likes being carried; a thrown duck lands on its side and trusts the hand less. This is the Chao garden's hand.
5. **Put things in the garden.** A duck decides for itself whether to wear a hat, by its vanity. A playful, bored duck pushes and kicks a ball and taps a drum. The music box plays the player's own music, and ducks that like music dance to it.
6. **Possess.** Select a duck and see through its eyes. Two modes:
   - *Ride along*: the brain keeps driving; the view carries an overlay of what the optic lobe is receiving (the hex retina) and which descending neurons are firing.
   - *Take the wheel*: motor output is muted and the player drives with W, A, S and D. Sensory input keeps flowing, so the brain keeps seeing and learning. Getting off gives the duck its legs back.

### 2.7 Legibility

Part of the architecture, not the visuals. Without it the fly brain is indistinguishable from noise.

- A **mood shape** floats over a duck, its shape carrying the mood so it reads without colour.
- **Emotes** (`brain/emotes.py`): every few seconds a duck may act out its strongest feeling, with its head, its voice and a word over it, the more readily the chattier it is. Strong characters have a **signature** of their own, scaled by the knob and not switched by the label: an aggressive duck stomps, a sleepy one yawns, a water lover splashes, a chatty one sings, a timid one cowers. One definition of each act serves every body: head poses on the robot's four head joints.
- Selecting a duck shows **who it is** (label, mood, favourite fruit, what it makes of the hand, its friend and its grudge, its skills), **bars** for its needs, moods and wants, and **its brain**: 12,000 of its 139,000 neurons drawn where FlyWire has them, each flaring as it fires.
- Events that matter produce a one-line **toast**.

### 2.8 Acceptance scenarios

Each gate proves one of these in the 2D or MuJoCo body before any visuals exist:

- Feeding cluster forms; a Bully displaces a Scaredy.
- A hand-fed duck follows the hand.
- Two ducks avoid each other after repeated startles.
- Ducks nap in shade at night and forage by day.
- A Napper rests while others forage.
- Shuffled-weights control: a viewer cannot tell it from the real brain, or can. Either answer is recorded.

---

## 3. System architecture

### 3.1 Processes

```
┌──────────────────────────────┐   sensory frames (per duck)   ┌──────────────────────────┐
│  Brain server (Python)       │◄──────────────────────────────│  Body adapter            │
│  5 × FlyWire LIF instances   │                               │  one per body type:      │
│  physiology per duck         │──────────────────────────────►│   2D stub  │ MuJoCo │ real│
│  sensory encoder / decoder   │   intents: move, head, do     │  speaks microduck JSON-RPC│
│  mushroom-body plasticity    │                               └────────────┬─────────────┘
│  headless mode + CSV logging │                                            │ world snapshot
└──────────────┬───────────────┘                               ┌────────────▼─────────────┐
               │ state frames (mood, drives, DN rates)         │  Viewer                  │
               └──────────────────────────────────────────────►│  2D debug │ Godot garden │
                                                               │  possession view          │
                                                               └──────────────────────────┘
```

### 3.2 Body contract (proposed, decision 8)

Adopt microduck's JSON-RPC methods as the seam so every body is the same to the brain:

- Brain → body: `robot.move`, `robot.head`, `robot.do`, `robot.stop`, `robot.relax`, `robot.init`.
- Body → brain: a **sensory frame** per duck at 50 Hz: retina samples (§3.4), odor vector, sugar contact, touch, ToF distance, IMU, body temperature proxy, and a list of nearby-duck ids with distance.

Three bodies implement it:

| Body | Physics | Sensors | Purpose |
|---|---|---|---|
| **2D stub** (Python, pygame) | none; kinematic circles on a plane | 1D raycast retina, synthetic odor field, contact by radius | Gates 1 to 4. Fast, headless-capable, deterministic |
| **MuJoCo microduck sim** (`scripts/duck-sim boot 5`) | real, shared, ducks collide | rendered camera per duck, synthetic odor from known positions, contact from physics | Gates 5 to 7. Same commands as the robots |
| **Physical microducks** | reality | camera, ToF, IMU, BLE RSSI; odor synthesized by the room tracker (§3.5) | The room |

### 3.3 Brain server

- Loads FlyWire v783 once as a sparse matrix (about 139k neurons, 2.7M synapses, 12 MB compressed). Weights shared across ducks. Per-duck state is membrane potential, refractory timers, and the mushroom-body plasticity weights, the only synapses that change.
- 1 ms timestep, all five instances in one batched step.
- **Sensory encoder**: sensory frame → input currents on named neuron sets. Gains scaled by physiology and personality.
- **Motor decoder**: descending populations → intents. DNa02 to yaw, DNp09 to forward, giant fiber to escape (`Startle` skill plus backward `vx`), moonwalker DNs to backward, grooming circuits to `Preen`/`Ruffle`. Thresholds and smoothing are the tuning surface.
- **Plasticity**: dopamine-gated depression at Kenyon cell → MBON synapses, gated by PAM (reward) and PPL1 (punishment) activity that the encoder drives from sugar contact, petting, and startle.
- **Shuffled-weights control** behind a flag.
- **Headless mode**: scripted world, time scaled up, CSV of drives, DN rates, and plasticity weights per simulated hour. Every gate's test runs here.

### 3.4 Vision pipeline (decision 5)

Flies have about 750 ommatidia per eye; `flyvis` models the visual system on a hexagonal lattice of 721 columns and is connectome-constrained. Plan:

1. **Acquire** a per-duck view: 1D raycast fan in the 2D stub; a low-res rendered camera in MuJoCo; the WebRTC camera stream on a real duck, decoded on the laptop.
2. **Resample** to the hex lattice (721 luminance samples, frontal field only, since one camera covers roughly the frontal 90° of a fly's 270°). Compute ON and OFF contrast per column with a short temporal filter, which is what the lamina does.
3. **Inject** per column into the lamina input neurons (L1 ON pathway, L2 OFF pathway) of the matching optic-lobe column. This needs the column-to-neuron assignment, which is **not** in the standard FlyWire export. Sources: the Codex "Visual Columns Mapping Challenge" download (sign-in) or Matsliah et al. 2024 Supplementary Data 2. `flyvis` is confirmed at 721 columns, MIT licence. **Risk**: if per-column assignment is not clean in the export, fall back to driving `flyvis` as a front end and feeding its output neurons into the central brain. Gate 3 decides.
4. **Payoff**: looming detection (LPLC2), motion, and object approach come out of real wiring instead of a scalar hack, and the possession overlay can show the actual hex retina.

Transport size: 5 ducks × 721 columns × 2 channels × 60 Hz ≈ 430k floats/s, about 1.7 MB/s raw. Not a bottleneck on any option in §6.3, but wasteful as JSON.

### 3.5 The viewer's pipe, and the room that was dropped

The garden publishes a **world snapshot** for a viewer in another process (`viewer/snapshot.py`): one JSON datagram a step on the loopback, about 1.5 KB, from either body. It says where everything is, and each duck's pose, posture, mood, emote and commanded head. For the simulated robots it adds their real joint angles, trunk height and lean. For the selected duck it adds its needs, its social summary and which of its shown neurons fired. For a ridden duck it adds the two retinas and the descending readouts. The player's actions come back the same way, as the garden's own `garden.*` calls. The garden reports and ignores a call it cannot take, since what comes off a socket is not the garden's to trust. Godot draws and decides nothing.

A bite, a shove or a song is true for one step only. `frames.latest` gathers such events over every frame a slow brain skips and wipes them from a frame a quick brain sees twice. Each then counts once, however the two clocks fall.

The physical room (real microducks on a floor, an overhead tracker) was dropped on 2026-09-21. `PLAN.md` Phase E keeps what was planned.

### 3.6 Named neuron sets needed from FlyWire

Sugar GRNs (Gr5a, Gr64f), a few ORN glomeruli, bristle and Johnston's organ mechanosensory, lamina L1/L2 per column, LPLC2, DNa02, DNp09, giant fiber, moonwalker DNs, Kenyon cells, MBONs, PAM and PPL1 DANs, and grooming command neurons. Pulling and validating these is Gate 1.

### 3.7 Compute (decision 3, measure first)

The arithmetic is small: five brains × 2.7M synapses × 1 kHz is under 15 GMAC/s. The risk is kernel-launch overhead at 1,000 steps per second on Metal. Fly brains are sparse in time as well as space, so an event-driven CPU loop that propagates only actual spikes may win. Gate 1 benchmarks PyTorch sparse on MPS against an event-driven NumPy/Numba (or Rust) loop on the same brain and picks from data. Nothing else in the design depends on the answer.

### 3.8 Determinism and persistence (decision 7)

One seeded RNG. Fixed timestep. Brain state checkpointable. Save is physiology, plasticity weights, world objects, personality presets, and the seed. On launch, elapsed wall-clock time is simulated forward with rendering off, capped at a few simulated days. Always-on background brain later.

---

## 4. Visual brief

Built last, and changed by being looked at. The brief began (2026-09-16) as a Dreamcast low-poly garden finished as a halftone print in five inks on cream paper. On 2026-09-21 Chris saw it running beside a model of Sonic Adventure 2's Chao garden and asked for that. This section describes what the Godot garden (`viewer/godot/`) is now.

**The place** (`scenery.gd`). A lawn cradled in tall rounded rock with grass on top, along the two back edges; a pond in the back corner under a waterfall of pale stone columns with water on each; a fruit tree whose shade is real; a dark cave mouth and palms for scenery; a rail fence along the open front; blue sea below and a flat blue sky. Rock is made by rule, from lumps: rings of vertices pushed in and out. The cliffs stand outside the garden's square and the fence runs along it, so both draw walls the ducks already have; the waterfall's foot is rocks that are solid in the world, so nothing drawn can be walked through. A white flag on the near cliff flies the way the breeze goes. The garden people watch is 6 m a side; every gate but the soak and Gate 12 measures in a 4 m one.

**The finish** (`print.gdshader`). One unshaded material: flat faces from the geometry's own normals, the reference's colours, and a light halftone screen of navy for shading, fixed to the window. Blob shadows. An inside-out hull for the ink line round what needs its silhouette kept. Night is a wash of blue over the finished picture and adds no ink; as "more ink" it buried the garden in dots. From a duck's back the screen is mostly off.

**The ducks** (`duck.gd`). The real microduck: Pollen Robotics' meshes (Apache 2.0), posed from the robot's MuJoCo description and simplified to about ten thousand triangles (`build_robot.py`, `robot.json`), built as the robot is, every body on its own hinge. With the MuJoCo body it is posed by the simulated robot's real joints. With the 2D body it plays the robot's own motions, recorded once from the simulator (`record_clips.py`, `clips.json`): its sit, walk, turns, kick, roll, peck, fall and getting up. Only the dance is by rule, since no robot has one. Personality shows as size, width and head size, the grey plastic takes the duck's colour, and each duck has a voice of its own. Hats are made from a style number (`hats.gd`), five shapes in six colours.

**Sound.** Quacks, the pond and the drum are synthesised; nothing is shipped. The music box plays the player's own mp3s from a folder outside the repository.

**What is on screen.** A small card for the selected duck, bars for its needs, moods and wants, its brain firing, toasts in a corner, and a menu that stays collapsed until `/` opens it over the whole window. The ride view is first person, wide and a little coarse, with the hex retinas and descending-neuron bars as an overlay. A reduced-motion switch stills the camera and halves animation.

**Budget.** Under 300 KB for everything Godot loads (`gates/gate_14_look.py`); the robot and its motions are most of it.

**2D stub look.** Top-down pygame. Ducks are circles with heading lines and a retina fan. Mood is a halo. Food is a dot. Odor is a faint gradient. A debug column shows DN firing and drive bars per duck. Every gate through Gate 4 is judged in this view.

---

## 5. Sources for this file

- pollen-robotics/microduck: README; `docs/design/robotd-design.md` (JSON-RPC methods, 61-float observation, 50 Hz loop); `docs/design/simulation.md` (MuJoCo multi-duck sim, Mac native); `docs/robot/duckctl.md` (skills, policies, teleop refusal over BLE); `docs/ideas/autonomous_behavior.md` (sixteen-state machine, BLE presence, "inputs, not modes"). Fetched 2026-09-16.
- `RESEARCH.md` §1a (Chao trait and drive system, verified against chao-island.com earlier), §2, §4, §8.
- flyvis (Lappalainen et al. 2024) for the 721-column hex lattice and connectome-constrained visual model.
- Not fetched today (403): chao-island.com Personality & Emotion page. The SA2 label list in §2.4 is from memory and marked to verify.

---

## 6. Open decisions with tradeoffs

### 6.1 Where does the world simulation live?

"World" means positions, odor and temperature fields, food, contact, day/night. Physics is owned by whichever body is running, so the question is only about the environment layer.

| Option | For | Against |
|---|---|---|
| **A. Python, beside the brain** (recommended) | One codebase for headless tuning and live play. 2D stub is a dumb drawing of it. Deterministic and checkpointable with the brain. Same odor/temperature code feeds all three bodies | Duplicates positions the MuJoCo sim or room tracker already know; needs a sync step each frame (cheap at 5 ducks). Python is the slowest place for an odor diffusion field (fine at a 64×64 grid, 30 Hz) |
| B. In the body (MuJoCo plugin / Godot script / tracker) | No sync; physics and environment in one place | Three implementations of odor and day/night, one per body. Headless tuning has to launch a body. Godot and MuJoCo are poor places for deterministic replay |
| C. Split: physics-adjacent things in the body, fields in Python | Each thing where it's cheapest | Two owners of "where is the food," which is the classic bug factory |

Recommendation: A. The body owns physics; Python owns everything the brain smells and feels.

### 6.2 Renderer and final room

| Option | For | Against |
|---|---|---|
| **A. Physical microducks in a real room, app as companion view** | It is literally the r/robots demo with five ducks and it is the strongest hook. Bodies, camera, ToF, speaker, sim-to-real all exist and are maintained by someone else. Sim is the same contract, so all system gates run in MuJoCo first | Cost: five robots. Room needs an overhead tracker for odor synthesis and god-view. Battery life bounds session length. Aesthetic control is limited to the companion view and the ducks' own voice/posture |
| B. Virtual Dreamcast garden in Godot 4, MuJoCo or kinematic bodies underneath | Full aesthetic control, zero hardware, ships to anyone. Godot gives orbit camera, raycast, fog, filtering, saves, exports | Nobody else maintains the body. Godot must render the MuJoCo state or run its own kinematics (losing the shared physics). The hook is weaker than a real robot on a floor |
| C. Both: Godot garden as the "skin" for the MuJoCo body now, physical room later | The garden is a viewer, not a body, so it never blocks the system work. Same brain, same contract, swap the body when robots arrive | Two viewers to maintain (2D debug already exists, so really three). Risk of polishing the garden before the system is proven, which is the failure mode you named |
| D. Three.js inside Tauri instead of Godot | Web skills reuse; the earlier Three.js plan in `RESEARCH.md` §5 applies as written | Hand-rolled camera, picking, fog, saves. WebGPU in Tauri's webview on macOS is workable but a second platform surface to debug |

Recommendation: C, with the discipline that the Godot garden does not start until the MuJoCo gates pass. If the answer is "physical room only," Godot is dropped and the companion view is the 2D debug view grown up.

### 6.3 Transport between brain and body

| Option | For | Against |
|---|---|---|
| **A. Microduck's JSON-RPC 2.0 over NDJSON (Unix socket locally, LAN/WebRTC to real robots) for intents; a separate binary channel for sensory frames** (recommended) | Intents already speak the robot's language, so the real body needs no adapter. Sensory frames (retina, 1.7 MB/s) go as raw float32 over a local UDP socket or shared memory | Two channels to keep in step; timestamps in both frames solve it |
| B. JSON over WebSocket for everything | One channel, trivial to debug, language-agnostic | Retina frames as JSON are 4 to 5× the bytes and cost CPU on both ends at 60 Hz. Still an adapter to the robot's JSON-RPC |
| C. Shared memory ring buffers | Lowest latency, zero copies | Same-machine only, so the real robots and any remote viewer need a second path anyway. Painful across Python, Rust, and Godot |

Recommendation: A. Speak the robot's protocol for commands because you have to anyway, and don't JSON-encode a retina.

---

## 7. Items to verify before the relevant gate

- Verified: column-to-neuron assignment is not in the standard export (see §3.4).
- Codex sign-in and terms acceptance are a Chris action.
- Verified: scripted LAN teleop is SSH tunnel to `/run/robotd.sock` only (§3.5).
- Verified: Pi Camera v2 (IMX219, about 62° horizontal, 720p30 tested); ToF VL53L5CX/8CX 8×8 zones, 45° field, 4 m range.
- FlyWire data license terms for a shipped app (believed CC BY 4.0; confirm).
- chao-island.com SA2 personality label list (blocked today).
- Verified: `flyvis` is MIT, 721 columns. Open: whether its lamina input convention matches the Codex column ids.
