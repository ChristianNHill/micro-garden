"""Brain server: sensory frames in, microduck intents out, one brain per body (PLAN.md Gate 4).

Each 20 ms body step: newest frame per duck -> encode -> 2 brain ticks -> decode -> robot.move, plus
robot.do ground_pick while feeding. Encoding: each directional sense feeds its own side's neurons
(food and danger smell, moist and dry air, heat and cold, touch); sugar or pond water on contact feeds
the sugar/water taste neurons, which FlyWire does not split.

Drive the free-running stub:  uv run python -m body.stub2d.stub --view   then
                              uv run python -m brain.server
"""
import argparse
import os
import string
import time

import numpy as np
import torch

from body import frames
from body.contract import Client
from brain import emotes, social
from brain.brainview import BrainView
from brain.data import load_connectome, named_sets, shuffled
from brain.decoder import Decoder
from brain.encoder import graded as graded_senses
from brain.lif import DT_MS, LIF
from brain.plasticity import Plasticity, sparsen, strip
from brain.vision import VIS_TONIC, Vision
from brain.physiology import Physiology, aggression_tone, pressing
from world.fields import CELL_M, DECAY, DIFFUSION

BODY_DT_MS = 20.0
TICKS_PER_STEP = int(BODY_DT_MS / DT_MS)
ODOR_HALF = 0.5  # odor concentration that gives input level 0.5; the dish itself is about 3
# A stink now carries as far as food does, so a duck would read a faint one everywhere as alarm. It
# should flee a stink it is standing in, not one it can just smell (audit, 2026-09-19).
DANGER_HALF = 1.5
# Antennae 10 cm apart see about 1.17:1; steering DNs need about 3:1 (Gate 4 model work). This gain on
# the normalized left/right difference is a modelling assumption standing in for peripheral sharpening.
# Real and shuffled brains get the same input.
ODOR_CONTRAST = 8.0
# That 8 was set when a smell's decay length was 0.62 m. Two antennae 10 cm apart in an exponential
# plume differ by about (10 cm / 2) / decay length, so a smell that carries further is a shallower one:
# at 1 m the difference is 5% where it was 8%, and Gate 4 fell from 20/20 to 10/20 with nothing else
# changed (A/B, 2026-09-19). The diffused smells scale the contrast with their decay length, so the
# brain is handed the same left/right ratio it was tuned on however far the smell carries.
PLUME_CONTRAST = ODOR_CONTRAST * (np.sqrt(DIFFUSION / DECAY) * CELL_M) / 0.625
DUCK_HALF = 0.3  # a duck about half a metre away gives an input level of 0.5
HUMID_HALF = 0.1  # humidity 1 m from the pond edge is about 0.08
DRY_LEVEL = 0.2  # dry-air neurons at full dryness; kept low so dry air does not swamp other senses
TEMP_COMFORT_C, TEMP_SPAN_C, TEMP_LEVEL = 25.0, 5.0, 0.5
TOUCH_LEVEL = 0.2  # bristle input per side while another duck touches that side
PET_LEVEL = 0.6  # a hand on the head is felt on every bristle, harder than a duck brushing past
VOICE_COOLDOWN_S = 3.0
NO_SPIKES = np.empty(0, np.int64)  # every sense is graded now; nothing is injected as spikes
SCARE_LEVEL = 0.8  # a clap, straight onto the looming detectors
# ... for as long as a clap lasts. The body reports it on one step, 20 ms, and driven for only that the giant
# fiber gave 2 or 3 spikes, so two claps in three startled nobody; over 200 ms it gives 7 to 9 (bench, 2026-09-20).
WEAR_P = (0.1, 0.95)  # chance a duck puts on a hat it comes upon, at no vanity and at full
HAT_SHY_S = 30.0
DANCE_EVERY_S = 3.0  # a dance lasts about this long, so a duck that goes on wanting to goes on dancing
DANCE_REST_S = (10.0, 25.0)  # between one dance and the next: back to back, five ducks danced without a pause (Chris)
PERFORM_BORED = 0.5  # boredom past which a duck starts to entertain itself
DANCE_LOUD = 0.1  # the Johnston's organ level at which the music is loud enough to dance to at full
DRUM_EVERY_S = 0.6  # a duck at a drum decides this often whether to tap it, so a keen one keeps a beat
KICK_EVERY_S = 1.0  # how often a duck with a ball at its feet decides whether to kick it
CLAP_S = 0.2
SIDED = ["orn_food", "orn_danger", "moist_air", "dry_air", "heat", "cold", "bristle", "orn_pheromone",
         "jo_push", "jo_pull"]
ANTENNA_OUT = np.pi / 4  # each antenna points about 45 degrees out from the nose
MUSIC_HALF = 0.4  # loudness that gives an input level of 0.5 on the Johnston's organ
HAT_LEVEL = 0.5  # a hat sits on the head, felt on every bristle for as long as it is there


def bilateral(left, right, half, contrast=ODOR_CONTRAST):
    """Per-side input levels: saturating overall level, left/right difference amplified by contrast."""
    mean = (left + right) / 2
    d = contrast * (left - right) / np.maximum(left + right, 1e-9)
    level = mean / (mean + half)
    return level * np.clip(1 + d, 0, 2), level * np.clip(1 - d, 0, 2)


def sense_levels(f: np.ndarray) -> dict:
    """Frame records (structured array, one per brain) -> encoder levels."""
    lv = {}
    for name, key, half, contrast in (("orn_food", "odor", ODOR_HALF, PLUME_CONTRAST),
                                      ("orn_danger", "danger", DANGER_HALF, PLUME_CONTRAST),
                                      ("moist_air", "humidity", HUMID_HALF, ODOR_CONTRAST),
                                      ("orn_pheromone", "duck", DUCK_HALF, ODOR_CONTRAST)):
        lv[f"{name}_left"], lv[f"{name}_right"] = bilateral(f[f"{key}_left"], f[f"{key}_right"], half, contrast)
    for side in ("left", "right"):
        temp = f[f"temp_{side}"]
        lv[f"heat_{side}"] = TEMP_LEVEL * np.clip((temp - TEMP_COMFORT_C) / TEMP_SPAN_C, 0, 1)
        lv[f"cold_{side}"] = TEMP_LEVEL * np.clip((TEMP_COMFORT_C - temp) / TEMP_SPAN_C, 0, 1)
        lv[f"dry_air_{side}"] = DRY_LEVEL * (1 - f[f"humidity_{side}"])
        lv[f"bristle_{side}"] = np.maximum(np.maximum(TOUCH_LEVEL * np.minimum(f[f"touch_{side}"], 1),
                                                      PET_LEVEL * f["petted"]),  # a hand covers both sides
                                           HAT_LEVEL * f["hat"])
    # Wind pushes back the antenna it blows onto and pulls the far one forward, most for wind along
    # that antenna's own axis, so the two sides between them say where it is coming from.
    for side, out in (("left", ANTENNA_OUT), ("right", -ANTENNA_OUT)):
        along = f["wind"] * np.cos(f["wind_from"] - out)
        lv[f"jo_push_{side}"], lv[f"jo_pull_{side}"] = np.maximum(along, 0), np.maximum(-along, 0)
    music = (f["music_left"] + f["music_right"]) / 2
    lv["johnstons_organ"] = music / (music + MUSIC_HALF)
    lv["LPLC2"] = SCARE_LEVEL * f["scared"]  # the clap the player makes
    lv["sugar"], lv["water_taste"] = f["sugar"], f["water"]  # combined into sugar_grn after gains
    return lv


class BrainServer:
    def __init__(self, W, ann, sets, bodies, seed: int, personality: dict | None = None,
                 hunger=0.5, thirst=0.5, provoked=0.0, body_temp=24.0, eyes: bool = True, learns: bool = False, **knobs):
        """bodies: list of (robot socket path, frame UDP port), one brain each. personality and knobs
        (brain/personality.py names) and starting physiology are scalars or one value per body; unset
        knobs are 0.5.

        eyes=False leaves the ducks blind and skips flyvis, for gates that only test the other senses
        or want the wall time back: vision costs about 12 ms a step on top of 14.

        learns=True gives each duck its own mushroom body, depressed by sugar and by the player's hand
        (brain/plasticity.py). Off by default: a duck that learns behaves differently in every scenario,
        so it wants proving at Gate 8 before the earlier gates inherit it.
        """
        side = ann["side"].to_numpy()
        self.sets = dict(sets)
        for name in SIDED:
            for s in ("left", "right"):
                self.sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
        self.n = len(bodies)
        knobs = {**(personality or {}), **knobs}
        self.brain = LIF(strip(W, sets) if learns else W, self.n)
        sparsen(self.brain, sets)  # a sparse odor code, as in the fly (Gate 7)
        self.vision = Vision(ann, self.n, device=self.brain.dev) if eyes else None
        if self.vision is not None:
            self.brain.calibrate(self.vision.index, VIS_TONIC)
        self.body = Physiology(self.n, knobs, hunger, thirst, provoked, body_temp)
        self.plastic = Plasticity(W, sets, self.n, self.brain.dev, self.body.k["smarts"]) if learns else None
        self.decoder = Decoder(ann, self.sets, self.n, seed, self.body.k["stink_affinity"])
        self.rng = np.random.default_rng(seed)
        self.voice_rng = np.random.default_rng(seed + 1)
        self.robots = [Client(path) for path, _ in bodies]
        self.rx = [frames.receiver(port) for _, port in bodies]
        self.frames = [None] * self.n
        self.last_vx = np.zeros(self.n)
        self.escaped = np.zeros(self.n, bool)
        self.quiet_until = np.zeros(self.n)
        self.clap_left = np.zeros(self.n)  # seconds of the last clap still ringing
        # Ducks the player has taken the wheel of (ARCHITECTURE.md 2.6). The brain goes on seeing, smelling
        # and learning, and what it would have done is still returned; it just is not sent to the legs.
        self.possessed = np.zeros(self.n, bool)
        self.zooming = np.zeros(self.n, bool)
        self.emote_rng = np.random.default_rng(seed + 7)  # its own, so emotes leave the senses' noise as it was
        social.init(self.body, seed)  # bonds, trust in the hand, skills: on the body, so a saved garden keeps them
        self.brainview = BrainView(ann, self.sets, self.brain.dev)  # for a viewer; idle until a duck is watched
        self.hat_was_near = np.zeros(self.n, bool)
        self.kick_at = np.zeros(self.n)
        self.drum_at = np.zeros(self.n)
        self.dance_at = np.zeros(self.n)
        self.dancing = np.zeros(self.n)
        self.performing = np.zeros(self.n)
        self.playing = np.zeros(self.n)
        self.hat_shy_until = np.zeros(self.n)  # a duck that has just shaken a hat off leaves hats alone a while
        self.emote_at = self.emote_rng.uniform(0, emotes.EVERY_S, self.n)  # staggered, so five ducks do not emote as one
        self.t = 0.0

    def step(self, lockstep: bool) -> list[dict]:
        """One body step. lockstep=True sends intents as answered requests so a stepped body sees them."""
        read = frames.newer if lockstep else frames.latest
        self.frames = [read(r, f) for r, f in zip(self.rx, self.frames)]
        f = np.array([x if x is not None else frames.blank() for x in self.frames], frames.FRAME)
        body = self.body
        self.t += BODY_DT_MS / 1000
        falls_asleep, wakes = body.step(BODY_DT_MS / 1000, f, self.escaped, self.last_vx)
        social.update(body, f, BODY_DT_MS / 1000, self.last_vx)
        among = social.steering(body, f)

        levels = self._levels(f)
        # Music is in the garden all day now, so it is a like that gives way to a need as the others do:
        # a duck that loved it and could always hear it would otherwise sit by it and starve.
        at_ease = 1 - pressing(np.maximum(body.hunger, body.thirst))
        tune = np.abs(2 * body.k["music_affinity"] - 1) * levels["johnstons_organ"] * at_ease  # strong taste, loud music
        # How much a duck feels like dancing: it likes music (past the middle of the dial), it can hear some,
        # and nothing is pressing. A duck that dislikes music never dances to it.
        self.dancing = np.clip(2 * body.k["music_affinity"] - 1, 0, 1) * np.clip(levels["johnstons_organ"] / DANCE_LOUD, 0, 1) * at_ease
        # And a duck with no music makes its own when it is bored (Chris, 2026-09-21): a song or a dance is
        # something to do, the more so for a chatty or a playful duck, and doing it takes the edge off.
        bored = np.clip((body.boredom - PERFORM_BORED) / (1 - PERFORM_BORED), 0, 1) * at_ease
        self.performing = np.maximum(self.dancing, 0.6 * bored * np.maximum(body.k["chattiness"], body.k["playfulness"]))
        # A ball is wanted as far as the duck wants to play and can see one: that gets it walking, and the
        # decoder turns it towards the eye the ball is in. Explicit, as music is, and for the same reason.
        self.playing = body.play() * at_ease
        toy_left, toy_right = np.maximum(f["ball_left"], f["drum_left"]), np.maximum(f["ball_right"], f["drum_right"])
        ball = self.playing * np.maximum(toy_left, toy_right)  # a ball or a drum: whichever it sees plainer
        to_pond, to_shade = body.cooling()
        cool_left, cool_right = (to_pond * f[f"pond_{s}"] + to_shade * f[f"shade_{s}"] for s in ("left", "right"))
        wants = np.maximum.reduce([tune, ball, among.pop("social_want") * at_ease, to_pond + to_shade])
        self.decoder.body = {**body.motor(wants=wants, damp=(f["humidity_left"] + f["humidity_right"]) / 2), **among,
                             "cool_left": cool_left, "cool_right": cool_right,
                             "play": self.playing, "ball_left": toy_left, "ball_right": toy_right,
                             "surge": self.following, "swimming": f["swimming"] > 0, "at_shore": f["water"] > 0,
                             "thirst": body.thirst, "hatted": f["hat"] > 0, "fear": body.fear,
                             "hunger": pressing(body.hunger),  # how far hunger outranks a smell it likes
                             "tasting": (f["sugar"] > 0) | (f["water"] > 0),
                             # the ears get the contrast the antennae get: raw, two ears 10 cm apart differ
                             # by 0.01 of full loudness, a 0.01 rad/s turn under 1.5 of steering noise, and
                             # music never steered a duck (Gate 8b's old pass was two paths diverging)
                             **dict(zip(("music_left", "music_right"),
                                        (at_ease * m for m in bilateral(f["music_left"], f["music_right"], MUSIC_HALF)))),
                             # where the other ducks are, by smell, for company, flight and pursuit; company
                             # gives way to a pressing need (`at_ease`), fear and a fight do not
                             **dict(zip(("duck_left", "duck_right"), bilateral(f["duck_left"], f["duck_right"], DUCK_HALF))),
                             "at_ease": at_ease,
                             "sociability": body.k["sociability"],
                             "fondness": self.plastic.fondness() if self.plastic is not None else 0.0,
                             "music_affinity": body.k["music_affinity"], "vanity": body.k["vanity"]}
        # Senses release steadily rather than firing a random subset of each set per tick: the same
        # mean current with none of the sampling noise, which is what makes a smell recognisable from
        # one whiff to the next (Gate 7). Vision already worked this way, so the two just concatenate.
        drive = graded_senses(self.sets, levels, self.n, self.brain.dev, self.rng)
        if self.vision is not None:
            eye = self.vision.step(f["lum"], body.sense_gains()["vision"])
            drive = (torch.cat([drive[0], eye[0]]), torch.cat([drive[1], eye[1]], dim=1))
        # Sugar and a hand on the head are what dopamine is for; a startle is the punishing kind.
        reward = np.maximum(f["ate"], f["petted"]) if self.plastic else None
        punish = np.maximum(f["scared"], self.escaped.astype(float)) if self.plastic else None
        self.escaped[:] = False  # after punish has read it: cleared first, no escape ever punished
        for _ in range(TICKS_PER_STEP):
            spk = self.brain.step(NO_SPIKES, NO_SPIKES, graded=drive, plastic=self.plastic)
            if self.plastic is not None:
                self.plastic.step(spk, reward, punish)
            self.brainview.tick(spk)
            intents = self.decoder.update(spk)
            self.escaped |= [it["escape"] for it in intents]
        self.last_vx = np.array([it["vx"] for it in intents])

        for i, (robot, it) in enumerate(zip(self.robots, intents)):
            if self.possessed[i]:
                continue
            self._send(robot.call if lockstep else robot.notify, i, it, f[i], falls_asleep[i], wakes[i])
        return intents

    def _levels(self, f) -> dict:
        """Encoder levels for this step's frames, shaped by the body."""
        body = self.body
        levels = sense_levels(f)
        self.clap_left = np.where(f["scared"] > 0, CLAP_S, np.maximum(self.clap_left - BODY_DT_MS / 1000, 0))
        levels["LPLC2"] = SCARE_LEVEL * (self.clap_left > 0)
        wet = f["swimming"] > 0  # flies do not swim: wet reads as saturated humidity and touch all over
        for s in ("left", "right"):
            levels[f"moist_air_{s}"] = np.where(wet, 1.0, levels[f"moist_air_{s}"])
            levels[f"bristle_{s}"] = np.where(wet, TOUCH_LEVEL, levels[f"bristle_{s}"])
        gains = body.sense_gains()
        for key in levels:
            base = key.removesuffix("_left").removesuffix("_right")
            if base in gains:
                levels[key] = levels[key] * gains[base]
        # A fly finds food by turning into the wind when it smells it, not by comparing its antennae
        # (PLAN.md Gate 9b: no descending neuron carries the comparison). Nothing in this brain gates
        # wind on smell either, which the fly does in its fan-shaped body, so the body does it the way
        # it does everything else, by turning a sense up: a duck attends to the wind as far as it
        # smells food it wants, and food's own gain already carries the hunger.
        # Water is found the same way: a thirsty duck follows damp air up the wind, and thirst is already
        # in the damp sense's gain.
        # ... as far as hunger is pressing: the food sense keeps a floor so that a full duck still notices
        # a dish, and on that floor a duck that had just eaten went on following food smell at two thirds
        # strength. Water needs no such factor: its gain is thirst itself, plus what the duck likes.
        scent = np.clip((levels["orn_food_left"] + levels["orn_food_right"]) / 2, 0, 1) * pressing(body.hunger)
        damp = (np.clip((levels["moist_air_left"] + levels["moist_air_right"]) / 2, 0, 1)
                * (1 - body.at_water((f["humidity_left"] + f["humidity_right"]) / 2)) * ~wet)
        self.following = np.maximum(scent, damp) * (f["wind"] > 0)  # how far it is following its nose upwind
        for key in ("jo_push_left", "jo_push_right", "jo_pull_left", "jo_pull_right"):
            levels[key] = levels[key] * np.maximum(scent, damp)
        levels["sugar_grn"] = np.maximum(levels.pop("sugar"), levels.pop("water_taste"))
        food_odor = (f["odor_left"] + f["odor_right"]) / 2
        levels["pC1_aggr"] = aggression_tone(body.k["aggressiveness"], body.hunger, food_odor, body.anger,
                                             body.k["kindness"])
        return levels

    def _send(self, send, i, it, f, falls_asleep, wakes) -> None:
        """One duck's robot calls for this step."""
        if falls_asleep:
            send("robot.relax")
        if wakes:
            send("robot.init")
        send("robot.move", vx=it["vx"], vy=it["vy"], vyaw=it["vyaw"])
        if it["feed"]:
            send("robot.do", skill="drink" if f["water"] > 0 else "ground_pick")
        if it["attack"]:
            send("robot.do", skill="headbutt")
        if it["preen"]:
            send("robot.do", skill="preen")
            self.hat_shy_until[i] = self.t + HAT_SHY_S
        # A hat lying in the garden is the duck's to put on or walk past (Chris, 2026-09-21). It decides once,
        # as it comes upon one, and vanity is the chance: explicit, like shedding one, and a scale like it.
        # A ball at its feet is kicked, by a duck that wants to play: decided about once a second, not per tick.
        if f["ball_near"] > 0 and self.t >= self.kick_at[i] and not self.body.asleep[i]:
            self.kick_at[i] = self.t + KICK_EVERY_S
            if self.emote_rng.random() < self.playing[i]:
                send("robot.do", skill="kick")
        # A duck that likes what it hears dances (Chris, 2026-09-21): asked again every DANCE_EVERY_S, so it
        # dances now and then while the music and its mood last, with a rest after each.
        if self.t >= self.dance_at[i] and not self.body.asleep[i]:
            self.dance_at[i] = self.t + DANCE_EVERY_S
            if self.emote_rng.random() < self.performing[i] * (0.6 + 0.8 * self.body.dance_skill[i]):  # a practised duck performs more
                # a song, a dance, or mostly both: a chatty duck sings, a playful one or one that likes the
                # music dances, and each is decided by itself, so it is not always the same one
                k = self.body.k
                sings = self.emote_rng.random() < 0.3 + 0.6 * k["chattiness"][i]
                dances = self.emote_rng.random() < 0.3 + 0.6 * max(k["playfulness"][i], self.dancing[i]) or not sings
                send("robot.do", skill="emote_" + ("singdance" if sings and dances else "sing" if sings else "dance"))
                self.body.amuse(i)
                social.performed(self.body, i)
                self.dance_at[i] += self.emote_rng.uniform(*DANCE_REST_S)  # and then it has had its turn for a while
        if f["drum_near"] > 0 and self.t >= self.drum_at[i] and not self.body.asleep[i]:  # and a drum in reach is tapped
            self.drum_at[i] = self.t + DRUM_EVERY_S
            if self.emote_rng.random() < self.playing[i]:
                send("robot.do", skill="drum")
        near = f["hat_near"] > 0
        if near and not self.hat_was_near[i] and f["hat"] == 0 and not self.body.asleep[i] and self.t >= self.hat_shy_until[i]:
            if self.emote_rng.random() < WEAR_P[0] + (WEAR_P[1] - WEAR_P[0]) * float(self.body.k["vanity"][i]):
                send("robot.do", skill="wear")
        self.hat_was_near[i] = near
        if it["zoomies"] and not self.zooming[i]:  # once, as they start; a body shows them how it likes
            send("robot.do", skill="zoomies")
        self.zooming[i] = it["zoomies"]
        if self.t >= self.emote_at[i]:
            self.emote_at[i] = self.t + emotes.EVERY_S
            emote = emotes.pick(self.body, i, self.emote_rng)
            if emote:
                send("robot.do", skill=f"emote_{emote}")
                if emote in emotes.AMUSING:
                    self.body.amuse(i)
        tag = self._voice(i, f, falls_asleep, wakes)
        if tag:
            send("robot.sound", tag=tag)

    def _voice(self, i, f, falls_asleep, wakes) -> str | None:
        """A quack for this step, if any: events pick the tag, chattiness picks whether to speak."""
        if self.t < self.quiet_until[i]:
            return None
        tag = self._pick_voice(i, f, falls_asleep, wakes)
        if tag:
            self.quiet_until[i] = self.t + VOICE_COOLDOWN_S
        return tag

    def _pick_voice(self, i, f, falls_asleep, wakes) -> str | None:
        b = self.body
        events = [
            (self.escaped[i], "alarm"), (wakes, "greet"), (falls_asleep, "coo"),
            (f["ate"] > 0 or f["drank"] > 0, "chirp"), (f["bumped"] > 0, "alarm"),
        ]
        tag = next((t for happened, t in events if happened), None)
        if tag is None and not b.asleep[i]:
            # idle chatter, about once a minute at chattiness 1, colored by mood
            if self.voice_rng.random() < b.k["chattiness"][i] * BODY_DT_MS / 1000 / 60 * (1 + 3 * b.boredom[i]):
                tag = "peck" if b.sorrow[i] > 0.5 else "inquire" if b.boredom[i] > 0.5 else "chirp"
            return tag
        return tag if self.voice_rng.random() < 0.2 + 0.8 * b.k["chattiness"][i] else None

    def close(self) -> None:
        for c in self.robots:
            c.close()
        for r in self.rx:
            r.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ducks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sock-dir", default=os.path.expanduser("~/.cache/micro-garden"))
    ap.add_argument("--shuffled", action="store_true", help="drive the ducks with the shuffled control brain")
    args = ap.parse_args()
    W, ann = load_connectome()
    if args.shuffled:
        W = shuffled(W, args.seed)
    bodies = [(os.path.join(args.sock_dir, f"duck-{c}.sock"), frames.FRAME_PORT + i)
              for i, c in enumerate(string.ascii_lowercase[:args.ducks])]
    server = BrainServer(W, ann, named_sets(ann), bodies, args.seed)
    print(f"driving {args.ducks} ducks{' with shuffled brains' if args.shuffled else ''}; Ctrl-C to stop")
    next_t = time.monotonic()
    while True:
        server.step(lockstep=False)
        next_t += BODY_DT_MS / 1000
        lag = next_t - time.monotonic()
        if lag > 0:
            time.sleep(lag)
        else:
            next_t = time.monotonic()  # brain fell behind real time; don't try to catch up


if __name__ == "__main__":
    main()
