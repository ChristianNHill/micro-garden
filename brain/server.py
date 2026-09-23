"""Brain server: sensory frames in, microduck intents out, one brain per body.

Each 20 ms body step: newest frame per duck -> encode -> 2 brain ticks -> decode -> robot calls.
Each directional sense feeds its own side's neurons; sugar and pond water feed the taste neurons,
which FlyWire does not split by side.

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
from brain import emotes, reactions, social
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
# high, so a duck flees a stink it is standing in, not a faint one it can just smell
DANGER_HALF = 1.5
# Antennae 10 cm apart see about 1.17:1; steering DNs need about 3:1. A modelling assumption standing
# in for peripheral sharpening; real and shuffled brains get the same input.
ODOR_CONTRAST = 8.0
# The 8 was tuned at a decay length of 0.625 m. Two antennae differ by about 5 cm / decay length, so
# diffused smells scale the contrast by their decay length to keep the left/right ratio the same.
PLUME_CONTRAST = ODOR_CONTRAST * (np.sqrt(DIFFUSION / DECAY) * CELL_M) / 0.625
DUCK_HALF = 0.3  # a duck about half a metre away gives an input level of 0.5
HUMID_HALF = 0.1  # humidity 1 m from the pond edge is about 0.08
DRY_LEVEL = 0.2  # dry-air neurons at full dryness; kept low so dry air does not swamp other senses
TEMP_COMFORT_C, TEMP_SPAN_C, TEMP_LEVEL = 25.0, 5.0, 0.5
TOUCH_LEVEL = 0.2  # bristle input per side while another duck touches that side
PET_LEVEL = 0.6  # a hand on the head, on every bristle
VOICE_COOLDOWN_S = 3.0
NO_SPIKES = np.empty(0, np.int64)  # every sense is graded
SCARE_LEVEL = 0.8  # a clap, straight onto the looming detectors
# The body reports a clap on one 20 ms step. Held for 200 ms the giant fiber gives 7 to 9 spikes; one
# step gives only 2 or 3, too few to startle reliably.
WEAR_P = (0.1, 0.95)  # chance a duck puts on a hat it finds, at no vanity and at full
HAT_SHY_S = 30.0
DANCE_EVERY_S = 3.0  # about one dance's length
DANCE_REST_S = (10.0, 25.0)  # rest between dances
PERFORM_BORED = 0.5  # boredom past which a duck starts to entertain itself
DANCE_LOUD = 0.1  # music level for full dancing
DRUM_EVERY_S = 0.6  # how often a duck at a drum decides whether to tap it
KICK_EVERY_S = 1.0  # same, for a ball at its feet
CLAP_S = 0.2
SIDED = ["orn_food", "orn_danger", "moist_air", "dry_air", "heat", "cold", "bristle", "orn_pheromone",
         "jo_push", "jo_pull"]
ANTENNA_OUT = np.pi / 4  # each antenna points about 45 degrees out from the nose
MUSIC_HALF = 0.4  # loudness that gives an input level of 0.5 on the Johnston's organ
HAT_LEVEL = 0.5  # a worn hat, on every bristle


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
    # Wind pushes back the antenna it blows onto and pulls the far one forward, most along that
    # antenna's axis, so the two sides tell where it comes from.
    for side, out in (("left", ANTENNA_OUT), ("right", -ANTENNA_OUT)):
        along = f["wind"] * np.cos(f["wind_from"] - out)
        lv[f"jo_push_{side}"], lv[f"jo_pull_{side}"] = np.maximum(along, 0), np.maximum(-along, 0)
    music = (f["music_left"] + f["music_right"]) / 2
    lv["johnstons_organ"] = music / (music + MUSIC_HALF)
    lv["LPLC2"] = SCARE_LEVEL * f["scared"]
    lv["sugar"], lv["water_taste"] = f["sugar"], f["water"]  # combined into sugar_grn after gains
    return lv


class BrainServer:
    def __init__(self, W, ann, sets, bodies, seed: int, personality: dict | None = None,
                 hunger=0.5, thirst=0.5, provoked=0.0, body_temp=24.0, eyes: bool = True, learns: bool = False, **knobs):
        """bodies: list of (robot socket path, frame UDP port), one brain each. personality, knobs
        (brain/personality.py names) and starting physiology are scalars or one value per body; unset
        knobs are 0.5.

        eyes=False leaves the ducks blind and skips flyvis (vision costs about 12 ms a step on top of 14).
        learns=True gives each duck its own mushroom body (brain/plasticity.py); off by default.
        """
        side = ann["side"].to_numpy()
        self.sets = dict(sets)
        for name in SIDED:
            for s in ("left", "right"):
                self.sets[f"{name}_{s}"] = sets[name][side[sets[name]] == s]
        self.n = len(bodies)
        # One server can drive several gardens, and a frame names other ducks by their number in its own
        # garden. A garden's ducks share a socket directory; self.first is each duck's garden offset.
        dirs = [os.path.dirname(path) for path, _ in bodies]
        self.first = np.array([dirs.index(d) for d in dirs])
        knobs = {**(personality or {}), **knobs}
        self.brain = LIF(strip(W, sets) if learns else W, self.n)
        sparsen(self.brain, sets)  # a sparse odor code, as in the fly
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
        # Ducks the player is steering: the brain still runs and returns intents,
        # but they are not sent to the legs.
        self.possessed = np.zeros(self.n, bool)
        self.zooming = np.zeros(self.n, bool)
        self.emote_rng = np.random.default_rng(seed + 7)  # separate, so emotes do not shift the senses' noise
        self.slip_rng = np.random.default_rng(seed + 13)  # mishaps, on their own dice for the same reason
        self.mishap = [None] * self.n
        self.laughs = np.zeros(self.n, bool)
        self.squabble = np.zeros(self.n, bool)
        self.reactions = reactions.Reactions(self.n, np.random.default_rng(seed + 17))
        self.shown: list[reactions.Stimulus] = []  # feelings acted out last step, for the others to react to
        self.react_acts: list[list[str]] = [[] for _ in range(self.n)]
        social.init(self.body, seed)  # bonds, trust, skills live on the body so a save keeps them
        self.brainview = BrainView(ann, self.sets, self.brain.dev)  # idle until a duck is watched
        self.hat_was_near = np.zeros(self.n, bool)
        self.kick_at = np.zeros(self.n)
        self.drum_at = np.zeros(self.n)
        self.dance_at = np.zeros(self.n)
        self.dancing = np.zeros(self.n)
        self.performing = np.zeros(self.n)
        self.playing = np.zeros(self.n)
        self.hat_shy_until = np.zeros(self.n)  # after shaking a hat off, a duck leaves hats alone a while
        self.emote_at = self.emote_rng.uniform(0, emotes.EVERY_S, self.n)  # staggered across ducks
        self.t = 0.0

    def step(self, lockstep: bool) -> list[dict]:
        """One body step. lockstep=True sends intents as answered requests so a stepped body sees them."""
        read = frames.newer if lockstep else frames.latest
        self.frames = [read(r, f) for r, f in zip(self.rx, self.frames)]
        f = np.array([x if x is not None else frames.blank() for x in self.frames], frames.FRAME)
        body = self.body
        self.t += BODY_DT_MS / 1000
        falls_asleep, wakes = body.step(BODY_DT_MS / 1000, f, self.escaped, self.last_vx)
        f = self._in_all_gardens(f)
        self.laughs = social.update(body, f, BODY_DT_MS / 1000, self.last_vx, self.slip_rng)
        self.squabble = social.squabbles(body, f, BODY_DT_MS / 1000, self.slip_rng)
        self.mishap = social.mishaps(body, f, BODY_DT_MS / 1000, self.last_vx, self.slip_rng)
        left, right = self._scents(f)
        among = social.steering(body, f, (left, right))
        self._react(body, f, left, right, among)

        levels = self._levels(f)
        self.decoder.body = self._decoder_input(f, levels, among)
        # Senses are graded currents, not random spikes, so a smell reads the same from one whiff to the
        # next. Vision is graded too, so the two just concatenate.
        drive = graded_senses(self.sets, levels, self.n, self.brain.dev, self.rng)
        if self.vision is not None:
            eye = self.vision.step(f["lum"], body.sense_gains()["vision"])
            drive = (torch.cat([drive[0], eye[0]]), torch.cat([drive[1], eye[1]], dim=1))
        # Sugar and a hand on the head are what dopamine is for; a startle is the punishing kind.
        reward = np.maximum(f["ate"], f["petted"]) if self.plastic else None
        punish = np.maximum(f["scared"], self.escaped.astype(float)) if self.plastic else None
        self.escaped[:] = False  # only after punish has read it
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

    def _react(self, body, f, left, right, among) -> None:
        """Explicit code: this step's stimuli, the others' responses to them, and the ducks walking over to
        comfort or to shove (brain/reactions.py). The intents steer through the decoder."""
        stimuli, R = self.shown, reactions.Stimulus
        self.shown = []
        for j in range(self.n):
            by = int(f["bumped_by"][j])
            if by >= 0:
                stimuli.append(R("shove", by, j, self.reactions.heat_of.pop((by, j), 1.0)))
            by = int(f["comforted_by"][j])
            if by >= 0:
                stimuli.append(R("comfort", by, j))
        stimuli += [R("fall", i) for i, m in enumerate(self.mishap) if m in ("trip", "flounder", "whiff")]
        stimuli += [R("laugh", i, int(f["saw_fall_by"][i])) for i in np.flatnonzero(self.laughs)]
        total = left + right
        near = total / (total + reactions.SCENT_HALF)
        acts = self.reactions.react(body, stimuli, near, f["light"], self.t)
        turn, want, arrived = self.reactions.steer(body, f, left, right, self.t)
        among["intent_turn"], among["intent"] = turn, want
        among["social_want"] = np.maximum(among["social_want"], want)
        self.react_acts = [a + b for a, b in zip(acts, arrived)]

    def _in_all_gardens(self, f):
        """The frames with duck ids made global across this server's gardens."""
        f = f.copy()
        for name in frames.IDS:
            if name != "ate_kind":  # a fruit, not a duck
                f[name] = np.where(f[name] >= 0, f[name] + self.first, -1)
        return f

    def _scents(self, f) -> tuple[np.ndarray, np.ndarray]:
        """Each duck's smell of each other duck, (n, n) left and right, from the per-garden columns."""
        j = self.first[:, None] + np.arange(frames.MAX_DUCKS)[None]  # global id of each column
        same = (j < self.n) & (self.first[np.minimum(j, self.n - 1)] == self.first[:, None])
        rows = np.broadcast_to(np.arange(self.n)[:, None], j.shape)
        left, right = np.zeros((self.n, self.n)), np.zeros((self.n, self.n))
        left[rows[same], j[same]] = f["scent_left"][same]
        right[rows[same], j[same]] = f["scent_right"][same]
        return left, right

    def _decoder_input(self, f, levels, among) -> dict:
        """What the body and the garden tell the decoder this step besides the spikes: the likes and wants
        that get a duck walking and turning (see brain/decoder.py)."""
        body = self.body
        # Likes give way to a pressing need, so a duck that loves music does not sit by it and starve.
        at_ease = 1 - pressing(np.maximum(body.hunger, body.thirst))
        tune = np.abs(2 * body.k["music_affinity"] - 1) * levels["johnstons_organ"] * at_ease  # strong taste, loud music
        # Explicit code: how much a duck feels like dancing. Only a duck past the middle of the music dial dances.
        self.dancing = np.clip(2 * body.k["music_affinity"] - 1, 0, 1) * np.clip(levels["johnstons_organ"] / DANCE_LOUD, 0, 1) * at_ease
        # Explicit code: a bored duck sings or dances on its own, more if chatty or playful.
        bored = np.clip((body.boredom - PERFORM_BORED) / (1 - PERFORM_BORED), 0, 1) * at_ease
        self.performing = np.maximum(self.dancing, 0.6 * bored * np.maximum(body.k["chattiness"], body.k["playfulness"]))
        # Explicit code: wanting to play plus seeing a ball gets a duck walking; the decoder turns it
        # towards the eye the ball is in.
        # Explicit code: a duck that loves music wants to play an instrument it can see, the more when bored,
        # whether or not it is playful; the chattiest and most musical personalities play the most.
        instrument = np.clip(3 * np.maximum(f["drum_left"], f["drum_right"]), 0, 1)
        musical = np.clip(2 * body.k["music_affinity"] - 1, 0, 1) * instrument * (0.4 + 0.6 * body.boredom)
        self.playing = np.maximum(body.play(), musical) * at_ease
        toy_left, toy_right = np.maximum(f["ball_left"], f["drum_left"]), np.maximum(f["ball_right"], f["drum_right"])
        ball = self.playing * np.maximum(toy_left, toy_right)  # whichever toy it sees plainer
        to_pond, to_shade = body.cooling()
        cool_left, cool_right = (to_pond * f[f"pond_{s}"] + to_shade * f[f"shade_{s}"] for s in ("left", "right"))
        wants = np.maximum.reduce([tune, ball, among.pop("social_want") * at_ease, to_pond + to_shade])
        return {**body.motor(wants=wants, damp=(f["humidity_left"] + f["humidity_right"]) / 2), **among,
                "cool_left": cool_left, "cool_right": cool_right,
                "play": self.playing, "ball_left": toy_left, "ball_right": toy_right,
                "surge": self.following, "swimming": f["swimming"] > 0, "at_shore": f["water"] > 0,
                "thirst": body.thirst, "hatted": f["hat"] > 0, "fear": body.fear,
                "hunger": pressing(body.hunger),  # how far hunger outranks a smell it likes
                "tasting": (f["sugar"] > 0) | (f["water"] > 0),
                # the ears get the antennae's contrast gain: raw, two ears 10 cm apart differ by
                # only 0.01 of full loudness, far below the steering noise
                **dict(zip(("music_left", "music_right"),
                           (at_ease * m for m in bilateral(f["music_left"], f["music_right"], MUSIC_HALF)))),
                # other ducks by smell, for company, flight and pursuit; company gives way to a
                # pressing need (`at_ease`), fear and a fight do not
                **dict(zip(("duck_left", "duck_right"), bilateral(f["duck_left"], f["duck_right"], DUCK_HALF))),
                "at_ease": at_ease,
                "sociability": body.k["sociability"],
                "fondness": self.plastic.fondness() if self.plastic is not None else 0.0,
                "music_affinity": body.k["music_affinity"], "vanity": body.k["vanity"]}

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
        # Explicit code: a fly finds food by turning upwind when it smells it (no descending neuron here
        # carries the left/right comparison, and the fan-shaped body gating is not modelled). So the body
        # turns up the wind sense as far as the duck smells food it wants, scaled by pressing hunger
        # because the food gain has a floor. Water works the same way through damp air; its gain is
        # already thirst.
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
        if it["attack"] or self.squabble[i]:  # the brain's attack, or a shove personality made likely
            send("robot.do", skill=social.strike(self.body, i, self.slip_rng))
        if self.mishap[i]:
            send("robot.do", skill=self.mishap[i])
        if self.laughs[i]:
            send("robot.do", skill="emote_laugh")
        for skill in self.react_acts[i]:  # a reaction: an emote now, or the shove it walked over to give
            send("robot.do", skill=social.strike(self.body, i, self.slip_rng) if skill == "strike" else skill)
        if it["preen"]:
            send("robot.do", skill="preen")
            self.hat_shy_until[i] = self.t + HAT_SHY_S
        if not self.body.asleep[i]:
            self._toy(send, i, f["ball_near"] > 0, self.kick_at, KICK_EVERY_S, "kick")
            self._perform(send, i)
            self._toy(send, i, f["drum_near"] > 0, self.drum_at, DRUM_EVERY_S, "drum")
            self._consider_hat(send, i, f)
        self.hat_was_near[i] = f["hat_near"] > 0
        if it["zoomies"] and not self.zooming[i]:  # once, as they start
            send("robot.do", skill="zoomies")
        self.zooming[i] = it["zoomies"]
        self._emote(send, i)
        tag = self._voice(i, f, falls_asleep, wakes)
        if tag:
            send("robot.sound", tag=tag)

    def _toy(self, send, i, near, at, every_s, skill) -> None:
        """Explicit code: a duck that wants to play kicks a ball or taps a drum, decided every `every_s`."""
        if near and self.t >= at[i]:
            at[i] = self.t + every_s
            if self.emote_rng.random() < self.playing[i]:
                send("robot.do", skill=social.plays(self.body, i, self.slip_rng) if skill == "drum" else skill)

    def _perform(self, send, i) -> None:
        """Explicit code: a duck may sing or dance, asked every DANCE_EVERY_S, with a rest after each."""
        if self.t < self.dance_at[i]:
            return
        self.dance_at[i] = self.t + DANCE_EVERY_S
        if self.emote_rng.random() >= self.performing[i] * (0.6 + 0.8 * self.body.dance_skill[i]):  # a practised duck performs more
            return
        # a chatty duck sings, a playful or music-loving one dances; each is decided on its own
        k = self.body.k
        sings = self.emote_rng.random() < 0.3 + 0.6 * k["chattiness"][i]
        dances = self.emote_rng.random() < 0.3 + 0.6 * max(k["playfulness"][i], self.dancing[i]) or not sings
        send("robot.do", skill="emote_" + ("singdance" if sings and dances else "sing" if sings else "dance"))
        if dances and social.falls_dancing(self.body, i, self.slip_rng):
            send("robot.do", skill="trip")
        self.body.amuse(i)
        social.performed(self.body, i)
        self.dance_at[i] += self.emote_rng.uniform(*DANCE_REST_S)

    def _consider_hat(self, send, i, f) -> None:
        """Explicit code: a duck that finds a hat decides once whether to put it on; vanity sets the chance."""
        comes_upon = f["hat_near"] > 0 and not self.hat_was_near[i]
        if comes_upon and f["hat"] == 0 and self.t >= self.hat_shy_until[i]:
            if self.emote_rng.random() < WEAR_P[0] + (WEAR_P[1] - WEAR_P[0]) * float(self.body.k["vanity"][i]):
                send("robot.do", skill="wear")

    def _emote(self, send, i) -> None:
        """Every EVERY_S a duck may act out its strongest feeling (brain/emotes.py)."""
        if self.t < self.emote_at[i]:
            return
        self.emote_at[i] = self.t + emotes.EVERY_S
        emote = emotes.pick(self.body, i, self.emote_rng)
        if emote:
            send("robot.do", skill=f"emote_{emote}")
            shown = {"angry": "angry", "stomp": "angry", "sad": "sad", "cry": "sad", "lonely": "sad"}.get(emote)
            if shown:
                self.shown.append(reactions.Stimulus(shown, i))
            if emote in emotes.AMUSING:
                self.body.amuse(i)

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
