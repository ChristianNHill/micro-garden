# One duck, built from a dozen low-poly primitives and animated by rule. It faces +x and stands on y = 0.
# Its proportions come from its personality knobs, so who a duck is shows in its outline before it moves:
# a big appetite is a wide body, a timid duck is small with a long neck, a vain one grows its tuft.
# It is told where it is and how it feels (`show`), and everything else here is how that looks.
extends Node3D

const Ink := preload("res://ink.gd")
const CELL := 5.0  # a finer screen than the ground's: at 9 px a duck this small came out spotted like a dalmatian
const LOOK := 1.9  # drawn this much larger than life: a 4 m garden on one screen leaves a true duck a speck
const EMOTE_S := 2.4
const RIBBONS := [Ink.CORAL, Ink.TEAL, Ink.MUSTARD, Ink.NAVY, Ink.GRASS]
const MOOD_SHAPES := {"joy": "ball", "fear": "spike", "anger": "block", "sorrow": "drop"}

var model := Node3D.new()  # everything that waddles, sits and falls over; the shadow and the signs do not
var neck := Node3D.new()  # the head turns about this
var legs: Array[MeshInstance3D] = []
var hat: MeshInstance3D
var shadow: MeshInstance3D
var mood := Node3D.new()
var mood_shapes := {}
var sign := Label3D.new()
var ring: MeshInstance3D

var state := {}
var target := Vector3.ZERO
var speed := 0.0
var stride := 0.0
var emote := ""
var emote_age := 99.0
var emote_seen := -1.0
var calm := 1.0  # 0.5 under reduced motion


func build(index: int, knobs: Dictionary) -> void:
	var k := func(name: String) -> float: return float(knobs.get(name, 0.5))
	var girth: float = 0.85 + 0.4 * k.call("appetite")
	var size: float = 0.9 + 0.25 * k.call("aggressiveness") - 0.2 * k.call("timidity")
	var neck_len: float = 0.03 + 0.04 * k.call("timidity") + 0.02 * k.call("vanity")
	var bill: float = 0.7 + 0.7 * k.call("chattiness")
	var tuft: float = 0.4 + 1.4 * k.call("vanity")
	var leg: float = 0.04 + 0.04 * k.call("energy")
	var slump: float = 0.25 * k.call("sleepiness")

	add_child(model)
	model.scale = Vector3.ONE * LOOK * size
	var body_y := leg + 0.055
	var body := Ink.part(model, Ink.ball(0.07), Ink.CREAM, Vector3(0, body_y, 0), Vector3(1.25, 0.9 - slump * 0.3, girth), CELL, -1.0, true)
	Ink.part(model, Ink.cone(0.03, 0.07), Ink.CREAM, Vector3(-0.09, body_y + 0.03, 0), Vector3.ONE, CELL, -1.0, true).rotation.z = 1.0
	for side in [-1, 1]:
		Ink.part(model, Ink.ball(0.045), Ink.CREAM, Vector3(-0.01, body_y + 0.005, side * 0.06 * girth), Vector3(1.2, 0.7, 0.35), CELL, -1.0, true)
		var l := Ink.part(model, Ink.cone(0.009, leg, 0.009, 5), Ink.MUSTARD, Vector3(0.0, leg / 2, side * 0.03 * girth), Vector3.ONE, CELL)
		Ink.part(l, Ink.ball(0.022, 6), Ink.MUSTARD, Vector3(0.015, -leg / 2 + 0.004, 0), Vector3(1.3, 0.25, 1.0), CELL, -1.0, true)
		legs.append(l)

	neck.position = Vector3(0.055, body_y + 0.03, 0)
	neck.rotation.z = -slump
	model.add_child(neck)
	var head_at := Vector3(0.02, neck_len + 0.04, 0)
	Ink.part(neck, Ink.cone(0.028, neck_len + 0.03, 0.024), Ink.CREAM, Vector3(0.008, neck_len / 2, 0), Vector3.ONE, CELL, -1.0, true)
	Ink.part(neck, _torus(0.028, 0.012), RIBBONS[index % RIBBONS.size()], Vector3(0.004, 0.012, 0), Vector3.ONE, CELL, 0.9)
	Ink.part(neck, Ink.ball(0.052), Ink.CREAM, head_at, Vector3.ONE, CELL, -1.0, true)
	var beak := Ink.part(neck, Ink.cone(0.024, 0.06), Ink.CORAL, head_at + Vector3(0.05 + 0.02 * bill, -0.008, 0), Vector3(1.0, bill, 0.6), CELL, -1.0, true)
	beak.rotation.z = -PI / 2
	beak.scale = Vector3(0.6, bill, 1.0)  # after the turn: flat top to bottom, long forwards
	for side in [-1, 1]:
		Ink.part(neck, Ink.ball(0.009, 6), Ink.NAVY, head_at + Vector3(0.03, 0.015, side * 0.04), Vector3.ONE, CELL, 0.0)
	Ink.part(neck, Ink.cone(0.012, 0.03 * tuft), Ink.CREAM, head_at + Vector3(-0.01, 0.05 + 0.012 * tuft, 0), Vector3.ONE, CELL, -1.0, true).rotation.z = 0.5
	hat = Ink.part(neck, Ink.cone(0.04, 0.07), Ink.MUSTARD, head_at + Vector3(0, 0.075, 0), Vector3.ONE, CELL, -1.0, true)

	shadow = Ink.part(self, Ink.cone(0.1 * LOOK * size * girth, 0.001, 0.1 * LOOK * size * girth, 12), Ink.GRASS, Vector3(0, 0.003, 0), Vector3.ONE, 7.0, 0.35)
	ring = Ink.part(self, _torus(0.17 * LOOK, 0.19 * LOOK), Ink.CORAL, Vector3(0, 0.004, 0), Vector3(1, 0.2, 1), 7.0, 1.0)
	ring.visible = false

	mood.position = Vector3(0, (body_y + neck_len + 0.2) * LOOK * size, 0)
	add_child(mood)
	mood_shapes = {
		"ball": Ink.part(mood, Ink.ball(0.03), Ink.MUSTARD, Vector3.ZERO, Vector3.ONE, CELL, -1.0, true),
		"spike": Ink.part(mood, Ink.cone(0.0, 0.08, 0.03, 4), Ink.TEAL, Vector3.ZERO, Vector3.ONE, CELL, -1.0, true),
		"block": Ink.part(mood, BoxMesh.new(), Ink.CORAL, Vector3.ZERO, Vector3.ONE * 0.05, CELL, -1.0, true),
		"drop": Ink.part(mood, Ink.cone(0.03, 0.07, 0.0, 8), Ink.NAVY, Vector3.ZERO, Vector3.ONE, CELL, -1.0, true),
	}
	sign.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	sign.no_depth_test = true
	sign.font_size = 40
	sign.pixel_size = 0.0022
	sign.modulate = Ink.NAVY
	sign.outline_modulate = Ink.CREAM
	sign.outline_size = 10
	sign.position = mood.position + Vector3(0, 0.12, 0)
	add_child(sign)


func _torus(inner: float, outer: float) -> TorusMesh:
	var m := TorusMesh.new()
	m.inner_radius = inner
	m.outer_radius = outer
	m.rings = 10
	m.ring_segments = 5
	return m


func show_state(s: Dictionary, first: bool) -> void:
	state = s
	target = Vector3(s.x, 0, -s.y)
	if first:
		position = target
		rotation.y = s.h
	if s.emote != "" and s.emote_t != emote_seen:
		emote = s.emote
		emote_age = 0.0
		emote_seen = s.emote_t


func _process(dt: float) -> void:
	if state.is_empty():
		return
	var before := position
	position = position.lerp(target, min(1.0, 12.0 * dt))
	rotation.y = lerp_angle(rotation.y, state.h, min(1.0, 10.0 * dt))
	speed = lerp(speed, before.distance_to(position) / max(dt, 1e-4), min(1.0, 6.0 * dt))
	stride += speed * dt * 45.0
	emote_age += dt
	var t := Time.get_ticks_msec() / 1000.0 + get_index()

	# posture: standing, sitting, asleep, afloat, or flat on its side
	var swimming: bool = state.swimming
	var low: bool = state.sat or state.asleep
	var walk: float = clamp(speed / 0.08, 0.0, 1.0) * calm
	var want_y := -0.05 * LOOK if swimming else (-0.035 * LOOK if low else 0.0)
	var want_roll := 1.45 if state.down else sin(stride) * 0.16 * walk + (sin(t * 1.7) * 0.05 * calm if swimming else 0.0)
	model.position.y = lerp(model.position.y, want_y + abs(sin(stride)) * 0.008 * walk, min(1.0, 8.0 * dt))
	model.rotation.x = lerp(model.rotation.x, want_roll, min(1.0, 8.0 * dt))
	for i in 2:
		legs[i].visible = not (swimming or low)
		legs[i].rotation.z = sin(stride + PI * i) * 0.6 * walk
	shadow.visible = not swimming
	hat.visible = state.hat

	# the head: what it is doing outranks what it is feeling, and breathing fills the gaps
	var pitch := sin(t * 0.9) * 0.04 * calm
	var yaw := sin(t * 0.37) * 0.25 * calm
	var roll := 0.0
	var hop := 0.0
	var squash := 0.0
	if state.asleep:
		pitch = 0.9; yaw = 1.3
	elif state.eating:
		pitch = 0.7 + sin(t * 9.0) * 0.5; yaw = 0.0
	elif emote_age < EMOTE_S:
		var u := emote_age / EMOTE_S
		var beat := sin(u * TAU * 3.0)
		match emote:
			"happy": hop = abs(beat) * 0.05; roll = beat * 0.3; squash = -abs(beat) * 0.12
			"playful": hop = abs(beat) * 0.07; model.rotation.y = u * TAU
			"scared": pitch = 0.5; squash = 0.25; yaw = sin(u * 60.0) * 0.15
			"angry": pitch = -0.2 + max(beat, 0.0) * 0.7; squash = -0.1
			"sad": pitch = 0.75; yaw = sin(u * TAU) * 0.2
			"lonely": pitch = -0.25; yaw = sin(u * TAU) * 1.0
			"bored": yaw = sin(u * TAU) * 0.8; roll = 0.25
			"hungry": pitch = 0.4 + max(beat, 0.0) * 0.5
			"thirsty": pitch = -0.5 * sin(u * PI); yaw = 0.0
			"sleepy": pitch = 0.5 * abs(sin(u * PI * 2.0)); roll = 0.15
			"curious": roll = sin(u * TAU) * 0.45; pitch = -0.1
			"proud": pitch = -0.45; squash = -0.15; yaw = sin(u * TAU) * 0.5
	if emote_age >= EMOTE_S or emote != "playful":
		model.rotation.y = lerp_angle(model.rotation.y, 0.0, min(1.0, 8.0 * dt))
	neck.rotation = neck.rotation.lerp(Vector3(roll, yaw, -pitch), min(1.0, 10.0 * dt))
	model.position.y += hop * LOOK * calm
	var s := model.scale.x
	model.scale = model.scale.lerp(Vector3(s, s * (1.0 - squash * calm), s), min(1.0, 12.0 * dt))

	# the sign overhead: a shape for the mood, a word for the emote, and z for sleep
	var shape: String = MOOD_SHAPES.get(state.mood, "")
	for name in mood_shapes:
		mood_shapes[name].visible = name == shape and state.strength > 0.25 and not state.asleep
	mood.scale = Vector3.ONE * (0.6 + 0.9 * state.strength) * LOOK
	mood.rotation.y += dt * (4.0 if shape == "block" else 1.0) * calm
	mood.position.x = sin(t * 40.0) * 0.006 * calm if shape == "spike" else 0.0
	sign.text = "z" if state.asleep else (emote if emote_age < EMOTE_S + 0.6 else "")
