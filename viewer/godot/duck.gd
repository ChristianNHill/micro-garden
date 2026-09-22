# One duck: the real microduck, simplified (build_robot.py writes robot.json from Pollen Robotics' meshes) and
# built as the robot is, every body on its own hinge under its parent. So it is posed by joint angles: the
# simulated robot's own, live, when the garden sends them, and otherwise the robot's own motions as recorded
# from the simulator (clips.json: its sit, its walk, its kick, its roll, its fall and its getting up). It faces +x and stands on y = 0. Personality shows in its proportions, as far
# as a robot's can: a big appetite is a wide one, a timid duck is small, a vain one has a big head, and the grey
# plastic takes the duck's own colour.
# It is told where it is and how it feels (`show_state`), and everything else here is how that looks.
extends Node3D

const Ink := preload("res://ink.gd")
const Hats := preload("res://hats.gd")
const CELL := 5.0  # a finer screen than the ground's: at 9 px a duck this small came out spotted like a dalmatian
const LOOK := 1.9  # drawn a little larger than life, so a duck reads from across the garden
const EMOTE_S := 2.8
const RIBBONS := [Ink.CORAL, Ink.TEAL, Ink.MUSTARD, Color("7d6bd0"), Color("e58ac0")]
const DOES := {"cry": "cries", "dance": "dances", "singdance": "sings and dances", "stomp": "stomps", "yawn": "yawns", "splash": "splashes", "sing": "sings", "cower": "cowers"}  # a signature is what it does
const MOOD_SHAPES := {"joy": "ball", "fear": "spike", "anger": "block", "sorrow": "drop"}

var model := Node3D.new()  # everything that waddles, sits and falls over; the shadow and the signs do not
var rig := Node3D.new()  # the robot in MuJoCo's own frame (z up), turned once to stand in Godot's
var trunk := Node3D.new()
var hinges := {}  # joint name -> [the node it turns, its axis]
var frames := {}  # body name -> its node
var head: Node3D  # the body the hat sits on
var hat: Node3D  # whichever hat it wears, made from its style number
var hat_style := -1
var shadow: MeshInstance3D
var mood := Node3D.new()
var mood_shapes := {}
var sign := Label3D.new()
var tears: Array[MeshInstance3D] = []  # two, falling from a crying duck
var zs: Array[Label3D] = []  # a sleeper's z's, streaming up from its head: zzZZ
var ring: MeshInstance3D

var state := {}
var target := Vector3.ZERO
var speed := 0.0
var emote := ""
var emote_age := 99.0
var emote_seen := -1.0
var calm := 1.0  # 0.5 under reduced motion
var sway := 0.0  # a lean an emote asks for, which the next frame's posture takes up (set, never added to the tilt)
var lifted := false  # the player's hand has it: up off the grass, legs dangling
var heading_was := 0.0
var clip := "stand"  # which recorded motion is playing, and how far into it
var clip_t := 0.0
var hop := 0.0  # and a lift, the same way: added to the height each frame, a hop became a hover
const WIRE := ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
	"head_yaw", "head_roll", "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]  # robotd's joint order
# Motions played once through at this rate, and finished before the next begins (a knock-down interrupts any).
# The 2D body gets a duck up, sat or walking the moment it says so, where a robot takes seconds, so these hurry.
const ONCE := {"sit_down": 2.0, "stand_up": 3.0, "kick": 2.0, "roll": 1.3, "get_up": 2.5, "go_limp": 1.0}
static var clips := {}  # clips.json: the robot's own motions, recorded
static var robot := {}  # robot.json with its meshes made, built once and shared by every duck


static func robot_data() -> Dictionary:
	if robot.is_empty():
		robot = JSON.parse_string(FileAccess.get_file_as_string("res://robot.json"))
		for body in robot.bodies:
			for p in body.parts:
				var st := SurfaceTool.new()
				st.begin(Mesh.PRIMITIVE_TRIANGLES)
				for k in range(0, p.v.size(), 3):
					st.add_vertex(Vector3(p.v[k], p.v[k + 1], p.v[k + 2]) * robot.unit)
				for index in p.i:
					st.add_index(int(index))
				st.generate_normals()  # smooth ones, for the outline to grow along; the print shader shades flat
				p["mesh"] = st.commit()
	return robot


func build(index: int, knobs: Dictionary) -> void:
	var k := func(name: String) -> float: return float(knobs.get(name, 0.5))
	var girth: float = 0.85 + 0.3 * k.call("appetite")
	var size: float = 0.9 + 0.25 * k.call("aggressiveness") - 0.2 * k.call("timidity")
	# the servos are dark on the real robot; drawn dark they turned a small duck into a scribble, so they are
	# grey here and the links between them carry the duck's own colour
	var inks := {"cream": Ink.CREAM, "navy": Ink.STONE, "coral": Ink.CORAL, "mustard": Ink.MUSTARD,
		"stone": RIBBONS[index % RIBBONS.size()]}

	add_child(model)
	model.scale = Vector3(1.0, 1.0, girth) * LOOK * size
	var data := robot_data()
	rig.rotation.x = -PI / 2  # MuJoCo's z is up and its y is to the left; Godot's y is up and its z to the right
	rig.position.y = data.stand_z
	model.add_child(rig)
	rig.add_child(trunk)
	for body in data.bodies:
		var node := Node3D.new()
		if body.parent == "world":
			trunk.add_child(node)  # the trunk: where it is and how it leans is set on `trunk` each frame
		else:
			node.position = Vector3(body.pos[0], body.pos[1], body.pos[2])
			node.quaternion = Quaternion(body.quat[1], body.quat[2], body.quat[3], body.quat[0])
			frames[body.parent].add_child(node)
		var turns := Node3D.new()  # the hinge: everything of this body, and every body after it, turns here
		node.add_child(turns)
		frames[body.name] = turns
		if body.joint != null:
			hinges[body.joint.name] = [turns, Vector3(body.joint.axis[0], body.joint.axis[1], body.joint.axis[2]).normalized()]
		for part in body.parts:
			var shell: bool = part.ink == "cream"  # the line goes round the shells; round every servo it is a blot
			var piece := Ink.part(turns, part.mesh, inks[part.ink], Vector3.ZERO, Vector3.ONE, CELL, -1.0, shell)
			piece.material_override.set_shader_parameter("lift", 0.25)  # a duck stays bright on its shaded side
			if shell:
				(piece.material_override.next_pass as ShaderMaterial).set_shader_parameter("grow", 0.0025)
	head = frames[data.hat.body]
	frames["neck_pitch"].scale = Vector3.ONE * (0.9 + 0.3 * k.call("vanity"))  # the head and all that rides on it

	shadow = Ink.part(self, Ink.cone(0.1 * LOOK * size * girth, 0.001, 0.1 * LOOK * size * girth, 12), Ink.GRASS, Vector3(0, 0.003, 0), Vector3.ONE, 7.0, 0.35)
	ring = Ink.part(self, _torus(0.17 * LOOK, 0.19 * LOOK), Ink.CORAL, Vector3(0, 0.004, 0), Vector3(1, 0.2, 1), 7.0, 1.0)
	ring.visible = false

	mood.position = Vector3(0, 0.46 * LOOK * size, 0)  # over the head
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
	for _tear in 2:
		var tear := Ink.part(self, Ink.ball(0.012, 6), Color("4f9fe0"), Vector3.ZERO, Vector3(1.0, 1.5, 1.0), CELL, 0.9)
		tear.visible = false
		tears.append(tear)
	for _z in 4:
		var z := Label3D.new()
		z.billboard = BaseMaterial3D.BILLBOARD_ENABLED
		z.no_depth_test = true
		z.font_size = 64
		z.outline_size = 14
		z.outline_modulate = Ink.CREAM
		z.visible = false
		add_child(z)
		zs.append(z)


func _play(want: String, dt: float) -> Array:
	# Advance the playing clip towards what the duck is doing and return its pose now, [joints, height, lean].
	if clips.is_empty():
		clips = JSON.parse_string(FileAccess.get_file_as_string("res://clips.json"))
	var hz: float = clips.hz
	var frames_now: Array = clips.clips[clip].frames
	var length: float = (frames_now.size() - 1) / hz
	var finishing: bool = ONCE.has(clip) and clip_t < length and want != "go_limp"
	if want == "go_limp":
		# Down for a set time in the 2D body: it falls, lies, and is getting up over the last of it, so that it
		# stands as it is freed. Getting up is placed by the time left, not played forward.
		var rise: float = (clips.clips["get_up"].frames.size() - 1) / hz / ONCE["get_up"]
		var left: float = state.get("down_left", 0.0)
		if left < rise:
			clip = "get_up"
			clip_t = max(((clips.clips["get_up"].frames.size() - 1) / hz) - left * ONCE["get_up"], 0.0)
			return _pose_at(clip, clip_t)
	if not finishing:
		var sat: bool = clip in ["sitting", "sit_down"]
		var next := want
		if want == "sitting" and not sat:
			next = "sit_down"
		elif sat and not (want in ["sitting", "go_limp"]):
			next = "stand_up"
		elif clip == "go_limp" and want != "go_limp":
			next = "get_up"  # freed early, which the robot body can be: it still has to get up
		if next != clip:
			clip = next
			clip_t = 0.0
			frames_now = clips.clips[clip].frames
			length = (frames_now.size() - 1) / hz
	var rate: float = ONCE.get(clip, clamp(speed / 0.12, 0.6, 2.5) if clip == "walk" else 1.0)
	clip_t += dt * rate * (0.5 + 0.5 * calm)
	if clips.clips[clip].loop:
		clip_t = fmod(clip_t, length)
	else:
		clip_t = min(clip_t, length)  # and held there: a duck that is down stays as it fell
	return _pose_at(clip, clip_t)


func _pose_at(which: String, seconds: float) -> Array:
	var frames_now: Array = clips.clips[which].frames
	var at: float = clamp(seconds * float(clips.hz), 0.0, frames_now.size() - 1.0)
	var a: Array = frames_now[int(at)]
	var b: Array = frames_now[min(int(at) + 1, frames_now.size() - 1)]
	var u: float = at - int(at)
	var joints := []
	for k in a[0].size():
		joints.append(lerp(float(a[0][k]), float(b[0][k]), u))
	var qa := Quaternion(a[2][1], a[2][2], a[2][3], a[2][0]).normalized()  # written to three places, so not quite unit
	var qb := Quaternion(b[2][1], b[2][2], b[2][3], b[2][0]).normalized()
	var q := qa.slerp(qb, u)
	return [joints, lerp(float(a[1]), float(b[1]), u), [q.w, q.x, q.y, q.z]]


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
	emote_age += dt
	var t := Time.get_ticks_msec() / 1000.0 + get_index()
	var real: bool = state.has("joints")  # the simulated robot's own joints: then nothing here is by rule
	var angle := _real_pose(dt) if real else _clip_pose(dt, t)
	heading_was = rotation.y
	for leg in ["yaw2roll", "bearing_roll"]:  # where each leg joins the trunk: under water they are not seen
		frames[leg].visible = real or not state.swimming
	shadow.visible = not state.swimming
	_wear()
	if not real:
		_nod(angle, t)
	for name in hinges:  # and the pose is set: every hinge eased to its angle about its own axis
		var hinge: Array = hinges[name]
		var node: Node3D = hinge[0]
		node.quaternion = node.quaternion.slerp(Quaternion(hinge[1], angle.get(name, 0.0)), min(1.0, (20.0 if real else 9.0) * dt))
	_act_out(dt)
	_signs(dt, t)


func _stand() -> Dictionary:
	var angle := {}
	for name in robot_data().stand:
		angle[name] = float(robot.stand[name])
	return angle


func _real_pose(dt: float) -> Dictionary:
	# The simulated robot's own joints, trunk height and lean, eased to.
	var angle := _stand()
	for k in WIRE.size():
		angle[WIRE[k]] = float(state.joints[k])
	trunk.quaternion = trunk.quaternion.slerp(Quaternion(state.tilt[1], state.tilt[2], state.tilt[3], state.tilt[0]).normalized(), min(1.0, 12.0 * dt))
	rig.position.y = lerp(rig.position.y, float(state.z), min(1.0, 12.0 * dt))
	model.rotation.x = 0.0
	model.position.y = 0.0
	return angle


func _wanted_clip(dt: float) -> String:
	# Which of the robot's recorded motions fits what this duck is doing.
	var moved_to_act: bool = emote_age < EMOTE_S and emote in ["dance", "singdance", "happy", "playful", "stomp"]  # on its feet for these
	var low: bool = (state.sat and not moved_to_act) or state.asleep
	var yaw_rate: float = angle_difference(heading_was, rotation.y) / max(dt, 1e-4)
	if state.down:
		return "go_limp"
	if low or state.swimming:
		return "sitting"
	if state.get("kicking", false):
		return "kick"
	if state.eating:
		return "peck"
	if emote == "playful" and emote_age < 0.4:
		return "roll"
	if speed > 0.015:
		return "walk"
	if abs(yaw_rate) > 0.4:
		return "turn_left" if yaw_rate > 0.0 else "turn_right"
	return "stand"


func _clip_pose(dt: float, t: float) -> Dictionary:
	# The robot's own motions, recorded from the simulator (record_clips.py) and played for whatever this
	# duck is doing: it sits as a microduck sits, walks its walk, and gets up the way one gets up.
	var angle := _stand()
	var pose := _play(_wanted_clip(dt), dt)
	for k in WIRE.size():
		angle[WIRE[k]] = float(pose[0][k])
	rig.position.y = lerp(rig.position.y, float(pose[1]), min(1.0, 14.0 * dt))
	trunk.quaternion = trunk.quaternion.slerp(Quaternion(pose[2][1], pose[2][2], pose[2][3], pose[2][0]).normalized(), min(1.0, 14.0 * dt))
	if emote in ["dance", "singdance"] and emote_age < EMOTE_S:  # no robot has a dance: steps in place, by rule
		var step := sin(emote_age * TAU / 0.7) * 0.3
		for side in ["left", "right"]:
			angle[side + "_hip_pitch"] = angle.get(side + "_hip_pitch", 0.0) + step
			angle[side + "_ankle"] = angle.get(side + "_ankle", 0.0) - step
	var swimming: bool = state.swimming
	var afloat := -0.05 * model.scale.y if swimming else (0.32 if lifted else 0.0)  # the pond comes up to its trunk; a hand lifts it clear
	model.position.y = lerp(model.position.y, afloat + hop * LOOK * calm, min(1.0, 14.0 * dt))
	model.rotation.x = lerp(model.rotation.x, (sin(t * 1.7) * 0.05 * calm if swimming else 0.0) + sway * calm, min(1.0, 8.0 * dt))
	return angle


func _wear() -> void:
	# The hat it wears, remade only when which hat changes.
	var wearing: int = state.hat_style if state.hat else -1
	if wearing == hat_style:
		return
	if hat != null:
		hat.queue_free()
		hat = null
	if wearing >= 0:
		hat = Hats.make(wearing)
		var seat: Dictionary = robot_data().hat  # the crown of the head shell, and which way is up there, in the head's frame
		var up := Vector3(seat.up[0], seat.up[1], seat.up[2])
		var forward := Vector3(seat.forward[0], seat.forward[1], seat.forward[2])
		hat.transform = Transform3D(Basis(forward, up, forward.cross(up)).scaled(Vector3.ONE * 1.25), Vector3(seat.at[0], seat.at[1], seat.at[2]))
		head.add_child(hat)
	hat_style = wearing


func _nod(angle: Dictionary, t: float) -> void:
	# The head does what the body was told (`state.head`: neck_pitch, head_pitch, head_yaw, head_roll, offsets
	# on the robot's own joints), so an emote here is the emote the robot makes. Sleeping and eating have no
	# head commands of their own yet and are posed here; breathing fills the gaps.
	var told: Array = state.get("head", [0.0, 0.0, 0.0, 0.0])
	var nod: Array = [told[0], told[1] + sin(t * 0.9) * 0.03 * calm, told[2] + sin(t * 0.37) * 0.1 * calm, told[3]]
	if state.asleep:
		nod = [0.45, 0.6, 0.9, 0.0]
	elif state.eating:
		nod = [0.45 + sin(t * 9.0) * 0.2, 0.5, 0.0, 0.0]
	var names := ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
	for k in 4:
		angle[names[k]] = angle.get(names[k], 0.0) + nod[k]


func _act_out(dt: float) -> void:
	# The rest of the body joins in a little, for the feelings that would move more than a head.
	var squash := 0.0
	sway = 0.0
	hop = 0.0
	if emote_age < EMOTE_S and not state.asleep:
		var beat := sin(emote_age / EMOTE_S * TAU * 3.0)
		match emote:
			"happy": hop = abs(beat) * 0.04; squash = -abs(beat) * 0.08
			"playful": hop = abs(beat) * 0.06; model.rotation.y = emote_age / EMOTE_S * TAU
			"scared": squash = 0.18
			"proud": squash = -0.1
			"stomp": hop = max(beat, 0.0) * 0.025; squash = max(-beat, 0.0) * 0.1
			"yawn": squash = -0.07 * sin(emote_age / EMOTE_S * PI)
			"splash": hop = abs(beat) * 0.03; squash = -abs(beat) * 0.06
			"sing": sway = beat * 0.08
			"cower": squash = 0.24
			"dance", "singdance":  # on the beat: down and up, a sway, a twist from side to side
				var bar := sin(emote_age * TAU / 0.7)
				hop = abs(bar) * 0.02
				squash = max(-bar, 0.0) * 0.1
				sway = sin(emote_age * TAU / 1.4) * 0.12
				model.rotation.y = sin(emote_age * TAU / 1.4) * 0.5
	if emote_age >= EMOTE_S or not (emote in ["playful", "dance", "singdance"]):
		model.rotation.y = lerp_angle(model.rotation.y, 0.0, min(1.0, 8.0 * dt))
	var s := model.scale.x
	model.scale = model.scale.lerp(Vector3(s, s * (1.0 - squash * calm), model.scale.z), min(1.0, 12.0 * dt))


func _signs(dt: float, t: float) -> void:
	# The sign overhead: a shape for the mood, a word for the emote, tears, and z for sleep.
	var shape: String = MOOD_SHAPES.get(state.mood, "")
	for name in mood_shapes:
		mood_shapes[name].visible = name == shape and state.strength > 0.25 and not state.asleep
	mood.scale = Vector3.ONE * (0.6 + 0.9 * state.strength) * LOOK
	mood.rotation.y += dt * (4.0 if shape == "block" else 1.0) * calm
	mood.position.x = sin(t * 40.0) * 0.006 * calm if shape == "spike" else 0.0
	sign.text = "" if state.asleep else (DOES.get(emote, emote) if emote_age < EMOTE_S + 0.6 else "")
	for k in tears.size():  # a tear from each side of the head, again and again
		tears[k].visible = state.get("crying", false)
		var fall := fmod(t * 1.4 + 0.5 * k, 1.0)
		tears[k].position = Vector3(0.05 * LOOK, (0.26 - 0.22 * fall) * LOOK, (0.06 if k == 0 else -0.06) * LOOK)
	for k in zs.size():  # each z rises from the head, drifts, grows from z to Z, and fades: a stream of them
		var z := zs[k]
		z.visible = state.asleep
		if state.asleep:
			var u := fmod(t * 0.3 * (0.5 + 0.5 * calm) + float(k) / zs.size(), 1.0)
			z.text = "z" if u < 0.45 else "Z"
			z.position = Vector3(0.0, 0.2 * LOOK, 0.0) + Vector3(0.04 + 0.12 * u + sin(u * 7.0) * 0.02, 0.3 * u, 0.0) * LOOK
			z.pixel_size = 0.0012 + 0.0026 * u
			var there: float = min(1.0, u * 6.0) * sqrt(1.0 - u)  # in quickly, out slowly
			z.modulate = Color(Color.WHITE, there)
			z.outline_modulate = Color(Ink.NAVY, there)
