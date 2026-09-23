# Draws the garden from the JSON snapshots viewer/snapshot.py sends over UDP, one a step. Player actions go
# back as the garden's control calls. Godot only draws; it decides nothing about a duck. Start the garden with
# --godot and then this project, in either order.
#
# After `--`:
#   --orbit=yaw,pitch,distance  where the camera starts       --ride=N   start riding duck N, --select=N select it
#   --reduced-motion            a still camera, half the animation       --mute     no sound
#   --light=0.1                 force this daylight level; --asleep=N draws duck N asleep
#   --music=DIR                 the folder of mp3s the music box plays (default ~/.cache/micro-garden/music)
#   --keys=MBT                  press these keys after three seconds, to check they reach the garden
#   --port=N                    listen there, for a garden started with MICRO_GARDEN_PORT=N
#   --shot=file.png             save the window after --shot-after=S seconds (6 by default)
#   --record=DIR                save numbered pngs to DIR, --record-fps=N a second (10), from
#                               --record-after=S seconds (6) for --record-for=S (20)
#   --feed-at=x,y               drop food there once snapshots arrive, and --quit-after=S: both for Gate 13
extends Node3D

const Ink := preload("res://ink.gd")
const Duck := preload("res://duck.gd")
const Scenery := preload("res://scenery.gd")
const Hats := preload("res://hats.gd")
const BrainView := preload("res://brainview.gd")
const MusicBox := preload("res://musicbox.gd")
const SNAPSHOT_PORT := 7650  # actions go back on port + 1; --port moves both
# The controls, by what you are doing, as [heading, [[what you press, what happens] ...]]. _build_menu
# lays them out, so no line here needs spacing by hand.
const CONTROLS := [
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
		["T", "leave a hat at the mouse, for whoever wants it"],
		["B", "leave a ball at the mouse"],
		["D", "set the drum down at the mouse, or pick it back up"],
		["I", "leave an instrument at the mouse, one of ten, picked at random"],
		["M", "set the music box down at the mouse, or pick it back up"],
		["tap the music box", "turn it up, turn it down, turn it off"],
		["C", "clap, which startles every duck"],
	]],
	["the view", [
		["drag", "turn the camera. Right-drag instead while your hand is out"],
		["scroll", "zoom in and out"],
		["/", "close this"],
	]],
]
const FEELS := {"joy": "happy", "fear": "scared", "anger": "angry", "sorrow": "sad"}  # mood name -> card word
const HELP_HINT := "/  controls"
const HELP_RIDING := "W A S D  steer      O  hide what it sees      Tab  get off"
const PAPER_ROUND := 10  # every panel is the same cream paper: corner radius and padding
const PAPER_PAD := 12
const BARS_W := 302.0  # the selected duck's readout, under its card
const STINK_PUFFS := 9  # puffs rising off a stink patch, three to a strand
const DUCK_TALL := 0.42  # how tall a duck is drawn, for taps that land on the duck itself
const KICKED_M := 0.3  # a fruit that jumps further than this in one step was kicked, not carried
const KICKED_S := 0.7  # how long a kicked fruit takes to bounce to where it lands
# Each instrument's sound (body/stub2d/stub.py INSTRUMENTS, in order): its lowest note in Hz, how long a note
# rings, and which of _note's voices plays it.
const INSTRUMENT_TONES := [[880.0, 0.4, "struck"], [0.0, 0.14, "shaken"], [0.0, 0.3, "jingle"], [2400.0, 1.0, "ring"],
	[523.0, 0.6, "struck"], [196.0, 0.9, "plucked"], [349.0, 0.4, "brass"], [660.0, 1.0, "ring"], [784.0, 0.45, "blown"],
	[392.0, 0.9, "harp"]]
const PITCH := Vector2(0.35, 1.25)  # camera pitch range, radians above the horizon
const DRIFT := 0.12  # radians the idle camera sways either way

var udp := PacketPeerUDP.new()
var out := PacketPeerUDP.new()
var snap := {}
var got := 0
var ducks: Array = []
var selected := -1
var flags: Array = []  # the cloths, which turn with the wind
var falls: Array = []  # waterfall sheets
var reeds: Array = []  # the reeds round the pond, which lean with the wind
var props := {}  # kind -> nodes drawn for the items of that kind

var cam := Camera3D.new()
var orbit := Vector3(0.75, 0.7, 8.8)  # yaw, pitch, distance
var centre := Vector3.INF  # camera orbit centre
var glove := Node3D.new()  # the player's hand
var hand_mode := false  # H: the cursor is the glove and the left button grips; off, the left button turns the camera
var mouse_inside := true
var gripping := false
var pressed_at := 0.0
var trail: Array = []  # recent [seconds, garden xy] of the hand, for throw speed
var dragging := false
var dragged := 0.0
var still := 0.0  # seconds since the player last moved the camera
var calm := 1.0

var possessing := false  # riding: the camera is on the selected duck and W A S D steer it
var voice := AudioStreamPlayer.new()
var heard := -1.0  # garden time of the last quack played
var water := AudioStreamPlayer.new()  # pond noise, quieter at night
var water_level := 0.0
var water_t := 0.0

var music := Node.new()
var said := ""  # a line from this window, shown with the toasts for a moment
var said_until := 0.0
var bars := Control.new()  # the selected duck's needs, moods and wants
var brain := Control.new()
var watching := -2  # whose brain the garden was last asked for
var asked := 0.0
var overlay := Control.new()  # ride view: retinas and descending-neuron bars; O hides it
var card := PanelContainer.new()  # the selected duck's card: its name, how it is, and what it has learned
var card_name := Label.new()
var card_trait := Label.new()
var card_body := Label.new()
var card_skills := Label.new()
var help := Label.new()  # the one-line hint in the corner, and the riding line
var menu := Control.new()  # the controls, built by _build_menu
var paper := StyleBoxFlat.new()  # the cream panel every readout is drawn on
var help_on := false  # the hint until / opens the menu
var toasts := Label.new()
var args := {}


func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		var kv := a.trim_prefix("--").split("=")
		args[kv[0]] = kv[1] if kv.size() > 1 else "1"
	calm = 0.5 if args.has("reduced-motion") else 1.0
	if args.has("orbit"):  # yaw,pitch,distance: where the camera starts
		var o: PackedStringArray = args["orbit"].split(",")
		orbit = Vector3(float(o[0]), float(o[1]), float(o[2]))
	var port := int(args.get("port", str(SNAPSHOT_PORT)))
	udp.bind(port, "127.0.0.1")
	out.set_dest_address("127.0.0.1", port + 1)
	cam.fov = 30.0
	add_child(cam)
	var ui := CanvasLayer.new()
	add_child(ui)
	paper = _paper(Color(Ink.CREAM, 0.88), PAPER_ROUND, PAPER_PAD)
	for label in [toasts, help]:  # every panel is the same cream paper
		label.add_theme_color_override("font_color", Ink.NAVY)
		label.add_theme_font_size_override("font_size", 16)
		label.add_theme_constant_override("line_spacing", 5)
		label.add_theme_stylebox_override("normal", paper)
		ui.add_child(label)
	_build_card(ui)
	bars.mouse_filter = Control.MOUSE_FILTER_IGNORE
	bars.draw.connect(_draw_bars)
	ui.add_child(bars)
	brain.set_script(BrainView)
	ui.add_child(brain)
	brain.visible = false
	overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
	overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	overlay.draw.connect(_draw_overlay)
	ui.add_child(overlay)
	help.add_theme_font_size_override("font_size", 13)
	help.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT, Control.PRESET_MODE_MINSIZE, 24)
	help.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	help.grow_vertical = Control.GROW_DIRECTION_BEGIN
	# the news goes top centre, clear of the card and readout on the left, the brain on the right and the ride view
	toasts.set_anchors_and_offsets_preset(Control.PRESET_CENTER_TOP, Control.PRESET_MODE_MINSIZE, 20)
	toasts.grow_horizontal = Control.GROW_DIRECTION_BOTH
	toasts.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	_build_menu(ui)
	# a white glove with a cuff: a palm, four fingers and a thumb
	Ink.part(glove, Ink.ball(0.07, 8), Color.WHITE, Vector3.ZERO, Vector3(1.0, 0.55, 1.1), 5.0, -1.0, true).material_override.set_shader_parameter("lift", 0.4)
	for k in 4:
		Ink.part(glove, Ink.cone(0.018, 0.09, 0.015, 6), Color.WHITE, Vector3(0.075, -0.01, -0.05 + 0.033 * k), Vector3.ONE, 5.0, -1.0, true).rotation.z = -PI / 2 - 0.35
	Ink.part(glove, Ink.cone(0.02, 0.07, 0.016, 6), Color.WHITE, Vector3(0.01, -0.005, 0.085), Vector3.ONE, 5.0, -1.0, true).rotation.x = PI / 2 - 0.4
	Ink.part(glove, Ink.cone(0.055, 0.05, 0.05, 8), Ink.CORAL, Vector3(-0.075, 0, 0), Vector3.ONE, 5.0, -1.0, true).rotation.z = PI / 2
	glove.scale = Vector3.ONE * 1.6
	glove.visible = false
	add_child(glove)
	music.set_script(MusicBox)
	add_child(music)
	music.setup(args.get("music", OS.get_environment("HOME").path_join(".cache/micro-garden/music")))
	var gen := AudioStreamGenerator.new()
	gen.mix_rate = 22050.0
	gen.buffer_length = 1.0
	voice.stream = gen
	add_child(voice)
	voice.play()
	var pond_gen := AudioStreamGenerator.new()
	pond_gen.mix_rate = 22050.0
	pond_gen.buffer_length = 0.2
	water.stream = pond_gen
	water.volume_db = -26.0
	add_child(water)
	water.play()
	if args.has("mute"):
		AudioServer.set_bus_mute(0, true)
	var second := Timer.new()
	second.timeout.connect(_report)
	add_child(second)
	second.start(1.0)
	if args.has("shot"):
		get_tree().create_timer(float(args.get("shot-after", "6"))).timeout.connect(
			func() -> void: get_viewport().get_texture().get_image().save_png(args["shot"]))
	if args.has("record"):  # on the wall clock, so the recording plays at the garden's own pace
		var shots := Timer.new()
		var taken := [0]
		var last := int(float(args.get("record-for", "20")) * float(args.get("record-fps", "10")))
		shots.timeout.connect(func() -> void:
			get_viewport().get_texture().get_image().save_png(args["record"].path_join("%04d.png" % taken[0]))
			taken[0] += 1
			if taken[0] >= last:
				shots.stop())
		add_child(shots)
		get_tree().create_timer(float(args.get("record-after", "6"))).timeout.connect(
			func() -> void: shots.start(1.0 / float(args.get("record-fps", "10"))))
	if args.has("keys"):  # with the mouse in the middle of the window
		get_tree().create_timer(3.0).timeout.connect(func() -> void:
			get_viewport().warp_mouse(get_viewport().get_visible_rect().size / 2)
			for letter in args["keys"]:
				var e := InputEventKey.new()
				e.keycode = KEY_SLASH if letter == "/" else OS.find_keycode_from_string(letter)
				e.pressed = true
				Input.parse_input_event(e)
				print("pressed ", letter))
	if args.has("quit-after"):
		get_tree().create_timer(float(args["quit-after"])).timeout.connect(get_tree().quit)


func _notification(what: int) -> void:
	# The glove replaces the cursor only over the garden; outside or unfocused, the cursor comes back.
	if what == NOTIFICATION_WM_MOUSE_EXIT or what == NOTIFICATION_APPLICATION_FOCUS_OUT or what == NOTIFICATION_WM_WINDOW_FOCUS_OUT:
		mouse_inside = false
		if gripping:
			gripping = false
			_let_go()
	elif what == NOTIFICATION_WM_MOUSE_ENTER:
		mouse_inside = true
	_cursor()


func _cursor() -> void:
	Input.mouse_mode = Input.MOUSE_MODE_HIDDEN if hand_mode and mouse_inside and not possessing else Input.MOUSE_MODE_VISIBLE


func _report() -> void:
	print("snapshots hz=%d" % got)  # Gate 13 reads these
	got = 0


func _process(dt: float) -> void:
	_roll_kicked(dt)
	var latest := ""
	while udp.get_available_packet_count() > 0:
		latest = udp.get_packet().get_string_from_utf8()
		got += 1
	if latest != "":
		var parsed = JSON.parse_string(latest)
		if parsed is Dictionary:
			var first := snap.is_empty()
			snap = parsed
			if args.has("light"):
				snap.light = float(args["light"])
			if args.has("asleep"):
				snap.ducks[int(args["asleep"])].asleep = true
			if first:
				_build()
			_show(first)
	_camera(dt)
	_hand(dt)
	_pond_sound()


func _build() -> void:
	var size: float = snap.size
	if not args.has("orbit"):
		orbit = Vector3(2.2, 0.5, 1.7 * size)  # from the open front of the lawn
	var eye := Vector3(size / 2, 0, -size / 2) + Vector3(cos(orbit.x), 0, sin(orbit.x)) * orbit.z
	# a flag on the cliff rock nearest the viewer and another on the rocks across the garden, both
	# showing the breeze the ducks smell by
	for summit in Scenery.build(self, snap, falls, eye, reeds):
		flags.append(_raise_flag(summit))
	for i in snap.ducks.size():
		var d := Node3D.new()
		d.set_script(Duck)
		add_child(d)
		d.calm = calm
		d.build(i, snap.ducks[i].knobs)
		ducks.append(d)
	if args.has("select"):
		selected = int(args["select"])
	if args.has("ride"):
		selected = int(args["ride"])
		possessing = true
	if args.has("feed-at"):
		var xy: PackedStringArray = args["feed-at"].split(",")
		_act("garden.hand", {"x": float(xy[0]), "y": float(xy[1]), "feed": 3})


func _show(first: bool) -> void:
	RenderingServer.global_shader_parameter_set("daylight", snap.light)
	RenderingServer.set_default_clear_color(Scenery.SKY_NIGHT.lerp(Scenery.SKY_DAY, snap.light))
	var day: float = snap.day * TAU
	RenderingServer.global_shader_parameter_set("sun_dir", Vector3(cos(day), 0.35 + 0.65 * snap.light, 0.5 * sin(day)).normalized())
	for i in ducks.size():
		ducks[i].show_state(snap.ducks[i], first)
		ducks[i].ring.visible = i == selected and not possessing
		ducks[i].model.visible = not (possessing and i == selected)
	_show_props()
	_show_weather()
	_show_held()
	_ask_for_brain()
	music.set_on(snap.music != null)
	music.follow(snap.get("music_volume", 0.75))
	for q in snap.get("sounds", []):  # [t, duck, tag], oldest first
		if q[0] > heard:
			heard = q[0]
			_quack(int(q[1]), q[2])
	if possessing:
		_act("garden.wheel", {"duck": selected, "fwd": int(Input.is_key_pressed(KEY_W)) - int(Input.is_key_pressed(KEY_S)),
			"turn": int(Input.is_key_pressed(KEY_A)) - int(Input.is_key_pressed(KEY_D))})
	_show_words()


func _show_props() -> void:
	_props("food", snap.food, _fruit)
	_props("hats", snap.get("hats", []), func(n: Node3D, h: Array) -> void:
		var lying := Hats.make(int(h[2]))
		lying.scale = Vector3.ONE * 1.9  # ducks are drawn 1.9x life size, so hats are too
		n.add_child(lying))
	var played: Array = snap.get("instruments", [])
	_props("instruments", played, _instrument)
	for i in played.size():  # the oldest goes when a fifth is put down, so the count can stay and the kinds change
		if props.instruments[i].get_meta("kind", -1) != int(played[i][2]):
			for old in props.instruments[i].get_children():
				old.queue_free()
			_instrument(props.instruments[i], played[i])
	var balls: Array = snap.get("balls", [])
	_props("balls", balls, _ball)
	for i in balls.size():  # the oldest ball goes when a fourth is put down, so the count can stay and the kinds change
		var style: int = int(balls[i][2]) if balls[i].size() > 2 else 0
		if props.balls[i].get_meta("style", -1) != style:
			for old in props.balls[i].get_children():
				old.queue_free()
			_ball(props.balls[i], balls[i])
	for n in props.get("balls", []):  # roll by the distance moved
		var ball: Node3D = n.get_child(0)
		var moved: Vector3 = n.position - n.get_meta("was", n.position)
		n.set_meta("was", n.position)
		if moved.length() > 1e-5:
			ball.rotate(Vector3(moved.z, 0, -moved.x).normalized(), moved.length() / (0.06 * 1.9))
	_props("drum", [snap.drum] if snap.get("drum") != null else [], func(n: Node3D, f: Array) -> void:
		Ink.part(n, Ink.cone(0.13, 0.16, 0.11, 10), Ink.CORAL, Vector3(0, 0.11, 0), Vector3.ONE, 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.3)
		Ink.part(n, Ink.cone(0.135, 0.02, 0.135, 10), Color("f6f1e6"), Vector3(0, 0.2, 0), Vector3.ONE, 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.4)
		for k in 3:
			Ink.part(n, Ink.cone(0.012, 0.06, 0.012, 4), Scenery.WOOD, Vector3(cos(TAU * k / 3.0) * 0.09, 0.03, sin(TAU * k / 3.0) * 0.09), Vector3.ONE, 6.0))
	_props("danger", snap.danger, func(n: Node3D, f: Array) -> void:
		Ink.part(n, Ink.cone(0.22, 0.002, 0.22, 10), Ink.MUSTARD, Vector3(0, 0.003, 0), Vector3.ONE, 5.0, 0.3)
		for k in STINK_PUFFS:  # puffs of stink, drifting up off the patch and thinning out
			Ink.part(n, Ink.ball(0.035, 7), Ink.MUSTARD, Vector3.ZERO, Vector3.ONE, 4.0, 0.7))
	var gust = snap.get("wind")  # the stink goes where the wind takes it, like the smell itself
	var drift := Vector3(gust[0], 0.0, -gust[1]) * 0.35 if gust != null else Vector3.ZERO
	for n in props.get("danger", []):
		for k in range(1, n.get_child_count()):  # each puff rises, curls aside and thins away
			var puff: Node3D = n.get_child(k)
			var strand := float((k - 1) % 3)
			var up: float = fposmod(Time.get_ticks_msec() / 2600.0 * calm + float(k) / STINK_PUFFS, 1.0)
			var swing := sin(up * TAU + strand * 2.1) * 0.07
			var a := TAU * strand / 3.0 + 0.4
			puff.position = Vector3(cos(a) * 0.08 + swing, 0.04 + up * 0.55, sin(a) * 0.08 + swing * 0.6) + drift * up * up
			puff.scale = Vector3.ONE * (1.0 - 0.75 * up)
	_props("music", [snap.music] if snap.music != null else [], func(n: Node3D, f: Array) -> void:
		Ink.part(n, BoxMesh.new(), Ink.CORAL, Vector3(0, 0.05, 0), Vector3(0.14, 0.1, 0.14), 7.0, -1.0, true)
		Ink.part(n, Ink.cone(0.02, 0.16, 0.1, 8), Ink.MUSTARD, Vector3(0.03, 0.18, 0), Vector3.ONE, 7.0, -1.0, true).rotation.z = -0.5)
	for n in props.get("music", []):
		n.scale = Vector3.ONE * (1.0 + 0.06 * calm * sin(Time.get_ticks_msec() / 90.0))


func _show_weather() -> void:
	var wind = snap.get("wind")
	var blowing: bool = wind != null and Vector2(wind[0], wind[1]).length() > 0.05
	for i in flags.size():
		var flag: Node3D = flags[i]
		var flutter := sin(Time.get_ticks_msec() / 160.0 + i * 0.9) * 0.14 * calm  # the two are not in step
		flag.rotation.y = lerp_angle(flag.rotation.y, (atan2(wind[1], wind[0]) if blowing else flag.rotation.y) + flutter, 0.08)
		flag.rotation.z = lerp(flag.rotation.z, 0.0 if blowing else -1.35, 0.05)  # hangs with no wind
	for i in falls.size():
		falls[i].scale.z = 0.5 * (1.0 + 0.12 * calm * sin(Time.get_ticks_msec() / 130.0 + i * 1.7))
	# the reeds lean the way the wind goes and rustle on top of it, so the breeze shows at ground level too
	var strength: float = clamp(Vector2(wind[0], wind[1]).length(), 0.0, 1.0) if blowing else 0.0
	var toward := Vector3(wind[0], 0.0, -wind[1]).normalized() if blowing else Vector3.FORWARD
	var axis := Vector3(toward.z, 0.0, -toward.x).normalized()
	var beat := Time.get_ticks_msec() / 520.0
	for i in reeds.size():
		var reed: Node3D = reeds[i]
		var lean: float = calm * (0.2 * strength + 0.04) * (0.75 + 0.25 * sin(beat + i * 0.8))
		reed.quaternion = reed.quaternion.slerp(Quaternion(axis, lean), 0.08)


func _show_held() -> void:
	# lift whatever the hand holds
	var held = snap.get("held")
	for kind in ["balls", "hats", "food", "music", "drum", "instruments"]:
		var nodes: Array = props.get(kind, [])
		for k in nodes.size():
			var up: bool = held != null and held[0] + ("s" if held[0] in ["ball", "hat", "instrument"] else "") == kind and int(held[1]) == k
			nodes[k].position.y = lerp(nodes[k].position.y, 0.3 if up else 0.0, 0.3)
	for i in ducks.size():
		ducks[i].lifted = held != null and held[0] == "duck" and int(held[1]) == i


func _ask_for_brain() -> void:
	# Ask for the selected duck's brain, again every second in case the garden restarted.
	asked += 0.02
	if watching != selected or asked > 1.0:
		watching = selected
		asked = 0.0
		_act("garden.watch", {"duck": selected})
	brain.visible = selected >= 0 and snap.has("brain") and snap.brain.duck == selected
	if brain.visible:
		brain.show_brain(snap.brain, snap.ducks[selected].name)
		brain.position = Vector2(get_viewport().get_visible_rect().size.x - brain.size.x - 24.0, 20.0)


func _show_words() -> void:
	bars.position = card.position + Vector2(0, card.size.y + 8.0)
	bars.queue_redraw()
	overlay.queue_redraw()
	var lines: Array = snap.toasts.duplicate()
	if Time.get_ticks_msec() / 1000.0 < said_until:
		lines.append(said)
	elif snap.music != null and music.title != "" and music.gain > 0.0:
		lines.append("♪ " + music.title)
	toasts.text = "\n".join(lines)
	toasts.visible = not lines.is_empty()
	card.visible = selected >= 0
	var open: bool = help_on and not possessing
	menu.visible = open
	help.visible = not open
	help.text = HELP_RIDING if possessing else HELP_HINT
	for other in [card, bars, toasts, brain]:  # the open menu has the window to itself
		other.modulate.a = 0.0 if open else 1.0
	if selected >= 0:
		_fill_card(snap.ducks[selected])


func _raise_flag(summit: Vector3) -> Node3D:
	# A pole on a cliff rock, and a white flag with a duck on it. The cloth turns with the wind, so the
	# duck is printed on both faces.
	Ink.part(self, Ink.cone(0.035, 1.5, 0.028, 5), Scenery.WOOD, summit + Vector3(0, 0.7, 0), Vector3.ONE, 7.0)
	var cloth := Node3D.new()
	cloth.position = summit + Vector3(0, 1.22, 0)
	add_child(cloth)
	Ink.part(cloth, BoxMesh.new(), Color.WHITE, Vector3(0.42, 0, 0), Vector3(0.84, 0.46, 0.02), 7.0, 1.0, true)  # no halftone screen, so it stays white
	for face in [0.016, -0.016]:
		var body := Ink.part(cloth, Ink.ball(0.1, 9), Ink.NAVY, Vector3(0.4, -0.03, face), Vector3(1.05, 0.8, 0.12), 6.0, 0.75)
		body.rotation.z = 0.15
		Ink.part(cloth, Ink.ball(0.062, 9), Ink.NAVY, Vector3(0.49, 0.09, face), Vector3(1, 1, 0.12), 6.0, 0.75)
		Ink.part(cloth, Ink.cone(0.03, 0.06, 0.0, 5), Ink.MUSTARD, Vector3(0.565, 0.085, face), Vector3(1, 1, 0.3), 6.0, 0.9).rotation.z = -PI / 2
		Ink.part(cloth, Ink.cone(0.0, 0.1, 0.05, 3), Ink.NAVY, Vector3(0.33, 0.0, face), Vector3(1, 1, 0.12), 6.0, 0.75).rotation.z = 2.0  # the tail
	return cloth


func _paper(fill: Color, radius: int, pad: int) -> StyleBoxFlat:
	var box := StyleBoxFlat.new()
	box.bg_color = fill
	box.set_corner_radius_all(radius)
	box.set_content_margin_all(pad)
	return box


func _build_card(ui: CanvasLayer) -> void:
	# The selected duck's card: its name in full, then who it is and how it is, then what it has learned.
	card.add_theme_stylebox_override("panel", paper)
	card.position = Vector2(24, 20)
	card.custom_minimum_size.x = BARS_W  # the same width as the readout under it
	ui.add_child(card)
	var stack := VBoxContainer.new()
	stack.add_theme_constant_override("separation", 3)
	card.add_child(stack)
	card_name.add_theme_font_size_override("font_size", 27)
	card_name.add_theme_color_override("font_color", Ink.NAVY)
	card_trait.add_theme_font_size_override("font_size", 14)
	card_trait.add_theme_color_override("font_color", Ink.CORAL)
	card_body.add_theme_font_size_override("font_size", 15)
	card_body.add_theme_color_override("font_color", Ink.NAVY)
	card_body.add_theme_constant_override("line_spacing", 5)
	card_skills.add_theme_font_size_override("font_size", 13)
	card_skills.add_theme_color_override("font_color", Color(Ink.NAVY, 0.65))
	card_skills.add_theme_constant_override("line_spacing", 4)
	for label in [card_name, card_trait, card_body, card_skills]:
		label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		stack.add_child(label)
	var rule := HSeparator.new()  # between how it is and what it can do
	var line := StyleBoxFlat.new()
	line.bg_color = Color(Ink.NAVY, 0.15)
	line.content_margin_top = 1
	rule.add_theme_stylebox_override("separator", line)
	stack.add_child(rule)
	stack.move_child(rule, card_skills.get_index())


func _build_menu(ui: CanvasLayer) -> void:
	# The controls as a printed card: a title, then two columns of sections, each a heading over its keys.
	var mono := SystemFont.new()
	mono.font_names = PackedStringArray(["Menlo", "Monaco", "Courier New", "monospace"])
	menu.set_anchors_preset(Control.PRESET_FULL_RECT)
	menu.mouse_filter = Control.MOUSE_FILTER_IGNORE
	menu.visible = false
	ui.add_child(menu)
	var dim := ColorRect.new()  # the garden dims behind it
	dim.color = Color(Ink.NAVY, 0.55)
	dim.set_anchors_preset(Control.PRESET_FULL_RECT)
	dim.mouse_filter = Control.MOUSE_FILTER_IGNORE
	menu.add_child(dim)
	var centre := CenterContainer.new()
	centre.set_anchors_preset(Control.PRESET_FULL_RECT)
	centre.mouse_filter = Control.MOUSE_FILTER_IGNORE
	menu.add_child(centre)
	var card_box := _paper(Ink.CREAM, 16, 34)
	card_box.border_width_bottom = 6  # the card sits on an inked edge, like a printed card
	card_box.border_color = Ink.NAVY
	var paper := PanelContainer.new()
	paper.add_theme_stylebox_override("panel", card_box)
	centre.add_child(paper)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 10)
	paper.add_child(column)
	var title := Label.new()
	title.text = "MICRO GARDEN"
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	title.add_theme_font_size_override("font_size", 34)
	title.add_theme_color_override("font_color", Ink.NAVY)
	column.add_child(title)
	var under := Label.new()
	under.text = "the controls"
	under.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	under.add_theme_font_size_override("font_size", 15)
	under.add_theme_color_override("font_color", Ink.CORAL)
	column.add_child(under)
	var spread := HBoxContainer.new()  # the sections, half in each column
	spread.add_theme_constant_override("separation", 46)
	column.add_child(spread)
	var sides := [VBoxContainer.new(), VBoxContainer.new()]
	for side in sides:
		side.add_theme_constant_override("separation", 22)
		spread.add_child(side)
	for s in CONTROLS.size():
		var section: Array = CONTROLS[s]
		var holder := VBoxContainer.new()
		holder.add_theme_constant_override("separation", 8)
		var head := Label.new()
		head.text = section[0]
		head.add_theme_font_size_override("font_size", 19)
		head.add_theme_color_override("font_color", Ink.TEAL)
		holder.add_child(head)
		var grid := GridContainer.new()
		grid.columns = 2
		grid.add_theme_constant_override("h_separation", 16)
		grid.add_theme_constant_override("v_separation", 7)
		for row in section[1]:
			var key := Label.new()
			key.text = row[0]
			key.add_theme_font_override("font", mono)
			key.add_theme_font_size_override("font_size", 13)
			key.add_theme_color_override("font_color", Ink.NAVY)
			key.add_theme_stylebox_override("normal", _paper(Ink.STONE, 5, 6))
			key.size_flags_horizontal = Control.SIZE_SHRINK_BEGIN  # the chip hugs the key, not the column
			var what := Label.new()
			what.text = row[1]
			what.add_theme_font_size_override("font_size", 16)
			what.add_theme_color_override("font_color", Ink.NAVY)
			what.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
			what.custom_minimum_size.x = 355  # a long one wraps rather than widening the card off the window
			what.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
			grid.add_child(key)
			grid.add_child(what)
		holder.add_child(grid)
		sides[0 if s < CONTROLS.size() / 2 else 1].add_child(holder)


func _fill_card(d: Dictionary) -> void:
	# One line each, no blank ones: who it is, how it is, what it has come to like, and what it can do.
	var state: String = "asleep" if d.asleep else ("crying" if d.get("crying", false) else FEELS.get(d.mood, d.mood))
	card_name.text = d.name
	card_trait.text = "%s  ·  %s" % [d.label, state] if d.label != "" else state
	var a: Dictionary = d.get("among", {})
	card_skills.visible = not a.is_empty()
	if a.is_empty():
		card_body.text = ""
		return
	var lines := ["likes %s best" % a.favourite, a.hand]
	if a.friend != "":
		lines.append("friends with %s" % a.friend)
	if a.grudge != "":
		lines.append("holds a grudge against %s" % a.grudge)
	card_body.text = "\n".join(lines)
	var names := ["swimming", "walking", "dancing", "eating", "fighting", "fashion", "music"]  # two to a line, so the card keeps its width
	var learned := PackedStringArray()
	for i in min(names.size(), a.skills.size()):
		learned.append("%s %d%%%s" % [names[i], a.skills[i] * 100, "\n" if i % 2 == 1 else "     "])
	card_skills.text = "".join(learned).strip_edges()


func _fruit(n: Node3D, f: Array) -> void:
	# Ten fruits (world/fields.py FRUITS), all the same food; each duck has a favourite.
	var kind: int = int(f[2])
	var leaf := Color("2f8f4a")
	var skin := func(mesh: Mesh, colour: Color, at: Vector3, size := Vector3.ONE) -> MeshInstance3D:
		var piece := Ink.part(n, mesh, colour, at, size, 6.0, -1.0, true)
		piece.material_override.set_shader_parameter("lift", 0.3)
		return piece
	var stalk := func(at: Vector3, lean := 0.0) -> void:
		Ink.part(n, Ink.cone(0.005, 0.04, 0.005, 4), Color("6b4326"), at, Vector3.ONE, 6.0).rotation.z = lean
	if kind == 3:  # a pear
		skin.call(Ink.ball(0.07, 8), Color("b8c94a"), Vector3(0, 0.066, 0))
		skin.call(Ink.ball(0.045, 8), Color("b8c94a"), Vector3(0, 0.13, 0))
		stalk.call(Vector3(0, 0.18, 0), 0.2)
	elif kind == 4:  # a pair of cherries on joined stalks
		for side in [-1.0, 1.0]:
			skin.call(Ink.ball(0.038, 8), Color("b3142c"), Vector3(side * 0.04, 0.038, 0))
			stalk.call(Vector3(side * 0.022, 0.09, 0), side * 0.45)
	elif kind == 5:  # a bunch of grapes
		for spot in [Vector3(-0.04, 0.1, 0), Vector3(0, 0.1, 0.02), Vector3(0.04, 0.1, 0), Vector3(-0.02, 0.066, 0.01),
				Vector3(0.02, 0.066, -0.01), Vector3(0, 0.033, 0), Vector3(0, 0.1, -0.03)]:
			skin.call(Ink.ball(0.026, 6), Color("6b3f8f"), spot)
		stalk.call(Vector3(0, 0.14, 0))
	elif kind == 6:  # a strawberry, point down, with a green crown
		skin.call(Ink.cone(0.0, 0.1, 0.055, 8), Color("e0293a"), Vector3(0, 0.05, 0))
		for k in 5:
			var a := TAU * k / 5.0
			Ink.part(n, Ink.ball(0.018, 4), leaf, Vector3(cos(a) * 0.03, 0.102, sin(a) * 0.03), Vector3(1.6, 0.4, 1.0), 6.0)
	elif kind == 7:  # a lemon, long, with a nub at each end
		skin.call(Ink.ball(0.06, 8), Color("f5dc3a"), Vector3(0, 0.06, 0), Vector3(1.35, 0.9, 0.9))
		for side in [-1.0, 1.0]:
			skin.call(Ink.ball(0.014, 5), Color("f5dc3a"), Vector3(side * 0.083, 0.06, 0))
	elif kind == 8:  # a plum
		skin.call(Ink.ball(0.062, 8), Color("5a2a6e"), Vector3(0, 0.062, 0), Vector3(1.0, 1.05, 0.9))
		stalk.call(Vector3(0, 0.13, 0))
	elif kind == 9:  # a peach, with a leaf
		skin.call(Ink.ball(0.07, 8), Color("f6a26b"), Vector3(0, 0.068, 0))
		Ink.part(n, Ink.ball(0.022, 5), leaf, Vector3(0.025, 0.14, 0), Vector3(1.6, 0.4, 0.9), 6.0)
	elif kind == 0:
		Ink.part(n, Ink.ball(0.075, 8), Color("f39a2b"), Vector3(0, 0.072, 0), Vector3.ONE, 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.3)
		Ink.part(n, Ink.ball(0.018, 5), leaf, Vector3(0, 0.148, 0), Vector3(1.6, 0.5, 1.0), 6.0)
	elif kind == 1:
		Ink.part(n, Ink.ball(0.075, 8), Color("d8342c"), Vector3(0, 0.068, 0), Vector3(1.0, 0.9, 1.0), 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.3)
		Ink.part(n, Ink.cone(0.006, 0.045, 0.006, 4), Color("6b4326"), Vector3(0, 0.15, 0), Vector3.ONE, 6.0)
		Ink.part(n, Ink.ball(0.022, 5), leaf, Vector3(0.03, 0.155, 0), Vector3(1.6, 0.4, 0.9), 6.0)
	else:
		for k in 5:  # a curve of short pieces, fat in the middle
			var u := (k - 2) / 2.0
			var piece := Ink.part(n, Ink.cone(0.03 - 0.008 * abs(u), 0.062, 0.03 - 0.008 * abs(u), 6), Color("f4d23c"),
				Vector3(u * 0.075, 0.035 + 0.03 * u * u, 0), Vector3.ONE, 6.0, -1.0, true)
			piece.rotation.z = PI / 2 - u * 0.55
			piece.material_override.set_shader_parameter("lift", 0.3)
	n.rotation.y = f[0] * 7.0 + f[1] * 3.0


func _ball(n: Node3D, b: Array) -> void:
	# Ten kinds of ball (body/stub2d/stub.py BALL_STYLES), all the same size. The markings are children of the
	# ball, so they roll with it.
	var style: int = int(b[2]) if b.size() > 2 else 0
	n.set_meta("style", style)
	var r := 0.06 * 1.9
	var base: Color = [Color("f6f1e6"), Color("f6f1e6"), Color("f6f1e6"), Color("e0782c"), Color("c8d84a"), Ink.NAVY,
		Ink.CORAL, Ink.TEAL, Ink.MUSTARD, Ink.STONE][style % 10]
	var ball := Ink.part(n, Ink.ball(r, 10), base, Vector3(0, r, 0), Vector3.ONE, 5.0, -1.0, true)
	ball.material_override.set_shader_parameter("lift", 0.3)
	var mark := func(colour: Color, size: Vector3, turn := Vector3.ZERO, at := Vector3.ZERO) -> void:
		var piece := Ink.part(ball, Ink.ball(r * 1.01, 10), colour, at, size, 5.0)
		piece.rotation = turn
		piece.material_override.set_shader_parameter("lift", 0.3)
	var spots := func(colour: Color, count: int, size: float) -> void:  # evenly over the ball
		for k in count:
			var y := 1.0 - 2.0 * (k + 0.5) / count
			var ring := sqrt(1.0 - y * y)
			var at := Vector3(cos(k * 2.4) * ring, y, sin(k * 2.4) * ring) * r * 0.93
			Ink.part(ball, Ink.ball(r * size, 6), colour, at, Vector3.ONE, 5.0)
	match style % 10:
		0:  # the classic: a red band
			mark.call(Color("e2483d"), Vector3(1.0, 0.34, 1.0))
		1:  # a beach ball: three bands of colour
			for k in 3:
				mark.call([Ink.CORAL, Ink.MUSTARD, Ink.TEAL][k], Vector3(0.5, 1.0, 1.0), Vector3(0, TAU * k / 6.0, 0))
		2:  # a football: navy patches
			spots.call(Ink.NAVY, 12, 0.3)
		3:  # a basketball: two dark seams
			mark.call(Ink.NAVY, Vector3(1.0, 0.06, 1.0))
			mark.call(Ink.NAVY, Vector3(0.06, 1.0, 1.0))
		4:  # a tennis ball: a pale curve
			mark.call(Color("f6f1e6"), Vector3(1.0, 0.08, 1.0), Vector3(0.6, 0, 0.4))
		5:  # a snooker ball: a cream spot
			Ink.part(ball, Ink.ball(r * 0.45, 8), Color("f6f1e6"), Vector3(0, 0, r * 0.8), Vector3(1, 1, 0.5), 5.0)
		6:  # a plain rubber ball
			pass
		7:  # stripes
			for y in [-0.45, 0.45]:
				mark.call(Color("f6f1e6"), Vector3(0.9, 0.16, 0.9), Vector3.ZERO, Vector3(0, y * r, 0))  # the ball is narrower up there
		8:  # polka dots
			spots.call(Ink.CORAL, 16, 0.24)
		9:  # a moon, with craters
			spots.call(Color("8f8a80"), 9, 0.32)


func _props(kind: String, items: Array, make: Callable) -> void:
	# one node per item, rebuilt only when the count changes
	var nodes: Array = props.get(kind, [])
	if nodes.size() != items.size():
		for n in nodes:
			n.queue_free()
		nodes = []
		for item in items:
			var n := Node3D.new()
			n.position = Vector3(item[0], 0, -item[1])  # where it is, before anything asks whether it moved
			add_child(n)
			make.call(n, item)
			nodes.append(n)
		props[kind] = nodes
	for i in items.size():
		var to := Vector3(items[i][0], nodes[i].position.y, -items[i][1])  # garden y is Godot -z; _show_held sets height
		var carried: bool = snap.get("held") != null and snap.held[0] == "food" and int(snap.held[1]) == i
		if kind == "food" and not carried and not nodes[i].has_meta("kicked") and nodes[i].position.distance_to(to) > KICKED_M:
			nodes[i].set_meta("kicked", {"from": nodes[i].position, "t": 0.0, "to": to})  # it jumped: bounce it there
		if nodes[i].has_meta("kicked"):
			nodes[i].get_meta("kicked").to = to  # _roll_kicked moves it
		else:
			nodes[i].position = to


func _roll_kicked(dt: float) -> void:
	# A fruit that moved further in one step than anything carries it was kicked: it bounces twice, lower each
	# time, and spins as it rolls to where it landed.
	for n in props.get("food", []):
		if not n.has_meta("kicked"):
			continue
		var roll: Dictionary = n.get_meta("kicked")
		roll.t += dt
		var u: float = min(roll.t / KICKED_S, 1.0)
		var from: Vector3 = roll.from
		var along: Vector3 = roll.to - from
		n.position = from + along * (1.0 - pow(1.0 - u, 2.0)) + Vector3(0, 0.22 * abs(sin(u * TAU)) * (1.0 - u), 0)
		if along.length() > 0.01:
			n.rotate(Vector3.UP.cross(along).normalized(), -dt * 14.0 * (1.0 - u))
		if u >= 1.0:
			n.remove_meta("kicked")


func _draw_bars() -> void:
	# One panel under the card: needs in coral, moods in mustard, wants in teal, a heading and a bar each
	if selected < 0 or snap.is_empty() or possessing:
		return
	var rows: Array = snap.ducks[selected].get("readout", [])
	if rows.is_empty():
		return
	var tints := {"needs": Ink.CORAL, "moods": Ink.MUSTARD, "wants": Ink.TEAL}
	var font := ThemeDB.fallback_font
	var row_h := 20.0
	var head_h := 27.0  # a heading in the section's own ink, over a rule of it
	var sections := 0
	var last := ""
	for row in rows:
		sections += int(row[0] != last)
		last = row[0]
	var edge := float(PAPER_PAD)
	bars.draw_style_box(paper, Rect2(0, 0, BARS_W, rows.size() * row_h + sections * head_h + 2.0 * edge))
	var bar_x := edge + 82.0
	var bar_w := BARS_W - bar_x - edge - 34.0
	var y := edge
	last = ""
	for row in rows:
		if row[0] != last:
			last = row[0]
			# the ink names the section on the bars and the rule; the word stays navy, which reads on cream
			var tint: Color = tints.get(last, Ink.NAVY)
			bars.draw_rect(Rect2(edge, y + 5.0, 8.0, 8.0), tint)
			bars.draw_string(font, Vector2(edge + 14.0, y + 14.0), last.to_upper(), HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Ink.NAVY)
			bars.draw_rect(Rect2(edge, y + 20.0, BARS_W - 2.0 * edge, 1.5), Color(tint, 0.55))
			y += head_h
		bars.draw_string(font, Vector2(edge, y + 13.0), row[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Ink.NAVY)
		bars.draw_rect(Rect2(bar_x, y + 3.0, bar_w, 11), Color(Ink.NAVY, 0.15))
		bars.draw_rect(Rect2(bar_x, y + 3.0, bar_w * float(row[2]), 11), tints.get(last, Ink.NAVY))
		bars.draw_string(font, Vector2(BARS_W - edge - 30.0, y + 13.0), "%d" % int(row[2] * 100.0),
			HORIZONTAL_ALIGNMENT_RIGHT, 30, 12, Color(Ink.NAVY, 0.7))
		y += row_h


func _draw_overlay() -> void:
	# The ridden duck's brain input and output: each eye's 721 columns on the fly's lattice, left eye on
	# the left, and the six descending readouts as bars. The brain sees only this, not the rendered view.
	if not possessing or not snap.has("ride"):
		return
	var ride: Dictionary = snap.ride
	var hexes := Marshalls.base64_to_raw(ride.hex)
	var lum := Marshalls.base64_to_raw(ride.lum)
	var n := lum.size() / 2
	var r := 110.0
	var size := overlay.size
	for eye in 2:
		var centre := Vector2(size.x - (3.0 - 2.0 * eye) * (r + 12.0) - 12.0, size.y - r - 24.0)
		overlay.draw_circle(centre, r + 8.0, Color(Ink.NAVY, 0.85))
		for k in n:
			var az := float(hexes[k] - 256 if hexes[k] > 127 else hexes[k]) / 127.0
			var el := float(hexes[n + k] - 256 if hexes[n + k] > 127 else hexes[n + k]) / 127.0
			overlay.draw_circle(centre + Vector2(az, -el) * r, 2.6, Ink.NAVY.lerp(Ink.CREAM, lum[eye * n + k] / 255.0))
	var left: float = size.x - 4.0 * (r + 12.0) - 12.0 + 4.0
	var y: float = size.y - 2.0 * r - 60.0 - 22.0 * ride.dn.size()
	overlay.draw_style_box(paper, Rect2(left - PAPER_PAD, y - PAPER_PAD, 330.0, 22.0 * ride.dn.size() + 2.0 * PAPER_PAD))
	for bar in ride.dn:
		overlay.draw_rect(Rect2(left, y, 160, 12), Color(Ink.NAVY, 0.25))
		overlay.draw_rect(Rect2(left, y, 160.0 * clamp(bar[1] / 5.0, 0.0, 1.0), 12), Ink.CORAL)
		overlay.draw_string(ThemeDB.fallback_font, Vector2(left + 170.0, y + 12.0), "%s %.1f Hz" % [bar[0], bar[1]], HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Ink.NAVY)
		y += 22.0


func _quack(duck: int, tag: String) -> void:
	# A quack: a pitch-swept buzz shaped by the tag. Each duck's voice is seeded from its number, so it
	# stays the same. Synthesised here; the garden ships no audio files.
	if tag == "drum":
		_thump()
		return
	if tag.begins_with("instrument:"):
		var parts := tag.split(":")
		_note(duck, int(parts[1]), parts.size() > 2)
		return
	var shape: Array = {"alarm": [900.0, 0.5, 0.12, 3], "greet": [520.0, 0.8, 0.16, 2], "inquire": [480.0, 1.3, 0.22, 1],
		"peck": [700.0, 0.9, 0.05, 2], "chirp": [1100.0, 1.1, 0.07, 2], "coo": [330.0, 0.9, 0.4, 1],
		"wheee": [600.0, 1.8, 0.45, 1]}.get(tag, [500.0, 0.8, 0.15, 1])
	var voice_rng := RandomNumberGenerator.new()
	voice_rng.seed = 9001 + duck * 7919
	var high: float = [0.62, 0.8, 1.0, 1.22, 1.5][duck % 5] * voice_rng.randf_range(0.94, 1.06)
	var reedy: float = voice_rng.randf_range(0.0, 1.0)  # 0 a soft hum, 1 a buzz
	var wobble: float = voice_rng.randf_range(0.0, 0.06)
	var wobble_hz: float = voice_rng.randf_range(9.0, 22.0)
	var pace: float = voice_rng.randf_range(0.8, 1.3)
	var playback: AudioStreamGeneratorPlayback = voice.get_stream_playback()
	var phase := 0.0
	for rep in shape[3]:
		var n := int(shape[2] * pace * 22050.0)
		for k in n + int(900 * pace):  # plus a gap
			if playback.get_frames_available() < 1:
				return
			var u := float(k) / n
			var f: float = shape[0] * high * lerp(1.0, float(shape[1]), u) * (1.0 + wobble * sin(TAU * wobble_hz * k / 22050.0))
			phase += f / 22050.0
			var saw := fmod(phase, 1.0) * 2.0 - 1.0
			var v: float = lerp(sin(TAU * phase), saw, reedy) * sin(min(u, 1.0) * PI) * 0.25 if k < n else 0.0
			playback.push_frame(Vector2(v, v))


func _note(duck: int, kind: int, sour: bool) -> void:
	# A note on an instrument, from a pentatonic scale so any run of them sounds like a tune. How well the duck
	# plays (its music skill, `tune`) sets how far off pitch it drifts; a sour note is well off.
	var tone: Array = INSTRUMENT_TONES[kind % INSTRUMENT_TONES.size()]  # [Hz, seconds, voice]
	var skill: float = float(snap.ducks[duck].get("tune", 1.0)) if duck < snap.get("ducks", []).size() else 1.0
	var off: float = randf_range(-1.0, 1.0) * ((1.0 - skill) * 0.6 + (1.8 if sour else 0.0))  # semitones out of tune
	var hz: float = float(tone[0]) * pow(2.0, ([0, 2, 4, 7, 9, 12][randi() % 6] + off) / 12.0)
	var out: AudioStreamGeneratorPlayback = voice.get_stream_playback()
	var n := int(float(tone[1]) * 22050.0)
	var phase := 0.0
	for k in n:
		if out.get_frames_available() < 1:
			return
		var u := float(k) / n
		var t := float(k) / 22050.0
		phase += hz / 22050.0
		var v := 0.0
		match tone[2]:
			"struck":
				v = (sin(TAU * phase) + 0.3 * sin(TAU * phase * 4.0)) * exp(-7.0 * u)
			"shaken":
				v = randf_range(-1.0, 1.0) * exp(-9.0 * u)
			"jingle":
				v = randf_range(-1.0, 1.0) * sin(TAU * t * 5200.0) * exp(-6.0 * u)
			"ring":
				v = (sin(TAU * phase) + 0.4 * sin(TAU * phase * 2.76) + 0.2 * sin(TAU * phase * 5.4)) * exp(-3.0 * u)
			"plucked":
				v = (fmod(phase, 1.0) * 2.0 - 1.0) * exp(-5.0 * u) * 0.6
			"harp":
				v = sin(TAU * phase) * exp(-4.0 * u)
			"brass":
				v = clamp(sin(TAU * phase) * 3.0, -1.0, 1.0) * min(u * 12.0, 1.0) * (1.0 - u) * 0.7
			"blown":
				v = sin(TAU * phase + 0.3 * sin(TAU * t * 5.5)) * min(u * 10.0, 1.0) * (1.0 - u)
		out.push_frame(Vector2(v, v) * 0.09)


func _instrument(n: Node3D, it: Array) -> void:
	# Ten instruments (body/stub2d/stub.py INSTRUMENTS), built from the same few shapes as everything else.
	var kind: int = int(it[2])
	n.set_meta("kind", kind)
	var wood := Scenery.WOOD
	var brass := Color("e6b54a")
	var part := func(mesh: Mesh, colour: Color, at: Vector3, size := Vector3.ONE, lined := false) -> MeshInstance3D:
		var piece := Ink.part(n, mesh, colour, at, size, 6.0, -1.0, lined)
		piece.material_override.set_shader_parameter("lift", 0.3)
		return piece
	match kind:
		0:  # a xylophone: bars from long to short on a wooden frame
			for side in [-1.0, 1.0]:
				part.call(BoxMesh.new(), wood, Vector3(0, 0.03, side * 0.07), Vector3(0.34, 0.03, 0.02))
			for k in 6:
				part.call(BoxMesh.new(), [Ink.CORAL, Ink.MUSTARD, Color("7ac74f"), Ink.TEAL, Color("3a7bd5"), Color("7d4fc2")][k],
					Vector3(-0.13 + 0.052 * k, 0.05, 0), Vector3(0.04, 0.015, 0.2 - 0.018 * k), true)
		1:  # maracas, crossed
			for side in [-1.0, 1.0]:
				part.call(Ink.ball(0.045, 8), [Ink.CORAL, Ink.MUSTARD][int(side > 0)], Vector3(side * 0.05, 0.045, -0.04), Vector3(1, 1, 1.2), true)
				part.call(Ink.cone(0.01, 0.12, 0.012, 5), wood, Vector3(side * 0.02, 0.02, 0.05), Vector3.ONE).rotation.x = PI / 2 - side * 0.3
		2:  # a tambourine: a drum hoop with jingles round the rim
			part.call(Ink.cone(0.12, 0.035, 0.12, 14), Color("f6f1e6"), Vector3(0, 0.02, 0), Vector3.ONE, true)
			for k in 6:
				var a := TAU * k / 6.0
				part.call(Ink.cone(0.018, 0.008, 0.018, 8), brass, Vector3(cos(a) * 0.12, 0.03, sin(a) * 0.12))
		3:  # a triangle, standing on a little stand, and its beater
			for k in 3:  # each side from one corner to the next
				var from := Vector2(cos(PI / 2 + TAU * k / 3.0), sin(PI / 2 + TAU * k / 3.0)) * 0.1
				var to := Vector2(cos(PI / 2 + TAU * (k + 1) / 3.0), sin(PI / 2 + TAU * (k + 1) / 3.0)) * 0.1
				var mid := (from + to) / 2.0
				var side: MeshInstance3D = part.call(BoxMesh.new(), brass, Vector3(mid.x, 0.12 + mid.y, 0), Vector3(from.distance_to(to), 0.02, 0.02), true)
				side.rotation.z = (to - from).angle()
			part.call(Ink.cone(0.04, 0.02, 0.05, 8), wood, Vector3(0, 0.01, 0))
			part.call(Ink.cone(0.004, 0.11, 0.004, 4), Color("c9c2b2"), Vector3(0.1, 0.006, 0.04)).rotation.z = PI / 2
		4:  # a toy piano
			part.call(BoxMesh.new(), Ink.CORAL, Vector3(0, 0.06, 0), Vector3(0.3, 0.12, 0.16), true)
			part.call(BoxMesh.new(), Color("f6f1e6"), Vector3(0, 0.1, 0.07), Vector3(0.26, 0.02, 0.05))
			for k in 5:
				part.call(BoxMesh.new(), Ink.NAVY, Vector3(-0.1 + 0.05 * k, 0.115, 0.06), Vector3(0.02, 0.012, 0.03))
		5:  # a guitar, lying on the grass
			part.call(Ink.ball(0.08, 10), Color("c46b2a"), Vector3(0, 0.03, 0), Vector3(1, 0.4, 1), true)
			part.call(Ink.ball(0.06, 10), Color("c46b2a"), Vector3(0.1, 0.03, 0), Vector3(1, 0.4, 1), true)
			part.call(Ink.ball(0.025, 8), Ink.NAVY, Vector3(0.02, 0.058, 0), Vector3(1, 0.2, 1))
			part.call(BoxMesh.new(), wood, Vector3(0.25, 0.04, 0), Vector3(0.22, 0.02, 0.035))
			part.call(BoxMesh.new(), wood, Vector3(0.38, 0.04, 0), Vector3(0.05, 0.025, 0.055))
		6:  # a trumpet
			part.call(Ink.cone(0.015, 0.08, 0.06, 10), brass, Vector3(0.14, 0.06, 0), Vector3.ONE, true).rotation.z = -PI / 2
			part.call(Ink.cone(0.013, 0.2, 0.013, 6), brass, Vector3(0, 0.06, 0), Vector3.ONE, true).rotation.z = PI / 2
			for k in 3:
				part.call(Ink.cone(0.01, 0.04, 0.01, 5), brass, Vector3(-0.02 + 0.025 * k, 0.09, 0))
		7:  # a hand bell
			part.call(Ink.cone(0.07, 0.1, 0.03, 10), brass, Vector3(0, 0.05, 0), Vector3.ONE, true)
			part.call(Ink.cone(0.012, 0.07, 0.012, 5), wood, Vector3(0, 0.13, 0))
			part.call(Ink.ball(0.018, 6), Ink.NAVY, Vector3(0, 0.01, 0))
		8:  # a recorder, lying down, with its holes
			part.call(Ink.cone(0.018, 0.3, 0.014, 8), Color("f6f1e6"), Vector3(0, 0.02, 0), Vector3.ONE, true).rotation.z = PI / 2
			for k in 5:
				part.call(Ink.ball(0.005, 4), Ink.NAVY, Vector3(-0.08 + 0.035 * k, 0.037, 0))
		9:  # a harp: a curved frame and its strings
			part.call(BoxMesh.new(), wood, Vector3(-0.08, 0.15, 0), Vector3(0.03, 0.3, 0.03), true)
			part.call(BoxMesh.new(), wood, Vector3(0.0, 0.02, 0), Vector3(0.2, 0.03, 0.04), true)
			part.call(BoxMesh.new(), wood, Vector3(0.0, 0.26, 0), Vector3(0.2, 0.03, 0.03), true).rotation.z = -0.35
			for k in 5:
				part.call(BoxMesh.new(), Color("f6f1e6"), Vector3(-0.05 + 0.03 * k, 0.13 + 0.01 * k, 0), Vector3(0.004, 0.2 - 0.025 * k, 0.004))
	n.rotation.y = float(it[0]) * 5.0 + float(it[1]) * 3.0


func _thump() -> void:
	# a soft, quiet thump, the same from any duck
	var skin: AudioStreamGeneratorPlayback = voice.get_stream_playback()
	var turn := 0.0
	for k in int(0.14 * 22050.0):
		if skin.get_frames_available() < 1:
			return
		var u := float(k) / (0.14 * 22050.0)
		turn += lerp(150.0, 80.0, u) / 22050.0
		var thump: float = sin(TAU * turn) * exp(-5.0 * u) * 0.09
		skin.push_frame(Vector2(thump, thump))


func _hand(_dt: float) -> void:
	# The glove follows the mouse over the lawn. While gripping, the garden is told where it is, so what
	# it holds moves with it and the ducks can see it.
	_cursor()
	if snap.is_empty() or possessing or not hand_mode or not mouse_inside:
		glove.visible = false
		return
	var xy = _mouse_ground()
	glove.visible = xy != null
	if xy == null:
		return
	var at: Vector2 = xy.clamp(Vector2.ZERO, Vector2(snap.size, snap.size))
	glove.position = glove.position.lerp(Vector3(at.x, 0.22 if gripping else 0.4, -at.y), 0.5)
	glove.rotation.y = orbit.x + PI  # fingers away from the camera
	var now := Time.get_ticks_msec() / 1000.0
	trail.append([now, at])
	while trail.size() > 1 and now - trail[0][0] > 0.12:
		trail.pop_front()
	if gripping:
		_act("garden.hand_at", {"x": at.x, "y": at.y})


func _let_go() -> void:
	var now := Time.get_ticks_msec() / 1000.0
	var v := Vector2.ZERO
	if trail.size() > 1 and now - pressed_at > 0.25:
		v = (trail[-1][1] - trail[0][1]) / max(trail[-1][0] - trail[0][0], 0.02)
	var at: Vector2 = trail[-1][1] if not trail.is_empty() else Vector2.ZERO
	_act("garden.release", {"x": at.x, "y": at.y, "vx": v.x, "vy": v.y})


func _pond_sound() -> void:
	if snap.is_empty() or snap.pond == null:
		return
	var playback: AudioStreamGeneratorPlayback = water.get_stream_playback()
	var loud: float = 0.4 + 0.6 * snap.light
	for k in playback.get_frames_available():
		water_t += 1.0 / 22050.0
		water_level = clamp(water_level * 0.995 + randf_range(-0.04, 0.04), -1.0, 1.0)  # brown noise: a leaky sum of white
		var v := water_level * loud * (0.6 + 0.4 * sin(water_t * 1.3) * sin(water_t * 0.37))
		playback.push_frame(Vector2(v, v))


func _camera(dt: float) -> void:
	still += dt
	if possessing and selected >= 0:
		var d: Node3D = ducks[selected]
		cam.fov = 100.0  # wide and low-res to look non-human; halftone mostly off, since it hid the garden at this scale
		get_viewport().scaling_3d_scale = 0.45
		RenderingServer.global_shader_parameter_set("screen", 0.25)
		cam.position = d.position + Vector3(0, 0.3, 0)
		cam.rotation = Vector3(-0.15, d.rotation.y - PI / 2, 0)  # a duck faces +x and a camera looks down -z
		return
	cam.fov = 38.0
	get_viewport().scaling_3d_scale = 1.0
	RenderingServer.global_shader_parameter_set("screen", 1.0)
	var size: float = snap.get("size", 4.0)
	var sway: float = sin(Time.get_ticks_msec() / 9000.0) * DRIFT * (1.0 if calm == 1.0 else 0.0) * clamp(still - 3.0, 0.0, 1.0)
	var yaw: float = orbit.x + sway
	# orbit the garden's middle, or ease over to the selected duck
	var want: Vector3 = ducks[selected].position + Vector3(0, 0.15, 0) if selected >= 0 and selected < ducks.size() else Vector3(size / 2, 0.1, -size / 2)
	centre = want if centre == Vector3.INF else centre.lerp(want, min(1.0, 3.0 * dt))
	cam.position = centre + Vector3(cos(yaw) * cos(orbit.y), sin(orbit.y), sin(yaw) * cos(orbit.y)) * orbit.z
	cam.look_at(centre)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP or e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			orbit.z = clamp(orbit.z * (0.92 if e.button_index == MOUSE_BUTTON_WHEEL_UP else 1.08), 2.0, 12.0)
		elif e.button_index == MOUSE_BUTTON_LEFT and hand_mode:  # down grabs, up releases
			if e.pressed:
				gripping = true
				pressed_at = Time.get_ticks_msec() / 1000.0
				dragged = 0.0
				var xy = _ground(e.position)
				if xy != null:
					_act("garden.grab", {"x": xy.x, "y": xy.y})
			else:
				gripping = false
				var quick: bool = Time.get_ticks_msec() / 1000.0 - pressed_at < 0.25 and dragged < 6.0
				_let_go()
				if quick:  # a tap, not a carry
					_click(e.position)
		elif e.button_index == MOUSE_BUTTON_LEFT:  # no hand: drag turns the camera, a tap clicks
			dragging = e.pressed
			if e.pressed:
				dragged = 0.0
			elif dragged < 6.0:
				_click(e.position)
		elif e.button_index == MOUSE_BUTTON_RIGHT or e.button_index == MOUSE_BUTTON_MIDDLE:
			dragging = e.pressed
	elif e is InputEventMouseMotion:
		dragged += e.relative.length()
		if dragging:
			orbit.x += e.relative.x * 0.006
			orbit.y = clamp(orbit.y + e.relative.y * 0.006, PITCH.x, PITCH.y)
			still = 0.0
	elif e is InputEventKey and e.pressed and not e.echo:
		_key(e.keycode)


func _ground(screen: Vector2):
	var from := cam.project_ray_origin(screen)
	var dir := cam.project_ray_normal(screen)
	if dir.y >= 0.0:
		return null
	var hit := from - dir * (from.y / dir.y)
	return Vector2(hit.x, -hit.z)


func _duck_at(screen: Vector2) -> int:
	# The duck under the pointer, by where it is drawn: its body, not the grass at its feet. Its height on
	# screen sets how near a tap has to land, so it works zoomed right in and from across the garden.
	var best := -1
	var nearest := INF
	for i in ducks.size():
		var feet: Vector3 = ducks[i].global_position
		if cam.is_position_behind(feet):
			continue
		var top := cam.unproject_position(feet + Vector3(0, DUCK_TALL, 0))
		var down := cam.unproject_position(feet)
		var reach: float = max(0.6 * down.distance_to(top), 12.0)
		var off := screen.distance_to((top + down) * 0.5)
		if off < reach and off < nearest:
			nearest = off
			best = i
	return best


func _click(screen: Vector2) -> void:
	var xy = _ground(screen)
	if xy == null or snap.is_empty():
		return
	var tapped := _duck_at(screen)
	if tapped >= 0:
		selected = tapped
		return
	if snap.music != null and xy.distance_to(Vector2(snap.music[0], snap.music[1])) < 0.4:  # each click steps the volume
		var volume: float = music.step_volume()
		_act("garden.volume", {"level": volume})
		_say("music off" if volume == 0.0 else "music volume %d%%" % int(volume * 100))
		return
	if xy.distance_to(Vector2(snap.tree[0], snap.tree[1])) < 0.3:
		_act("garden.shake_tree", {})
	else:
		selected = -1  # food is on F, so a stray click never feeds


func _key(code: int) -> void:
	if code == KEY_TAB and (possessing or selected >= 0):
		possessing = not possessing
		if not possessing:
			_act("garden.wheel", {"duck": -1, "fwd": 0, "turn": 0})
	elif code == KEY_SLASH:
		help_on = not help_on
	elif code == KEY_O:
		overlay.visible = not overlay.visible
	elif code == KEY_C:
		_act("garden.scare", {})
	elif code == KEY_F:
		_act("garden.shake_tree", {})
	elif code == KEY_M:
		var xy = _mouse_ground()
		if xy != null:
			_act("garden.music", {"x": xy.x, "y": xy.y, "on": int(snap.music == null)})
	elif selected >= 0 and code == KEY_G:  # hand-feeding teaches the duck your hand is good
		_act("garden.give", {"duck": selected})
	elif selected >= 0 and code == KEY_P:
		_act("garden.pet", {"duck": selected})
	elif code == KEY_D and not possessing:  # D steers while riding
		var here = _mouse_ground()
		if here != null:
			_act("garden.drum", {"x": here.x, "y": here.y, "on": int(snap.get("drum") == null)})
	elif code == KEY_I:
		var spot = _mouse_ground()
		if spot != null:
			_act("garden.instrument", {"x": spot.x, "y": spot.y})
	elif code == KEY_B:
		var where = _mouse_ground()
		if where != null and where.x > 0 and where.y > 0 and where.x < snap.size and where.y < snap.size:
			_act("garden.drop_ball", {"x": where.x, "y": where.y})
	elif code == KEY_H:
		hand_mode = not hand_mode
		if not hand_mode and gripping:
			gripping = false
			_let_go()
		_say("the hand is out: hold the left button to pick things up" if hand_mode else "the hand is put away")
	elif code == KEY_T:  # the ducks decide who wears it
		var spot = _mouse_ground()
		if spot != null:
			_act("garden.drop_hat", {"x": spot.x, "y": spot.y})


func _mouse_ground():
	return _ground(get_viewport().get_mouse_position())


func _say(line: String) -> void:
	said = line
	said_until = Time.get_ticks_msec() / 1000.0 + 2.5


func _act(method: String, params: Dictionary) -> void:
	out.put_packet(JSON.stringify({"method": method, "params": params}).to_utf8_buffer())
