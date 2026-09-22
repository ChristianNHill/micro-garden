# The garden, drawn from what the Python side publishes (viewer/snapshot.py): one JSON datagram a step, in.
# Whatever the player does goes back the same way, as the garden's own control calls. Nothing here decides
# anything about a duck. Run the garden with --godot and then this project; either can start first.
#
# After `--`:
#   --orbit=yaw,pitch,distance  where the camera starts       --ride=N   start on duck N's back, --select=N beside it
#   --reduced-motion            a still camera, half the animation       --mute     no sound
#   --light=0.1                 draw that daylight whatever the hour, to look at the night; --asleep=N draws duck N asleep
#   --music=DIR                 the folder of mp3s the music box plays (default ~/.cache/micro-garden/music)
#   --keys=MBT                  press these keys after three seconds, to check they reach the garden
#   --port=N                    listen there, for a garden started with MICRO_GARDEN_PORT=N
#   --shot=file.png             save the window after --shot-after=S seconds (6 by default)
#   --record=DIR                save the window to DIR as numbered pngs, --record-fps=N a second (10), from
#                               --record-after=S seconds (6) for --record-for=S (20): frames for a gif
#   --feed-at=x,y               drop food there once snapshots arrive, and --quit-after=S: both for Gate 13
extends Node3D

const Ink := preload("res://ink.gd")
const Duck := preload("res://duck.gd")
const Scenery := preload("res://scenery.gd")
const Hats := preload("res://hats.gd")
const BrainView := preload("res://brainview.gd")
const MusicBox := preload("res://musicbox.gd")
const SNAPSHOT_PORT := 7650  # actions go back on the next one up; --port moves the pair
const HELP := """
    MICRO GARDEN


    tap a duck            select it: its needs, its moods, its brain, who its friends are
    tap the grass         let it go

    with a duck selected
      Tab                 ride it (W A S D steer, O hides its eyes, Tab gets off)
      P                   pet it
      G                   hand it a fruit

    H                     bring out the hand, or put it away. With it out:
      hold and drag         pick up fruit, a hat, a ball, the drum, the music box, or a duck
      let go on the move    throw it (a thrown duck thinks less of you)

    F, or tap the tree    shake fruit down
    T                     drop a hat at the mouse, for whoever wants it
    B                     drop a ball at the mouse
    D                     put a drum down at the mouse, or take it up
    M                     put the music box down at the mouse, or take it up
    tap the music box     step its volume
    C                     clap

    drag                  turn the camera (right-drag while the hand is out)
    scroll                zoom

    /                     close this
"""
const HELP_HINT := "/  controls"
const HELP_RIDING := "W A S D  steer      O  hide its eyes      Tab  get off"
const PITCH := Vector2(0.35, 1.25)  # radians above the horizon the camera may sit
const DRIFT := 0.12  # radians the idle camera sways either way

var udp := PacketPeerUDP.new()
var out := PacketPeerUDP.new()
var snap := {}
var got := 0
var ducks: Array = []
var selected := -1
var flag := Node3D.new()  # turns to the wind
var falls: Array = []  # the waterfall's sheets of water, which shimmer
var props := {}  # what is in the garden now, by kind, as the nodes drawn for it

var cam := Camera3D.new()
var orbit := Vector3(0.75, 0.7, 8.8)  # yaw, pitch, distance
var centre := Vector3.INF  # what the camera turns about
var glove := Node3D.new()  # the player's hand, where the mouse is over the garden
var hand_mode := false  # H: the cursor is the glove and the left button is its grip; off, a plain cursor and the camera
var mouse_inside := true
var gripping := false  # the left button is down: the hand is closed on whatever was there
var pressed_at := 0.0
var trail: Array = []  # [seconds, garden xy] of the hand lately, for how fast it was moving when it let go
var dragging := false
var dragged := 0.0
var still := 0.0  # seconds since the player last moved the camera
var calm := 1.0

var possessing := false  # the camera rides the selected duck and W A S D has its wheel; its brain keeps watching
var voice := AudioStreamPlayer.new()
var heard := -1.0  # garden time of the last quack played
var water := AudioStreamPlayer.new()  # the pond: soft brown noise that swells and falls, quieter at night
var water_level := 0.0
var water_t := 0.0

var music := Node.new()  # the box's own music, from the player's folder
var said := ""  # a line from this window, shown with the garden's toasts for a moment
var said_until := 0.0
var bars := Control.new()  # the selected duck's needs, moods and wants, as bars under its card
var brain := Control.new()  # the selected duck's brain, firing
var watching := -2  # whose brain the garden was last asked for
var asked := 0.0
var overlay := Control.new()  # the ride view's retinas and descending-neuron bars; O hides it
var card := Label.new()  # who the selected duck is
var help := Label.new()  # what the player can do, bottom right; / hides it
var help_on := false  # collapsed to a hint until / opens it, over the whole window
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
	for label in [card, toasts, help]:
		label.add_theme_color_override("font_color", Ink.NAVY)
		label.add_theme_font_size_override("font_size", 18)
		var paper := StyleBoxFlat.new()  # text stays legible over a tree or a duck
		paper.bg_color = Color(Ink.CREAM, 0.85)
		paper.set_content_margin_all(8)
		label.add_theme_stylebox_override("normal", paper)
		ui.add_child(label)
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
	card.position = Vector2(24, 20)
	help.add_theme_font_size_override("font_size", 13)
	var mono := SystemFont.new()
	mono.font_names = PackedStringArray(["Menlo", "Monaco", "Courier New", "monospace"])
	help.add_theme_font_override("font", mono)  # the two columns line up
	help.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT, Control.PRESET_MODE_MINSIZE, 24)
	help.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	help.grow_vertical = Control.GROW_DIRECTION_BEGIN
	toasts.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT, Control.PRESET_MODE_MINSIZE, 24)
	toasts.grow_vertical = Control.GROW_DIRECTION_BEGIN
	# a white glove with a cuff, as in the Chao gardens: a palm, four fingers and a thumb
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
	if args.has("record"):  # on the wall clock, so the garden plays at its own pace in what is saved
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
	if args.has("keys"):  # press these as a player would, with the mouse over the middle of the window: a check of the keys
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
	# The glove stands in for the cursor only over the garden: outside the window, or with the window not
	# in front, the player has their own cursor back (it used to stay hidden wherever the mouse went).
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
	var latest := ""
	while udp.get_available_packet_count() > 0:
		latest = udp.get_packet().get_string_from_utf8()
		got += 1
	if latest != "":
		var parsed = JSON.parse_string(latest)
		if parsed is Dictionary:
			var first := snap.is_empty()
			snap = parsed
			if args.has("light"):  # for looking at the night without waiting for it
				snap.light = float(args["light"])
			if args.has("asleep"):  # and at a sleeper without waiting for one
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
		orbit = Vector3(2.2, 0.5, 1.7 * size)  # from the open front of the lawn, looking into the bowl
	var eye := Vector3(size / 2, 0, -size / 2) + Vector3(cos(orbit.x), 0, sin(orbit.x)) * orbit.z
	var summit := Scenery.build(self, snap, falls, eye)
	# a white flag on the cliff rock nearest the viewer, flying the way the breeze goes (it is the breeze the ducks smell by)
	Ink.part(self, Ink.cone(0.035, 1.5, 0.028, 5), Scenery.WOOD, summit + Vector3(0, 0.7, 0), Vector3.ONE, 7.0)
	flag.position = summit + Vector3(0, 1.22, 0)
	add_child(flag)
	Ink.part(flag, BoxMesh.new(), Color.WHITE, Vector3(0.42, 0, 0), Vector3(0.84, 0.46, 0.02), 7.0, 1.0, true)  # no screen on it: a white flag is white
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
	# Everything lying in the garden, each kind made by its own recipe.
	_props("food", snap.food, _fruit)
	_props("hats", snap.get("hats", []), func(n: Node3D, h: Array) -> void:
		var lying := Hats.make(int(h[2]))
		lying.scale = Vector3.ONE * 1.9  # the size it is on a duck's head, which is drawn that much over life
		n.add_child(lying))
	_props("balls", snap.get("balls", []), func(n: Node3D, b: Array) -> void:
		var ball := Ink.part(n, Ink.ball(0.06 * 1.9, 8), Color("f6f1e6"), Vector3(0, 0.06 * 1.9, 0), Vector3.ONE, 5.0, -1.0, true)
		ball.material_override.set_shader_parameter("lift", 0.3)
		Ink.part(ball, Ink.ball(0.06 * 1.9 * 1.01, 8), Color("e2483d"), Vector3.ZERO, Vector3(1.0, 0.34, 1.0), 5.0).material_override.set_shader_parameter("lift", 0.3))
	for n in props.get("balls", []):  # it rolls: turned by how far it has come
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
		Ink.part(n, Ink.cone(0.22, 0.002, 0.22, 10), Ink.MUSTARD, Vector3(0, 0.003, 0), Vector3.ONE, 5.0, 0.3))
	_props("music", [snap.music] if snap.music != null else [], func(n: Node3D, f: Array) -> void:
		Ink.part(n, BoxMesh.new(), Ink.CORAL, Vector3(0, 0.05, 0), Vector3(0.14, 0.1, 0.14), 7.0, -1.0, true)
		Ink.part(n, Ink.cone(0.02, 0.16, 0.1, 8), Ink.MUSTARD, Vector3(0.03, 0.18, 0), Vector3.ONE, 7.0, -1.0, true).rotation.z = -0.5)
	for n in props.get("music", []):
		n.scale = Vector3.ONE * (1.0 + 0.06 * calm * sin(Time.get_ticks_msec() / 90.0))  # it plays


func _show_weather() -> void:
	# The flag flies the way the breeze goes and hangs without one; the waterfall shimmers.
	var wind = snap.get("wind")
	var blowing: bool = wind != null and Vector2(wind[0], wind[1]).length() > 0.05
	var flutter := sin(Time.get_ticks_msec() / 160.0) * 0.14 * calm
	flag.rotation.y = lerp_angle(flag.rotation.y, (atan2(wind[1], wind[0]) if blowing else flag.rotation.y) + flutter, 0.08)
	flag.rotation.z = lerp(flag.rotation.z, 0.0 if blowing else -1.35, 0.05)  # no wind, and it hangs
	for i in falls.size():
		falls[i].scale.z = 0.5 * (1.0 + 0.12 * calm * sin(Time.get_ticks_msec() / 130.0 + i * 1.7))


func _show_held() -> void:
	# What the hand holds rides up in it.
	var held = snap.get("held")
	for kind in ["balls", "hats", "food", "music", "drum"]:
		var nodes: Array = props.get(kind, [])
		for k in nodes.size():
			var up: bool = held != null and held[0] + ("s" if held[0] in ["ball", "hat"] else "") == kind and int(held[1]) == k
			nodes[k].position.y = lerp(nodes[k].position.y, 0.3 if up else 0.0, 0.3)
	for i in ducks.size():
		ducks[i].lifted = held != null and held[0] == "duck" and int(held[1]) == i


func _ask_for_brain() -> void:
	# Ask for the selected duck's brain, and again every second in case the garden was restarted.
	asked += 0.02
	if watching != selected or asked > 1.0:
		watching = selected
		asked = 0.0
		_act("garden.watch", {"duck": selected})
	brain.visible = selected >= 0 and snap.has("brain") and snap.brain.duck == selected
	if brain.visible:
		brain.show_brain(snap.brain, snap.ducks[selected].name)
		brain.position = Vector2(get_viewport().get_visible_rect().size.x - brain.size.x - 20.0, 20.0)


func _show_words() -> void:
	# The card, the bars, the news and the help: all the writing on the window.
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
	help.visible = true
	help.text = HELP_RIDING if possessing else (HELP if help_on else HELP_HINT)
	var open: bool = help_on and not possessing
	help.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT if open else Control.PRESET_BOTTOM_RIGHT,
		Control.PRESET_MODE_MINSIZE, 0 if open else 24)
	help.add_theme_font_size_override("font_size", 17 if open else 13)
	for other in [card, bars, toasts, brain]:  # the menu is the whole window while it is open
		other.modulate.a = 0.0 if open else 1.0
	if selected >= 0:
		card.text = _card_text(snap.ducks[selected])


func _card_text(d: Dictionary) -> String:
	var text := "%s   %s\n%s" % [d.name, d.label, ("asleep" if d.asleep else ("crying" if d.get("crying", false) else d.mood))]
	var a: Dictionary = d.get("among", {})
	if a.is_empty():
		return text
	text += "\n\nlikes %s best\n%s" % [a.favourite, a.hand]
	if a.friend != "":
		text += "\nfriend: %s" % a.friend
	if a.grudge != "":
		text += "\ngrudge against: %s" % a.grudge
	return text + "\nswimmer %d%%   runner %d%%   dancer %d%%" % [a.skills[0] * 100, a.skills[1] * 100, a.skills[2] * 100]


func _fruit(n: Node3D, f: Array) -> void:
	# An orange, an apple or a banana (Chris, 2026-09-21), as the garden dealt it. All of them are the same food,
	# and each duck has one it likes best.
	var kind: int = int(f[2])  # the garden says which
	var leaf := Color("2f8f4a")
	if kind == 0:
		Ink.part(n, Ink.ball(0.075, 8), Color("f39a2b"), Vector3(0, 0.072, 0), Vector3.ONE, 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.3)
		Ink.part(n, Ink.ball(0.018, 5), leaf, Vector3(0, 0.148, 0), Vector3(1.6, 0.5, 1.0), 6.0)
	elif kind == 1:
		Ink.part(n, Ink.ball(0.075, 8), Color("d8342c"), Vector3(0, 0.068, 0), Vector3(1.0, 0.9, 1.0), 6.0, -1.0, true).material_override.set_shader_parameter("lift", 0.3)
		Ink.part(n, Ink.cone(0.006, 0.045, 0.006, 4), Color("6b4326"), Vector3(0, 0.15, 0), Vector3.ONE, 6.0)
		Ink.part(n, Ink.ball(0.022, 5), leaf, Vector3(0.03, 0.155, 0), Vector3(1.6, 0.4, 0.9), 6.0)
	else:
		for k in 5:  # a curve of short pieces, fat in the middle, lying on its side
			var u := (k - 2) / 2.0
			var piece := Ink.part(n, Ink.cone(0.03 - 0.008 * abs(u), 0.062, 0.03 - 0.008 * abs(u), 6), Color("f4d23c"),
				Vector3(u * 0.075, 0.035 + 0.03 * u * u, 0), Vector3.ONE, 6.0, -1.0, true)
			piece.rotation.z = PI / 2 - u * 0.55
			piece.material_override.set_shader_parameter("lift", 0.3)
	n.rotation.y = f[0] * 7.0 + f[1] * 3.0


func _props(kind: String, items: Array, make: Callable) -> void:
	# one node per thing of this kind, remade only when how many there are changes; then put where they are
	var nodes: Array = props.get(kind, [])
	if nodes.size() != items.size():
		for n in nodes:
			n.queue_free()
		nodes = []
		for item in items:
			var n := Node3D.new()
			add_child(n)
			make.call(n, item)
			nodes.append(n)
		props[kind] = nodes
	for i in items.size():
		nodes[i].position = Vector3(items[i][0], nodes[i].position.y, -items[i][1])  # its height is the hand's business


func _draw_bars() -> void:
	# needs fill towards coral as they press, moods in mustard, wants in teal: three short lists, a bar each
	if selected < 0 or snap.is_empty() or possessing:
		return
	var rows: Array = snap.ducks[selected].get("readout", [])
	if rows.is_empty():
		return
	var tints := {"needs": Ink.CORAL, "moods": Ink.MUSTARD, "wants": Ink.TEAL}
	var font := ThemeDB.fallback_font
	var sections := 0
	var last := ""
	for row in rows:
		sections += int(row[0] != last)
		last = row[0]
	bars.draw_rect(Rect2(0, 0, 300, rows.size() * 19.0 + sections * 24.0 + 10.0), Color(Ink.CREAM, 0.88))
	var y := 6.0
	last = ""
	for row in rows:
		if row[0] != last:
			last = row[0]
			bars.draw_string(font, Vector2(10, y + 15.0), last, HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Ink.NAVY)
			y += 24.0
		bars.draw_string(font, Vector2(10, y + 12.0), row[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Ink.NAVY)
		bars.draw_rect(Rect2(90, y + 2.0, 170, 11), Color(Ink.NAVY, 0.18))
		bars.draw_rect(Rect2(90, y + 2.0, 170.0 * float(row[2]), 11), tints.get(last, Ink.NAVY))
		bars.draw_string(font, Vector2(266, y + 12.0), "%d" % int(row[2] * 100.0), HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Ink.NAVY)
		y += 19.0


func _draw_overlay() -> void:
	# What the ridden duck's brain is given and what it asks for: each eye's 721 columns as the fly's
	# lattice, left eye on the left, and the six descending readouts as bars. The duck's brain sees this
	# and never the picture behind it.
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
	overlay.draw_rect(Rect2(left - 10.0, y - 10.0, 330.0, 22.0 * ride.dn.size() + 14.0), Color(Ink.CREAM, 0.9))
	for bar in ride.dn:
		overlay.draw_rect(Rect2(left, y, 160, 12), Color(Ink.NAVY, 0.25))
		overlay.draw_rect(Rect2(left, y, 160.0 * clamp(bar[1] / 5.0, 0.0, 1.0), 12), Ink.CORAL)
		overlay.draw_string(ThemeDB.fallback_font, Vector2(left + 170.0, y + 12.0), "%s %.1f Hz" % [bar[0], bar[1]], HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Ink.NAVY)
		y += 22.0


func _quack(duck: int, tag: String) -> void:
	# A quack is a buzz whose pitch moves, shaped by the tag and voiced by the duck. Each duck has its own
	# voice, made from its number so it is the same every day: how high, how reedy, how much it wobbles and
	# how fast it speaks. Synthesised here; the garden ships no audio files.
	if tag == "drum":
		_thump()
		return
	var shape: Array = {"alarm": [900.0, 0.5, 0.12, 3], "greet": [520.0, 0.8, 0.16, 2], "inquire": [480.0, 1.3, 0.22, 1],
		"peck": [700.0, 0.9, 0.05, 2], "chirp": [1100.0, 1.1, 0.07, 2], "coo": [330.0, 0.9, 0.4, 1],
		"wheee": [600.0, 1.8, 0.45, 1]}.get(tag, [500.0, 0.8, 0.15, 1])
	var voice_rng := RandomNumberGenerator.new()
	voice_rng.seed = 9001 + duck * 7919
	var high: float = [0.62, 0.8, 1.0, 1.22, 1.5][duck % 5] * voice_rng.randf_range(0.94, 1.06)  # spread wide, then jittered
	var reedy: float = voice_rng.randf_range(0.0, 1.0)  # 0 a soft hum, 1 a buzz
	var wobble: float = voice_rng.randf_range(0.0, 0.06)
	var wobble_hz: float = voice_rng.randf_range(9.0, 22.0)
	var pace: float = voice_rng.randf_range(0.8, 1.3)
	var playback: AudioStreamGeneratorPlayback = voice.get_stream_playback()
	var phase := 0.0
	for rep in shape[3]:
		var n := int(shape[2] * pace * 22050.0)
		for k in n + int(900 * pace):  # and a gap
			if playback.get_frames_available() < 1:
				return
			var u := float(k) / n
			var f: float = shape[0] * high * lerp(1.0, float(shape[1]), u) * (1.0 + wobble * sin(TAU * wobble_hz * k / 22050.0))
			phase += f / 22050.0
			var saw := fmod(phase, 1.0) * 2.0 - 1.0
			var v: float = lerp(sin(TAU * phase), saw, reedy) * sin(min(u, 1.0) * PI) * 0.25 if k < n else 0.0
			playback.push_frame(Vector2(v, v))


func _thump() -> void:
	# The drum is not a voice: a small soft thump, the same from any duck, and quiet (Chris).
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
	# The glove rides over the lawn under the mouse, lower when it is closed. While it is closed the garden is
	# told where it is, so what it holds goes with it and the ducks can see it coming.
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
		cam.fov = 100.0  # wide and a little coarse, to say "not human"; the halftone mostly off, since at this size it hid the garden
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
	# the camera turns about the garden's middle, or about the selected duck, easing from one to the other
	var want: Vector3 = ducks[selected].position + Vector3(0, 0.15, 0) if selected >= 0 and selected < ducks.size() else Vector3(size / 2, 0.1, -size / 2)
	centre = want if centre == Vector3.INF else centre.lerp(want, min(1.0, 3.0 * dt))
	cam.position = centre + Vector3(cos(yaw) * cos(orbit.y), sin(orbit.y), sin(yaw) * cos(orbit.y)) * orbit.z
	cam.look_at(centre)


func _unhandled_input(e: InputEvent) -> void:
	if e is InputEventMouseButton:
		if e.button_index == MOUSE_BUTTON_WHEEL_UP or e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			orbit.z = clamp(orbit.z * (0.92 if e.button_index == MOUSE_BUTTON_WHEEL_UP else 1.08), 2.0, 12.0)
		elif e.button_index == MOUSE_BUTTON_LEFT and hand_mode:  # the hand: down closes it on whatever is there, up opens it
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
				if quick:  # a tap and not a carry: select, shake the tree, step the volume
					_click(e.position)
		elif e.button_index == MOUSE_BUTTON_LEFT:  # no hand: the left button turns the camera, and a tap is a click
			dragging = e.pressed
			if e.pressed:
				dragged = 0.0
			elif dragged < 6.0:
				_click(e.position)
		elif e.button_index == MOUSE_BUTTON_RIGHT or e.button_index == MOUSE_BUTTON_MIDDLE:  # the camera
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


func _click(screen: Vector2) -> void:
	var xy = _ground(screen)
	if xy == null or snap.is_empty():
		return
	for i in snap.ducks.size():
		if xy.distance_to(Vector2(snap.ducks[i].x, snap.ducks[i].y)) < 0.25:
			selected = i
			return
	if snap.music != null and xy.distance_to(Vector2(snap.music[0], snap.music[1])) < 0.4:  # the box: its volume, a step a click
		var volume: float = music.step_volume()
		_act("garden.volume", {"level": volume})
		_say("music off" if volume == 0.0 else "music volume %d%%" % int(volume * 100))
		return
	if xy.distance_to(Vector2(snap.tree[0], snap.tree[1])) < 0.3:
		_act("garden.shake_tree", {})
	else:
		selected = -1  # a click on bare ground lets go of the duck; food is F, so a stray click feeds nobody


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
	elif code == KEY_F:  # fruit falls from the tree, as it does by itself: an orange, an apple or a banana, under the canopy
		_act("garden.shake_tree", {})
	elif code == KEY_M:
		var xy = _mouse_ground()
		if xy != null:
			_act("garden.music", {"x": xy.x, "y": xy.y, "on": int(snap.music == null)})
	elif selected >= 0 and code == KEY_G:  # a fruit held out to this duck: it learns your hand is a good thing
		_act("garden.give", {"duck": selected})
	elif selected >= 0 and code == KEY_P:
		_act("garden.pet", {"duck": selected})
	elif code == KEY_D and not possessing:  # a drum, down at the mouse or taken up again (D steers a ridden duck)
		var here = _mouse_ground()
		if here != null:
			_act("garden.drum", {"x": here.x, "y": here.y, "on": int(snap.get("drum") == null)})
	elif code == KEY_B:  # a ball for them, where the mouse is
		var where = _mouse_ground()
		if where != null and where.x > 0 and where.y > 0 and where.x < snap.size and where.y < snap.size:
			_act("garden.drop_ball", {"x": where.x, "y": where.y})
	elif code == KEY_H:  # the hand, on and off
		hand_mode = not hand_mode
		if not hand_mode and gripping:
			gripping = false
			_let_go()
		_say("the hand is out: hold the left button to pick things up" if hand_mode else "the hand is away")
	elif code == KEY_T:  # a hat, no two alike, left where the mouse is; the ducks decide who wears it
		var spot = _mouse_ground()
		if spot != null:
			_act("garden.drop_hat", {"x": spot.x, "y": spot.y})


func _mouse_ground():
	return _ground(get_viewport().get_mouse_position())


func _say(line: String) -> void:
	# a line from this window, shown with the garden's toasts for a moment
	said = line
	said_until = Time.get_ticks_msec() / 1000.0 + 2.5


func _act(method: String, params: Dictionary) -> void:
	out.put_packet(JSON.stringify({"method": method, "params": params}).to_utf8_buffer())
