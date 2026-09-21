# The garden, drawn from what the Python side publishes (viewer/snapshot.py): one JSON datagram a step, in.
# Whatever the player does goes back the same way, as the garden's own control calls. Nothing here decides
# anything about a duck. Run the garden with --godot and then this project; either can start first.
#
# After `--`:
#   --orbit=yaw,pitch,distance  where the camera starts       --ride=N   start on duck N's back, --select=N beside it
#   --reduced-motion            a still camera, half the animation       --mute     no sound
#   --light=0.1                 draw that daylight whatever the hour, to look at the night; --asleep=N draws duck N asleep
#   --port=N                    listen there, for a garden started with MICRO_GARDEN_PORT=N
#   --shot=file.png             save the window after --shot-after=S seconds (6 by default)
#   --feed-at=x,y               drop food there once snapshots arrive, and --quit-after=S: both for Gate 13
extends Node3D

const Ink := preload("res://ink.gd")
const Duck := preload("res://duck.gd")
const Scenery := preload("res://scenery.gd")
const Hats := preload("res://hats.gd")
const BrainView := preload("res://brainview.gd")
const SNAPSHOT_PORT := 7650  # actions go back on the next one up; --port moves the pair
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

var bars := Control.new()  # the selected duck's needs, moods and wants, as bars under its card
var brain := Control.new()  # the selected duck's brain, firing
var watching := -2  # whose brain the garden was last asked for
var asked := 0.0
var overlay := Control.new()  # the ride view's retinas and descending-neuron bars; O hides it
var card := Label.new()
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
	for label in [card, toasts]:
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
	toasts.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT, Control.PRESET_MODE_MINSIZE, 24)
	toasts.grow_vertical = Control.GROW_DIRECTION_BEGIN
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
	if args.has("quit-after"):
		get_tree().create_timer(float(args["quit-after"])).timeout.connect(get_tree().quit)


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
	_pond_sound()


func _build() -> void:
	var size: float = snap.size
	if not args.has("orbit"):
		orbit = Vector3(2.2, 0.5, 1.7 * size)  # from the open front of the lawn, looking into the bowl
	var eye := Vector3(size / 2, 0, -size / 2) + Vector3(cos(orbit.x), 0, sin(orbit.x)) * orbit.z
	var summit := Scenery.build(self, snap, falls, eye)
	# a white flag on the cliff rock nearest the viewer, flying the way the breeze goes (it is the breeze the ducks smell by)
	var pole := Ink.part(self, Ink.cone(0.035, 1.5, 0.028, 5), Scenery.WOOD, summit + Vector3(0, 0.7, 0), Vector3.ONE, 7.0)
	flag.position = summit + Vector3(0, 1.22, 0)
	add_child(flag)
	Ink.part(flag, BoxMesh.new(), Color.WHITE, Vector3(0.42, 0, 0), Vector3(0.84, 0.46, 0.02), 7.0, -1.0, true).material_override.set_shader_parameter("lift", 0.4)
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
	_props("food", snap.food, _fruit)
	_props("hats", snap.get("hats", []), func(n: Node3D, h: Array) -> void:
		var lying := Hats.make(int(h[2]))
		lying.scale = Vector3.ONE * 1.9  # the size it is on a duck's head, which is drawn that much over life
		n.add_child(lying))
	_props("danger", snap.danger, func(n: Node3D, f: Array) -> void:
		Ink.part(n, Ink.cone(0.22, 0.002, 0.22, 10), Ink.MUSTARD, Vector3(0, 0.003, 0), Vector3.ONE, 5.0, 0.3))
	_props("music", [snap.music] if snap.music != null else [], func(n: Node3D, f: Array) -> void:
		Ink.part(n, BoxMesh.new(), Ink.CORAL, Vector3(0, 0.05, 0), Vector3(0.14, 0.1, 0.14), 7.0, -1.0, true)
		Ink.part(n, Ink.cone(0.02, 0.16, 0.1, 8), Ink.MUSTARD, Vector3(0.03, 0.18, 0), Vector3.ONE, 7.0, -1.0, true).rotation.z = -0.5)
	_props("hand", [snap.hand] if snap.hand != null else [], func(n: Node3D, f: Array) -> void:
		Ink.part(n, Ink.ball(0.12, 8), Ink.CREAM, Vector3(0, 0.3, 0), Vector3(1, 0.5, 1), 7.0, -1.0, true))
	var wind = snap.get("wind")
	var blowing: bool = wind != null and Vector2(wind[0], wind[1]).length() > 0.05
	var flutter := sin(Time.get_ticks_msec() / 160.0) * 0.14 * calm
	flag.rotation.y = lerp_angle(flag.rotation.y, (atan2(wind[1], wind[0]) if blowing else flag.rotation.y) + flutter, 0.08)
	flag.rotation.z = lerp(flag.rotation.z, 0.0 if blowing else -1.35, 0.05)  # no wind, and it hangs
	for i in falls.size():
		falls[i].scale.z = 0.5 * (1.0 + 0.12 * calm * sin(Time.get_ticks_msec() / 130.0 + i * 1.7))
	for n in props.get("music", []):
		n.scale = Vector3.ONE * (1.0 + 0.06 * calm * sin(Time.get_ticks_msec() / 90.0))  # it plays
	# ask for the selected duck's brain, and again every second in case the garden was restarted
	asked += 0.02
	if watching != selected or asked > 1.0:
		watching = selected
		asked = 0.0
		_act("garden.watch", {"duck": selected})
	brain.visible = selected >= 0 and snap.has("brain") and snap.brain.duck == selected
	if brain.visible:
		brain.show_brain(snap.brain, snap.ducks[selected].name)
		brain.position = Vector2(get_viewport().get_visible_rect().size.x - brain.size.x - 20.0, 20.0)
	bars.position = card.position + Vector2(0, card.size.y + 8.0)
	bars.queue_redraw()
	toasts.text = "\n".join(snap.toasts)
	toasts.visible = not snap.toasts.is_empty()
	overlay.queue_redraw()
	for q in snap.get("sounds", []):  # [t, duck, tag], oldest first
		if q[0] > heard:
			heard = q[0]
			_quack(int(q[1]), q[2])
	for i in ducks.size():
		ducks[i].model.visible = not (possessing and i == selected)
	if possessing:
		_act("garden.wheel", {"duck": selected, "fwd": int(Input.is_key_pressed(KEY_W)) - int(Input.is_key_pressed(KEY_S)),
			"turn": int(Input.is_key_pressed(KEY_A)) - int(Input.is_key_pressed(KEY_D))})
	card.text = "click a duck, or the tree to shake it\nF  food      H  a hat      M  the music box, down or up  (all at the mouse)      C  clap"
	if selected >= 0:
		var d: Dictionary = snap.ducks[selected]
		card.text = "%s  %s\n%s\n\nTab  ride it: W A S D steer, O hides its eyes\nP  pet\nF  food      H  a hat      M  the music box, down or up  (all at the mouse)      C  clap" % [
			d.name, d.label, ("asleep" if d.asleep else d.mood)]


func _fruit(n: Node3D, f: Array) -> void:
	# An orange, an apple or a banana (Chris, 2026-09-21). Which one is drawn from where it lies, so a fruit
	# stays the fruit it is while others fall and are eaten around it. All of them are the same food.
	var kind := int(abs(f[0] * 731.0 + f[1] * 389.0) * 10.0) % 3
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
		nodes[i].position = Vector3(items[i][0], 0, -items[i][1])


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
	# A quack is a buzz whose pitch falls, shaped by the tag and set by the duck, so five ducks have five
	# voices and an alarm does not sound like a coo. Synthesised here; the garden ships no audio files.
	var shape: Array = {"alarm": [900.0, 0.5, 0.12, 3], "greet": [520.0, 0.8, 0.16, 2], "inquire": [480.0, 1.3, 0.22, 1],
		"peck": [700.0, 0.9, 0.05, 2], "chirp": [1100.0, 1.1, 0.07, 2], "coo": [330.0, 0.9, 0.4, 1],
		"wheee": [600.0, 1.8, 0.45, 1]}.get(tag, [500.0, 0.8, 0.15, 1])
	var playback: AudioStreamGeneratorPlayback = voice.get_stream_playback()
	var pitch: float = shape[0] * (0.8 + 0.1 * duck)
	var phase := 0.0
	for rep in shape[3]:
		var n := int(shape[2] * 22050.0)
		for k in n + 900:  # and a gap
			if playback.get_frames_available() < 1:
				return
			var u := float(k) / n
			phase += pitch * lerp(1.0, float(shape[1]), u) / 22050.0
			var v := (fmod(phase, 1.0) * 2.0 - 1.0) * sin(min(u, 1.0) * PI) * 0.25 if k < n else 0.0
			playback.push_frame(Vector2(v, v))


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
		cam.fov = 100.0  # wide and coarse, to say "not human"
		get_viewport().scaling_3d_scale = 0.3
		cam.position = d.position + Vector3(0, 0.3, 0)
		cam.rotation = Vector3(-0.15, d.rotation.y - PI / 2, 0)  # a duck faces +x and a camera looks down -z
		return
	cam.fov = 38.0
	get_viewport().scaling_3d_scale = 1.0
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
		elif e.button_index == MOUSE_BUTTON_LEFT:
			if e.pressed:
				dragging = true; dragged = 0.0
			else:
				dragging = false
				if dragged < 6.0:
					_click(e.position)
	elif e is InputEventMouseMotion and dragging:
		dragged += e.relative.length()
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
	if xy.distance_to(Vector2(snap.tree[0], snap.tree[1])) < 0.3:
		_act("garden.shake_tree", {})
	else:
		selected = -1  # a click on bare ground lets go of the duck; food is F, so a stray click feeds nobody


func _key(code: int) -> void:
	if code == KEY_TAB and (possessing or selected >= 0):
		possessing = not possessing
		if not possessing:
			_act("garden.wheel", {"duck": -1, "fwd": 0, "turn": 0})
	elif code == KEY_O:
		overlay.visible = not overlay.visible
	elif code == KEY_C:
		_act("garden.scare", {})
	elif code == KEY_F:
		var at = _ground(get_viewport().get_mouse_position())
		if at != null and at.x > 0 and at.y > 0 and at.x < snap.size and at.y < snap.size:
			_act("garden.hand", {"x": at.x, "y": at.y, "feed": 3})
	elif code == KEY_M:
		var xy = _ground(get_viewport().get_mouse_position())
		if xy != null:
			_act("garden.music", {"x": xy.x, "y": xy.y, "on": int(snap.music == null)})
	elif selected >= 0 and code == KEY_P:
		_act("garden.pet", {"duck": selected})
	elif code == KEY_H:  # a hat, no two alike, left where the mouse is; the ducks decide who wears it
		var spot = _ground(get_viewport().get_mouse_position())
		if spot != null:
			_act("garden.drop_hat", {"x": spot.x, "y": spot.y})

func _act(method: String, params: Dictionary) -> void:
	out.put_packet(JSON.stringify({"method": method, "params": params}).to_utf8_buffer())
