# The selected duck's brain, front on: 12,000 of its 139,000 neurons where FlyWire has them, each a dot in its
# group's colour that flares when it fires and fades over a third of a second. The garden picks the neurons,
# writes their places to a file once (brain/brainview.py) and then sends only which ones spiked.
extends Control

const Ink := preload("res://ink.gd")
const WIDTH := 600.0
const FADE_S := 0.35
const DIM := [Color("3f6fb0"), Color("b0428f"), Color("3f9a62"), Color("c9772e"), Color("77798a")]
const LIT := [Color("bfe0ff"), Color("ffb3ec"), Color("c6ffd6"), Color("ffe08a"), Color("ffffff")]

var dots := MultiMeshInstance2D.new()
var title := Label.new()
var names: Array = []
var group: Array = []
var glow := {}  # dot -> how lit, 1 down to 0
var loaded := ""
var height := 200.0


func _ready() -> void:
	mouse_filter = Control.MOUSE_FILTER_IGNORE
	dots.multimesh = MultiMesh.new()
	dots.multimesh.use_colors = true
	var dot := QuadMesh.new()
	dot.size = Vector2(3.4, 3.4)
	dots.multimesh.mesh = dot
	add_child(dots)
	title.add_theme_color_override("font_color", Ink.CREAM)
	title.add_theme_font_size_override("font_size", 14)
	add_child(title)
	title.position = Vector2(12, 8)


func load_points(path: String) -> void:
	var data = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not data is Dictionary:
		return
	loaded = path
	group = data.group
	var ys: Array = data.y
	height = WIDTH * float(ys.max()) + 70.0
	dots.multimesh.instance_count = 0
	dots.multimesh.instance_count = group.size()
	for k in group.size():
		dots.multimesh.set_instance_transform_2d(k, Transform2D(0.0, Vector2(12.0 + data.x[k] * (WIDTH - 24.0), 34.0 + data.y[k] * (WIDTH - 24.0))))
		dots.multimesh.set_instance_color(k, DIM[group[k]])
	names = data.groups
	title.set_meta("of", int(data.of))
	custom_minimum_size = Vector2(WIDTH, height)
	size = custom_minimum_size
	queue_redraw()


func show_brain(brain: Dictionary, duck_name: String) -> void:
	if brain.points != loaded:
		load_points(brain.points)
	title.text = "%s's brain: %d of its %d neurons, as they fire" % [duck_name, group.size(), title.get_meta("of", 0)]
	for k in brain.spikes:
		glow[int(k)] = 1.0


func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(WIDTH, height)), Color(Ink.NAVY, 0.92))
	var x := 12.0  # the legend, each group's name in its own colour
	for k in names.size():
		var word: String = names[k]
		draw_string(ThemeDB.fallback_font, Vector2(x, height - 12.0), word, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, LIT[k].lerp(DIM[k], 0.35))
		x += ThemeDB.fallback_font.get_string_size(word, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 12.0


func _process(dt: float) -> void:
	for k in glow.keys():
		var g: float = glow[k] - dt / FADE_S
		if g <= 0.0:
			glow.erase(k)
			dots.multimesh.set_instance_color(k, DIM[group[k]])
		else:
			glow[k] = g
			dots.multimesh.set_instance_color(k, DIM[group[k]].lerp(LIT[group[k]], g))
