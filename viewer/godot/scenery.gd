# The garden's setting, after the Chao gardens of Sonic Adventure 2 (Chris, 2026-09-21): a lawn cradled in
# tall rounded rock with grass on top, a waterfall of pale stone columns stepping down to the pond, a dark cave
# mouth, palms, a rail fence along the open front, blue sea and sky. Inspiration for forms, placement and
# colour; every shape here is made by rule, from lumps.
#
# None of it is the world's: the cliffs stand outside the garden's square and the fence runs along it, so
# both only draw the walls the ducks already have. The waterfall is built only where the pond lies near the
# back corner, since in any other garden it would run across the lawn (it did, in a 4 m garden: 2026-09-21).
extends RefCounted

const Ink := preload("res://ink.gd")
const LAWN := Color("6fc24a")
const ROCK := Color("a3978a")
const PALE_ROCK := Color("ddd5c6")
const WATER := Color("4f9fe0")
const SEA := Color("3b7fd4")
const FROND := Color("1f7a4d")
const WOOD := Color("8a5a3c")
const SKY_DAY := Color("a9d8f0")
const SKY_NIGHT := Color("1d2a4d")


static func lump(radius: float, height: float, rng: RandomNumberGenerator, sides := 9, rings := 4, taper := 0.8, wobble := 0.2, dome := 0.06) -> Array:
	# A rounded column of rock: rings of vertices pushed in and out, narrowing upward, with a domed cap.
	# Returns [sides mesh, cap mesh].
	var lean := []
	for j in sides:
		lean.append(rng.randf_range(-wobble, wobble))
	var rock := SurfaceTool.new()
	rock.begin(Mesh.PRIMITIVE_TRIANGLES)
	var ring_points := []
	for k in rings + 1:
		var u := float(k) / rings
		var points := []
		for j in sides:
			var r: float = radius * lerp(1.0, taper, u * u) * (1.0 + lean[j] + rng.randf_range(-wobble, wobble) * 0.5)
			var a := TAU * j / sides
			points.append(Vector3(cos(a) * r, height * u, sin(a) * r))
		ring_points.append(points)
	for k in rings:
		for j in sides:
			var j2 := (j + 1) % sides
			var a: Vector3 = ring_points[k][j]
			var b: Vector3 = ring_points[k][j2]
			var c: Vector3 = ring_points[k + 1][j]
			var d: Vector3 = ring_points[k + 1][j2]
			for v in [a, b, c, b, d, c]:  # clockwise from outside, which is Godot's front
				rock.add_vertex(v)
	var cap := SurfaceTool.new()
	cap.begin(Mesh.PRIMITIVE_TRIANGLES)
	var top: Array = ring_points[rings]
	var crown := Vector3(0, height + dome * radius, 0)
	for j in sides:
		for v in [crown, top[j], top[(j + 1) % sides]]:
			cap.add_vertex(v)
	return [rock.commit(), cap.commit()]


static func place_lump(root: Node3D, at: Vector3, radius: float, height: float, rng: RandomNumberGenerator,
		stone := ROCK, top := LAWN, sides := 9, taper := 0.8) -> void:
	var meshes := lump(radius, height, rng, sides, 4, taper)
	Ink.part(root, meshes[0], stone, at, Vector3.ONE, 8.0).material_override.set_shader_parameter("lift", 0.3)
	Ink.part(root, meshes[1], top, at, Vector3.ONE, 8.0).material_override.set_shader_parameter("lift", 0.2)


static func palm(root: Node3D, at: Vector3, height: float, rng: RandomNumberGenerator) -> void:
	var trunk := Ink.part(root, Ink.cone(0.05, height, 0.035, 5), WOOD, at + Vector3(0, height / 2, 0), Vector3.ONE, 7.0)
	trunk.rotation.z = rng.randf_range(-0.15, 0.15)
	for f in 7:
		var frond := Ink.part(root, Ink.cone(0.0, 0.7, 0.11, 3), FROND, at + Vector3(0, height, 0), Vector3(1, 1, 0.35), 7.0)
		frond.rotation = Vector3(0, TAU * f / 7.0 + rng.randf_range(-0.2, 0.2), -PI / 2 - rng.randf_range(0.25, 0.6))
		frond.translate_object_local(Vector3(0, -0.35, 0))


static func build(root: Node3D, snap: Dictionary, falls: Array, viewer: Vector3) -> Vector3:
	# Returns the top of the cliff rock nearest `viewer` (where the camera starts), which is where the garden's
	# flag goes: near enough to read.
	var summit := Vector3.INF
	var size: float = snap.size
	var rng := RandomNumberGenerator.new()
	rng.seed = 11
	# the plateau: one great lump under the lawn, its top the grass, its sides falling to the sea
	var under := lump(0.75 * size, 2.0, rng, 14, 3, 1.08, 0.04, 0.0)  # flat on top: the lawn is y = 0 everywhere
	var mid := Vector3(size / 2, -2.002, -size / 2)  # a hair under the pond and the shade, which lie on it
	Ink.part(root, under[0], ROCK, mid, Vector3.ONE, 8.0).material_override.set_shader_parameter("lift", 0.22)
	Ink.part(root, under[1], LAWN, mid, Vector3.ONE, 8.0, 0.97)
	Ink.part(root, Ink.cone(60.0, 0.01, 60.0, 24), SEA, Vector3(size / 2, -2.0, -size / 2), Vector3.ONE, 10.0, 0.95)
	for far in [[-7.0, 9.0, 1.6, 3.2], [13.0, 11.0, 2.2, 4.4], [16.0, -2.0, 1.4, 2.6]]:  # stacks out in the sea
		place_lump(root, Vector3(far[0], -2.0, -far[1]), far[2], far[3] + 2.0, rng)

	# the bowl: tall lumps shoulder to shoulder behind the north and east edges, tallest in the corner
	for edge in 2:
		var along := -0.6
		while along < size + 1.0:
			var r := rng.randf_range(0.95, 1.5)
			var corner: float = clamp(1.0 - abs(size - along) / (0.6 * size), 0.0, 1.0)
			var h: float = rng.randf_range(1.7, 2.5) + 1.3 * corner
			var out: float = r * 0.78 + rng.randf_range(0.0, 0.25)
			var at := Vector3(along, -0.3, -(size + out)) if edge == 0 else Vector3(size + out, -0.3, -along)
			place_lump(root, at, r, h, rng)
			if summit == Vector3.INF or (at - viewer).length() < (summit - viewer).length():
				summit = at + Vector3(0, h, 0)
			if rng.randf() < 0.3:
				palm(root, at + Vector3(rng.randf_range(-0.3, 0.3), h - 0.25, rng.randf_range(-0.3, 0.3)), rng.randf_range(0.5, 0.9), rng)
			along += r * 1.25
	# a cave mouth in the north cliff, dark and going nowhere
	Ink.part(root, Ink.ball(0.34, 10), Ink.NAVY, Vector3(0.3 * size, 0.2, -(size + 0.05)), Vector3(1.0, 1.4, 0.6), 7.0, 0.05)

	# the rail fence along the open south and west edges: the wall the ducks already have, drawn
	for edge in 2:
		var posts := int(size / 0.75)
		for i in posts + 1:
			var t := size * i / posts
			var at := Vector3(t, 0, 0.06) if edge == 0 else Vector3(-0.06, 0, -t)
			Ink.part(root, Ink.cone(0.035, 0.34, 0.03, 5), WOOD, at + Vector3(0, 0.17, 0), Vector3.ONE, 7.0)
		for rail in [0.14, 0.27]:
			Ink.part(root, BoxMesh.new(), WOOD, Vector3(size / 2, rail, 0.06) if edge == 0 else Vector3(-0.06, rail, -size / 2),
				Vector3(size, 0.035, 0.03) if edge == 0 else Vector3(0.03, 0.035, size), 7.0)

	if snap.pond != null:
		var p: Array = snap.pond
		var centre := Vector2(p[0], p[1])
		Ink.part(root, Ink.cone(p[2], 0.004, p[2], 28), WATER, Vector3(p[0], 0.003, -p[1]), Vector3.ONE, 8.0, 0.9)
		if centre.distance_to(Vector2(size, size)) < p[2] + 2.4:
			waterfall(root, snap, centre, size, rng, falls)
		for i in int(10 * p[2] / 0.35):  # reeds round the open shore
			var a := rng.randf_range(0.0, TAU)
			var at := Vector3(p[0] + cos(a) * (p[2] + 0.05), 0.0, -p[1] - sin(a) * (p[2] + 0.05))
			if at.x > 0.1 and at.x < size - 0.4 and -at.z > 0.1 and -at.z < size - 0.4:
				var h := rng.randf_range(0.2, 0.36)
				Ink.part(root, Ink.cone(0.012, h, 0.008, 4), FROND, at + Vector3(0, h / 2, 0), Vector3.ONE, 7.0)
				Ink.part(root, Ink.cone(0.016, 0.07, 0.016, 5), WOOD, at + Vector3(0, h, 0), Vector3.ONE, 7.0)

	for i in int(70 * size * size / 16.0):  # grass tufts: the same ones every run
		var at := Vector3(rng.randf_range(0.1, size - 0.1), 0.03, -rng.randf_range(0.1, size - 0.1))
		if snap.pond == null or Vector2(at.x - snap.pond[0], -at.z - snap.pond[1]).length() > snap.pond[2] + 0.1:
			Ink.part(root, Ink.cone(0.025, 0.07, 0.0, 4), FROND, at, Vector3.ONE, 7.0)

	var tree: Array = snap.tree  # the fruit tree, the one whose shade is real
	var trunk := Vector3(tree[0], 0, -tree[1])
	Ink.part(root, Ink.cone(tree[2], 0.001, tree[2], 18), LAWN, trunk + Vector3(0, 0.004, 0), Vector3.ONE, 8.0, 0.55)
	Ink.part(root, Ink.cone(0.1, 1.0, 0.07, 6), WOOD, trunk + Vector3(0, 0.5, 0), Vector3.ONE, 7.0, -1.0, true)
	for blob in [[0.0, 1.2, 0.0, 0.6], [0.32, 1.05, 0.14, 0.42], [-0.3, 1.1, -0.18, 0.45], [0.03, 1.6, 0.04, 0.38]]:
		Ink.part(root, Ink.ball(blob[3], 7), FROND, trunk + Vector3(blob[0], blob[1], blob[2]), Vector3(1, 0.8, 1), 8.0, -1.0, true).material_override.set_shader_parameter("lift", 0.15)
	return summit


static func waterfall(root: Node3D, snap: Dictionary, centre: Vector2, size: float, rng: RandomNumberGenerator, falls: Array) -> void:
	# Pale round columns stepping down from the cliff into the pond, water on each and between them. Built only
	# where the pond lies by the back corner.
	var p: Array = snap.pond
	var corner := Vector2(size, size)
	var toward := (corner - centre).normalized()
	var side := Vector2(-toward.y, toward.x)
	# The foot of the fall is the world's own rocks, which are solid (DEMO_GARDEN), lowest first, and it
	# carries on up into the cliff from the last of them; a garden without rocks gets columns by rule.
	var steps := []  # [x, y, radius, height]
	for i in snap.get("rocks", []).size():
		var rock: Array = snap.rocks[i]
		steps.append([rock[0], rock[1], rock[2], 0.35 + 0.5 * i])
	if steps.is_empty():
		steps.append([centre.x + toward.x * (p[2] + 0.1), centre.y + toward.y * (p[2] + 0.1), 0.55, 0.35])
	while steps.size() < 5:
		var last: Array = steps[-1]
		steps.append([last[0] + toward.x * 0.6, last[1] + toward.y * 0.6, last[2] + 0.12, last[3] + 0.65])
	for i in steps.size():
		var s: Array = steps[i]
		place_lump(root, Vector3(s[0], -0.2, -s[1]), s[2], s[3] + 0.2, rng, PALE_ROCK, WATER, 10, 0.9)
		var sheet := Ink.part(root, BoxMesh.new(), WATER, Vector3(s[0], s[3] / 2, -s[1]) - Vector3(toward.x, 0, -toward.y) * (s[2] * 0.93),
			Vector3(0.05, s[3], 0.5), 8.0, 0.97)
		sheet.rotation.y = atan2(toward.y, toward.x)
		falls.append(sheet)
	for flank in [-1.0, 1.0]:  # and darker rock either side of the fall, with a palm
		var at3: Vector2 = centre + toward * (p[2] + 0.9) + side * flank * 1.25
		place_lump(root, Vector3(at3.x, -0.2, -at3.y), 0.75, 1.5, rng)
		palm(root, Vector3(at3.x, 1.25, -at3.y), 0.7, rng)
