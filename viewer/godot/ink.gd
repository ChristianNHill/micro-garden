# The palette and the two materials everything in the garden is printed with (ARCHITECTURE.md section 4).
extends RefCounted

const NAVY := Color("1c2a4d")
const CREAM := Color("f4ead5")
const CORAL := Color("ee6f5c")
const TEAL := Color("2f9c8f")
const MUSTARD := Color("e6b54a")
const GRASS := Color("b9c46a")  # mustard pulled towards teal: the ground, so the five inks stay for things

const PRINT := preload("res://print.gdshader")
const OUTLINE := preload("res://outline.gdshader")


static func material(colour: Color, cell := 7.0, tone := -1.0, outlined := false) -> ShaderMaterial:
	var m := ShaderMaterial.new()
	m.shader = PRINT
	m.set_shader_parameter("base", colour)
	m.set_shader_parameter("ink", NAVY)
	m.set_shader_parameter("cell", cell)
	m.set_shader_parameter("tone_fixed", tone)
	if outlined:
		var line := ShaderMaterial.new()
		line.shader = OUTLINE
		line.set_shader_parameter("ink", NAVY)
		m.next_pass = line
	return m


static func part(parent: Node3D, mesh: Mesh, colour: Color, at := Vector3.ZERO, size := Vector3.ONE,
		cell := 7.0, tone := -1.0, outlined := false) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	node.mesh = mesh
	node.material_override = material(colour, cell, tone, outlined)
	node.position = at
	node.scale = size
	parent.add_child(node)
	return node


static func ball(radius: float, segments := 8) -> SphereMesh:
	var m := SphereMesh.new()
	m.radius = radius
	m.height = 2 * radius
	m.radial_segments = segments
	m.rings = segments / 2
	return m


static func cone(bottom: float, height: float, top := 0.0, segments := 6) -> CylinderMesh:
	var m := CylinderMesh.new()
	m.bottom_radius = bottom
	m.top_radius = top
	m.height = height
	m.radial_segments = segments
	m.rings = 1
	return m
