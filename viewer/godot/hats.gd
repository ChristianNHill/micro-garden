# A hat from its style number: the same number always makes the same hat, on the lawn or on a head.
# Fifteen shapes, in a colour and a trim of their own that the number picks at random.
extends RefCounted

const Ink := preload("res://ink.gd")
const SHAPES := 15


static func make(style: int) -> Node3D:
	var hat := Node3D.new()
	var dice := RandomNumberGenerator.new()
	dice.seed = style
	var main := Color.from_hsv(dice.randf(), dice.randf_range(0.45, 0.85), dice.randf_range(0.7, 0.95))
	var band := Color.from_hsv(fposmod(main.h + dice.randf_range(0.25, 0.75), 1.0), dice.randf_range(0.35, 0.8), dice.randf_range(0.75, 0.98))
	var part := func(mesh: Mesh, colour: Color, at: Vector3, size := Vector3.ONE, lined := false) -> MeshInstance3D:
		return Ink.part(hat, mesh, colour, at, size, 5.0, -1.0, lined)
	match style % SHAPES:
		0:  # a party cone with a bobble
			part.call(Ink.cone(0.045, 0.1, 0.0, 8), main, Vector3(0, 0.05, 0), Vector3.ONE, true)
			part.call(Ink.ball(0.014, 6), band, Vector3(0, 0.105, 0))
		1:  # a top hat
			part.call(Ink.cone(0.07, 0.008, 0.07, 10), main, Vector3(0, 0.004, 0), Vector3.ONE, true)
			part.call(Ink.cone(0.042, 0.085, 0.045, 10), main, Vector3(0, 0.05, 0), Vector3.ONE, true)
			part.call(Ink.cone(0.044, 0.016, 0.044, 10), band, Vector3(0, 0.02, 0))
		2:  # a beanie with a pompom
			part.call(Ink.ball(0.05, 8), main, Vector3(0, 0.012, 0), Vector3(1, 0.85, 1), true)
			part.call(Ink.cone(0.052, 0.018, 0.052, 10), band, Vector3(0, 0.008, 0))
			part.call(Ink.ball(0.017, 6), band, Vector3(0, 0.062, 0))
		3:  # a crown
			part.call(Ink.cone(0.045, 0.04, 0.045, 10), main, Vector3(0, 0.02, 0), Vector3.ONE, true)
			for k in 5:
				var a := TAU * k / 5.0
				part.call(Ink.cone(0.013, 0.035, 0.0, 4), main, Vector3(cos(a) * 0.036, 0.055, sin(a) * 0.036))
				part.call(Ink.ball(0.008, 5), band, Vector3(cos(a) * 0.036, 0.076, sin(a) * 0.036))
		4:  # a sun hat
			part.call(Ink.cone(0.095, 0.008, 0.1, 12), main, Vector3(0, 0.004, 0), Vector3.ONE, true)
			part.call(Ink.ball(0.042, 8), main, Vector3(0, 0.01, 0), Vector3(1, 0.8, 1), true)
			part.call(Ink.cone(0.044, 0.014, 0.043, 10), band, Vector3(0, 0.014, 0))
		5:  # a cowboy hat: a wide brim turned up at the sides, and a creased crown
			part.call(Ink.cone(0.1, 0.008, 0.1, 12), main, Vector3(0, 0.006, 0), Vector3(1, 1, 0.75), true)
			for side in [-1.0, 1.0]:
				part.call(Ink.cone(0.03, 0.006, 0.03, 8), main, Vector3(side * 0.085, 0.02, 0), Vector3(1, 1, 2.2)).rotation.z = side * 0.9
			part.call(Ink.cone(0.04, 0.06, 0.046, 10), main, Vector3(0, 0.038, 0), Vector3(1, 1, 0.85), true)
			part.call(Ink.cone(0.047, 0.012, 0.047, 10), band, Vector3(0, 0.016, 0), Vector3(1, 1, 0.86))
		6:  # a bowler: a round dome and a narrow rolled brim
			part.call(Ink.ball(0.045, 10), main, Vector3(0, 0.02, 0), Vector3(1, 0.95, 1), true)
			part.call(Ink.cone(0.062, 0.01, 0.062, 12), main, Vector3(0, 0.006, 0))
			part.call(Ink.cone(0.047, 0.012, 0.047, 10), band, Vector3(0, 0.015, 0))
		7:  # a beret, worn to one side, with a stalk
			var beret: MeshInstance3D = part.call(Ink.ball(0.06, 10), main, Vector3(0.01, 0.016, 0), Vector3(1, 0.35, 1), true)
			beret.rotation.z = -0.25
			part.call(Ink.cone(0.006, 0.02, 0.004, 5), band, Vector3(0.005, 0.042, 0))
		8:  # a chef's hat: a band and a tall puff
			part.call(Ink.cone(0.045, 0.035, 0.047, 10), Ink.CREAM, Vector3(0, 0.018, 0), Vector3.ONE, true)
			part.call(Ink.ball(0.058, 10), Ink.CREAM, Vector3(0, 0.07, 0), Vector3(1, 0.8, 1), true)
			part.call(Ink.cone(0.047, 0.01, 0.047, 10), main, Vector3(0, 0.008, 0))
		9:  # a cap with its peak to the front
			part.call(Ink.ball(0.047, 10), main, Vector3(0, 0.01, 0), Vector3(1, 0.8, 1), true)
			part.call(BoxMesh.new(), band, Vector3(0.055, 0.008, 0), Vector3(0.06, 0.006, 0.07))
			part.call(Ink.ball(0.008, 5), band, Vector3(0, 0.048, 0))
		10:  # a witch's hat: a wide brim and a tall cone with a bend in it
			part.call(Ink.cone(0.09, 0.008, 0.09, 12), main, Vector3(0, 0.004, 0), Vector3.ONE, true)
			part.call(Ink.cone(0.042, 0.075, 0.022, 8), main, Vector3(0, 0.042, 0), Vector3.ONE, true)
			part.call(Ink.cone(0.022, 0.05, 0.0, 6), main, Vector3(-0.012, 0.098, 0), Vector3.ONE, true).rotation.z = 0.5
			part.call(Ink.cone(0.044, 0.014, 0.043, 10), band, Vector3(0, 0.014, 0))
		11:  # a propeller beanie
			part.call(Ink.ball(0.048, 10), main, Vector3(0, 0.01, 0), Vector3(1, 0.8, 1), true)
			part.call(Ink.cone(0.004, 0.03, 0.004, 5), Ink.NAVY, Vector3(0, 0.058, 0))
			for side in [-1.0, 1.0]:
				part.call(BoxMesh.new(), band, Vector3(side * 0.028, 0.074, 0), Vector3(0.05, 0.004, 0.014)).rotation.x = side * 0.3
		12:  # a flower crown: blossoms round a ring of green
			part.call(Ink.cone(0.047, 0.01, 0.047, 12), Color("2fa36b"), Vector3(0, 0.008, 0))
			for k in 7:
				var a := TAU * k / 7.0
				part.call(Ink.ball(0.014, 6), main if k % 2 == 0 else band, Vector3(cos(a) * 0.047, 0.016, sin(a) * 0.047))
				part.call(Ink.ball(0.005, 4), Ink.MUSTARD, Vector3(cos(a) * 0.057, 0.02, sin(a) * 0.057))
		13:  # a horned helmet
			part.call(Ink.ball(0.05, 10), Ink.STONE, Vector3(0, 0.008, 0), Vector3(1, 0.85, 1), true)
			part.call(Ink.cone(0.052, 0.012, 0.052, 10), main, Vector3(0, 0.006, 0))
			for side in [-1.0, 1.0]:
				part.call(Ink.cone(0.012, 0.06, 0.0, 6), Ink.CREAM, Vector3(0, 0.04, side * 0.048), Vector3.ONE, true).rotation.x = -side * 0.9
		14:  # a bow
			part.call(Ink.ball(0.012, 6), band, Vector3(0, 0.02, 0))
			for side in [-1.0, 1.0]:
				part.call(Ink.cone(0.028, 0.045, 0.004, 6), main, Vector3(0, 0.02, side * 0.025), Vector3(1, 1, 0.5), true).rotation.x = -side * PI / 2
	for piece in hat.get_children():
		piece.material_override.set_shader_parameter("lift", 0.3)
	return hat
