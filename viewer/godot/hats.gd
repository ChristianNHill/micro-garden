# A hat from its style number: the same number always makes the same hat, on the lawn or on a head.
# Five shapes, six colours and a band in another.
extends RefCounted

const Ink := preload("res://ink.gd")
const COLOURS := [Color("e2483d"), Color("f4c430"), Color("3a7bd5"), Color("7d4fc2"), Color("2fa36b"), Color("f08fb5")]


static func make(style: int) -> Node3D:
	var hat := Node3D.new()
	var main: Color = COLOURS[(style / 5) % COLOURS.size()]
	var band: Color = COLOURS[(style / 30 + 1 + (style / 5) % COLOURS.size()) % COLOURS.size()]
	match style % 5:
		0:  # a party cone with a bobble
			Ink.part(hat, Ink.cone(0.045, 0.1, 0.0, 8), main, Vector3(0, 0.05, 0), Vector3.ONE, 5.0, -1.0, true)
			Ink.part(hat, Ink.ball(0.014, 6), band, Vector3(0, 0.105, 0), Vector3.ONE, 5.0)
		1:  # a top hat
			Ink.part(hat, Ink.cone(0.07, 0.008, 0.07, 10), main, Vector3(0, 0.004, 0), Vector3.ONE, 5.0, -1.0, true)
			Ink.part(hat, Ink.cone(0.042, 0.085, 0.045, 10), main, Vector3(0, 0.05, 0), Vector3.ONE, 5.0, -1.0, true)
			Ink.part(hat, Ink.cone(0.044, 0.016, 0.044, 10), band, Vector3(0, 0.02, 0), Vector3.ONE, 5.0)
		2:  # a beanie with a pompom
			Ink.part(hat, Ink.ball(0.05, 8), main, Vector3(0, 0.012, 0), Vector3(1, 0.85, 1), 5.0, -1.0, true)
			Ink.part(hat, Ink.cone(0.052, 0.018, 0.052, 10), band, Vector3(0, 0.008, 0), Vector3.ONE, 5.0)
			Ink.part(hat, Ink.ball(0.017, 6), band, Vector3(0, 0.062, 0), Vector3.ONE, 5.0)
		3:  # a crown
			Ink.part(hat, Ink.cone(0.045, 0.04, 0.045, 10), main, Vector3(0, 0.02, 0), Vector3.ONE, 5.0, -1.0, true)
			for k in 5:
				var a := TAU * k / 5.0
				Ink.part(hat, Ink.cone(0.013, 0.035, 0.0, 4), main, Vector3(cos(a) * 0.036, 0.055, sin(a) * 0.036), Vector3.ONE, 5.0)
				Ink.part(hat, Ink.ball(0.008, 5), band, Vector3(cos(a) * 0.036, 0.076, sin(a) * 0.036), Vector3.ONE, 5.0)
		4:  # a sun hat
			Ink.part(hat, Ink.cone(0.095, 0.008, 0.1, 12), main, Vector3(0, 0.004, 0), Vector3.ONE, 5.0, -1.0, true)
			Ink.part(hat, Ink.ball(0.042, 8), main, Vector3(0, 0.01, 0), Vector3(1, 0.8, 1), 5.0, -1.0, true)
			Ink.part(hat, Ink.cone(0.044, 0.014, 0.043, 10), band, Vector3(0, 0.014, 0), Vector3.ONE, 5.0)
	for piece in hat.get_children():
		piece.material_override.set_shader_parameter("lift", 0.3)
	return hat
