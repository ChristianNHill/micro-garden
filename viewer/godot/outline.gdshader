// The ink line round anything that needs its silhouette kept: the mesh again, a little fatter, inside out.
shader_type spatial;
render_mode unshaded, cull_front;

uniform vec3 ink : source_color = vec3(0.11, 0.16, 0.30);
uniform float grow = 0.005;

void vertex() {
	VERTEX += NORMAL * grow;
}

void fragment() {
	ALBEDO = ink;
}
