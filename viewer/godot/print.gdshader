// The whole look in one material: flat Dreamcast-era faces, shaded towards navy ink.
// Unshaded on purpose. The sun is a global direction and the tone is computed here, so the darker side of
// a duck is more ink on the same paper colour and never a muddier colour, which is what a print does.
// Night adds no ink: it is a wash of blue over the finished picture.
shader_type spatial;
render_mode unshaded;

global uniform vec3 sun_dir;
global uniform float daylight;
global uniform float screen;  // how much of the shading shows: 1 in the garden, less from a duck's back, where it got in the way
uniform vec3 base : source_color = vec3(0.96, 0.92, 0.84);
uniform vec3 ink : source_color = vec3(0.11, 0.16, 0.30);
uniform float cell = 7.0;  // unused since the shading is smooth; ink.gd still passes each part's screen size
uniform float lift = 0.0;  // added to the tone: rock in shadow is grey paper, not a wall of ink
uniform float tone_fixed = -1.0;  // 0 to 1 to skip the sun: water, shadows, anything flat on the ground

void fragment() {
	float tone = tone_fixed;
	if (tone < 0.0) {
		vec3 n = normalize(cross(dFdx(VERTEX), dFdy(VERTEX)));  // the face's own normal: flat shading
		n *= sign(n.z + 1e-6);  // towards the camera, whichever way the derivatives wind
		vec3 l = normalize((VIEW_MATRIX * vec4(sun_dir, 0.0)).xyz);
		tone = clamp(dot(n, l) * 0.7 + 0.47, 0.0, 1.0);  // a face the sun is on is bare paper; ink is for the turn away
	}
	tone = clamp(tone + lift, 0.0, 1.0);  // night is the same garden under more ink
	// A smooth shade of ink, as much as a halftone screen of dots would cover on average; the dots themselves
	// drew the eye away from the ducks.
	float shade = clamp((1.0 - tone) * 0.66 * 0.66 * 3.14159, 0.0, 1.0);
	// Night is a wash of blue over the same picture. It used to be more ink, and at 0.3 of the tone the whole
	// garden went under a grey screen of dots that nothing could be seen through (Chris, 2026-09-21).
	ALBEDO = mix(mix(base, ink, shade * 0.85 * screen), vec3(0.03, 0.05, 0.16), 0.5 * (1.0 - daylight));
}
