// The whole look in one material: flat Dreamcast-era faces, shaded by a halftone screen of navy ink.
// Unshaded on purpose. The sun is a global direction and the tone is computed here, so the darker side of
// a duck is more ink on the same paper colour and never a muddier colour, which is what a print does.
// Night adds no ink: it is a wash of blue over the finished picture.
// The screen is fixed to the window: it does not crawl while the camera is still, and the camera drifts
// slowly. ponytail: if Chris sees shimmer under the drift, lock the grid to the object instead.
shader_type spatial;
render_mode unshaded;

global uniform vec3 sun_dir;
global uniform float daylight;
global uniform float screen;  // how much of the halftone shows: 1 in the garden, less from a duck's back, where it got in the way
uniform vec3 base : source_color = vec3(0.96, 0.92, 0.84);
uniform vec3 ink : source_color = vec3(0.11, 0.16, 0.30);
uniform float cell = 7.0;  // pixels between dots; ducks are screened more coarsely than the ground
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
	vec2 g = fract(mat2(vec2(0.7071, -0.7071), vec2(0.7071, 0.7071)) * FRAGCOORD.xy / cell) - 0.5;
	float radius = sqrt(clamp(1.0 - tone, 0.0, 1.0)) * 0.66;  // ink by area
	float dotted = 1.0 - smoothstep(radius - 0.05, radius + 0.05, length(g));
	// Night is a wash of blue over the same picture. It used to be more ink, and at 0.3 of the tone the whole
	// garden went under a grey screen of dots that nothing could be seen through (Chris, 2026-09-21).
	ALBEDO = mix(mix(base, ink, dotted * 0.85 * screen), vec3(0.03, 0.05, 0.16), 0.5 * (1.0 - daylight));
}
