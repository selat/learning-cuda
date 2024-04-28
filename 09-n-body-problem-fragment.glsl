#version 330 core

uniform vec2 centers[1];
uniform float radii[1];
uniform int num_circles;
out vec4 FragColor;

void main()
{
	float dist = distance(gl_FragCoord.xy, centers[0]);
	if (dist < radii[0]) {
		FragColor = vec4(0.0, 1.0, 0.0, 1.0);
	} else {
		FragColor = vec4(0.0, 0.0, 0.0, 1.0);
	}
}
