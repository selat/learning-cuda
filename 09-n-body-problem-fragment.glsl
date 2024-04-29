#version 330 core

out vec4 FragColor;

void main()
{
	float dist = distance(gl_PointCoord.xy, vec2(0.5, 0.5));
	if (dist < 0.5) {
		FragColor = vec4(0.0, 1.0, 0.0, 1.0);
	} else {
		discard;
	}
}
