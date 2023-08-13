#version 440

#include "../common.glsl"
#include "../shadow.glsl"
#include "../transparent.glsl"

in vec2 vTexCoord;
in vec3 vPosition;

out vec4 fragColor;

layout(binding = 1) uniform sampler2D uRulerTex;

void main(void) {
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * uLight.intensity.rgb + uLight.ambient.rgb;
	shading = clamp(shading, 0, 1);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0.0, 1.0);

	fragColor = vec4(shading * f, 0.5);	
}

