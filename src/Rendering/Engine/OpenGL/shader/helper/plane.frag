#version 440

#extension GL_GOOGLE_include_directive: enable
#include "../common.glsl"
#include "../shadow.glsl"

layout(location=0) in vec2 vTexCoord;
layout(location=1) in vec3 vPosition;

layout(location=0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D uRulerTex;

void main(void) {
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * uRenderParams.intensity.rgb + uRenderParams.ambient.rgb;
	shading = clamp(shading, 0, 1);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0.0, 1.0);

	fragColor = vec4(shading * f, 0.5);	
}

