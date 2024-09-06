#version 440

#extension GL_GOOGLE_include_directive: enable
#include "../common.glsl"
#include "../shadow.glsl"

layout(location=0) in vec2 vTexCoord;
layout(location=1) in vec3 vPosition;

layout(location=0) out vec4 fragColor;

layout(binding = 1) uniform sampler2D uRulerTex;

layout(location = 0) uniform vec4 uPlaneColor;
layout(location = 1) uniform vec4 uRulerColor;

void main(void) {
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * uRenderParams.intensity.rgb + uRenderParams.ambient.rgb;
	float f = texture(uRulerTex, vTexCoord).r;

	vec4 color = mix(uPlaneColor, uRulerColor, f);
	fragColor = vec4(shading * color.rgb, color.a);	
}

