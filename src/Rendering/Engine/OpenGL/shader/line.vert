#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

layout(location = 0) in vec3 in_vert;

layout(location = 0) out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
	int  instanceID;	// not used for this type
} vs_out;

layout(location = 0) uniform int  uEdgeMode;
layout(location = 2) uniform vec3 uBaseColor;

void main(void) {

	vec4 worldPos = uRenderParams.model * vec4(in_vert, 1.0);
	vec4 cameraPos = uRenderParams.view * worldPos;
	vs_out.position = cameraPos.xyz;

	vs_out.color = uBaseColor;

	gl_Position = uRenderParams.proj * cameraPos;  
	
	// add a small offset to z so lines are drawn overlay on surface
	if (uEdgeMode==0) gl_Position.z -= 0.00003;
}