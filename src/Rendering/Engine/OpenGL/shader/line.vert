#version 440



#include "common.glsl"

layout(location = 0) in vec3 in_vert;

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
	int  instanceID;	// not used for this type
} vs_out;

// we treat color as per-vertex
uniform vec3 uBaseColor;

void main(void) {

	vec4 worldPos = uTransform.model * vec4(in_vert, 1.0);
	vec4 cameraPos = uTransform.view * worldPos;
	vs_out.position = cameraPos.xyz;

	vs_out.color = uBaseColor;

	gl_Position = uTransform.proj * cameraPos;  
}