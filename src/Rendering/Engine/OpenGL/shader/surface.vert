#version 460

#extension GL_ARB_shading_language_include : require

#include "common.glsl"

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_norm;

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
} vs_out;

// whether to use vertex normal
uniform bool uVertexNormal = false;
   
void main(void) {
	vec4 worldPos = uTransform.model * vec4(in_vert, 1.0);
	vec4 cameraPos = uTransform.view * worldPos;

	vs_out.position = cameraPos.xyz;
	vs_out.color = in_color;
	
	if(uVertexNormal)
	{
		mat4 VP = uTransform.view * uTransform.model;
		mat4 N = transpose(inverse(VP));
		vs_out.normal = (N * vec4(in_norm, 0)).xyz;
	}

	gl_Position = uTransform.proj * cameraPos;  
}