#version 440

#extension GL_ARB_shading_language_include : require

#include "common.glsl"

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_norm;

// instance transform
layout(location = 3) in vec3 in_translation;
layout(location = 4) in vec3 in_scaling;
layout(location = 5) in mat3 in_rotation; // location 5-7 is used for rotation matrix

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;

	int  instanceID;
} vs_out;

// whether to use vertex normal
uniform bool uVertexNormal = false;
// is instance rendering?
uniform bool uInstanced = false;

void main(void) {
	vec3 position = in_vert;

	if(uInstanced) {
		// apply instance transform
		vec3 scaled = in_scaling * in_vert;
		vec3 rotated = in_rotation * scaled;
		position = rotated + in_translation;
	}

	vec4 worldPos = uTransform.model * vec4(position, 1.0);
	vec4 cameraPos = uTransform.view * worldPos;

	vs_out.position = cameraPos.xyz;
	vs_out.color = in_color;
	vs_out.instanceID = gl_InstanceID;

	if(uVertexNormal)
	{
		mat4 MV = uTransform.view * uTransform.model;
		mat4 N = transpose(inverse(MV));
		vs_out.normal = (N * vec4(in_norm, 0)).xyz;
	}

	gl_Position = uTransform.proj * cameraPos;
}