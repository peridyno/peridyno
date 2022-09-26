#version 440

#extension GL_ARB_shading_language_include : require

#include "common.glsl"

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_norm;

// instancing...
layout(location = 3) in vec3 in_translation;
layout(location = 4) in vec3 in_scale;
layout(location = 5) in vec3 in_col0;
layout(location = 6) in vec3 in_col1;
layout(location = 7) in vec3 in_col2;

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
} vs_out;

// whether to use vertex normal
uniform bool uVertexNormal = false;

void main(void) {
	// apply instance transform
	vec3 scaled = vec3(in_scale.x * in_vert.x, in_scale.y * in_vert.y, in_scale.z * in_vert.z);
	vec3 rotated = in_col0 * scaled.x + in_col1 * scaled.y + in_col2 * scaled.z;
	vec3 translated = rotated + in_translation;

	vec4 worldPos = uTransform.model * vec4(translated, 1.0);
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