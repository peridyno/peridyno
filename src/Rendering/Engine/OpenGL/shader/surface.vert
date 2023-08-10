#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

// draw call input
layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec3 in_norm;
// instance transform
layout(location = 3) in vec3 in_translation;
layout(location = 4) in vec3 in_scaling;
layout(location = 5) in mat3 in_rotation; // location 5-7 is used for rotation matrix

// uniform input
// whether to use vertex normal
layout(location = 1) uniform bool uVertexNormal = false;
// is instance rendering?
layout(location = 2) uniform bool uInstanced = false;

// output
layout(location = 0) out VertexData {
	vec3 position;
	vec3 normal;
	vec3 color;
	int  instanceID;
} vs;

void main(void) {
	vs.color = in_color;
	vs.instanceID = -1;

	vec3 position = in_vert;

	if(uInstanced) {
		// apply instance transform
		vec3 scaled = in_scaling * in_vert;
		vec3 rotated = in_rotation * scaled;
		position = rotated + in_translation;		
		vs.instanceID = gl_InstanceID;
	}

	vec4 worldPos = uRenderParams.model * vec4(position, 1.0);
	vec4 cameraPos = uRenderParams.view * worldPos;

	vs.position = cameraPos.xyz;

	if(uVertexNormal)
	{
		mat4 MV = uRenderParams.view * uRenderParams.model;
		mat4 N = transpose(inverse(MV));
		vs.normal = (N * vec4(in_norm, 0)).xyz;
	}

	gl_Position = uRenderParams.proj * cameraPos;
}