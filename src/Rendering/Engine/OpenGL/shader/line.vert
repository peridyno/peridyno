#version 440

#extension GL_ARB_shading_language_include : require

#include "common.glsl"

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_norm;

layout(std430, binding = 1) buffer Points {
    float p[];
};

layout(std430, binding = 2) buffer Edges {
    int index[];
};

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;

	int  instanceID;
} vs_out;

uniform float uRadius = 1.0;
uniform vec3  uBaseColor;

// get real position
// TODO: use a compute shader would improve performance
void GetPosNormal(out vec3 pos, out vec3 normal)
{
	int idx0 = index[gl_InstanceID * 2] * 3;
	int idx1 = index[gl_InstanceID * 2 + 1] * 3;
	vec3 p0 = vec3(p[idx0], p[idx0 + 1], p[idx0 + 2]);
	vec3 p1 = vec3(p[idx1], p[idx1 + 1], p[idx1 + 2]);

	// get transform
	vec3 d = normalize(p1 - p0);

	// build rotation matrix
	mat3 R;

	// hard code here...
	const vec3 v = vec3(0, 0, 1);
	
	// rotation axis
	if(dot(d, v) == -1.0) {
		R = mat3(-1, 0, 0,  0, -1, 0,  0, 0, -1);
	}
	else {
		vec3 A = normalize(d + v);

		R = mat3(	A.x * A.x * 2 - 1, A.x * A.y * 2,     A.x * A.z * 2, 
					A.x * A.y * 2,     A.y * A.y * 2 - 1, A.y * A.z * 2, 
					A.x * A.z * 2,     A.y * A.z * 2,     A.z * A.z * 2 - 1);
	}

	float len = length(p1 - p0);
	vec3  translate = p0;

	// position
	pos = translate + R * (in_vert * vec3(uRadius, uRadius, len));
	
	// normal
	normal = R * in_norm;
}


void main(void) {

	vec3 pos;
	vec3 normal;
	GetPosNormal(pos, normal);

	vec4 worldPos = uTransform.model * vec4(pos, 1.0);
	vec4 cameraPos = uTransform.view * worldPos;
	vs_out.position = cameraPos.xyz;

	mat4 MV = uTransform.view * uTransform.model;
	mat4 N = transpose(inverse(MV));
	vs_out.normal = (N * vec4(normal, 0)).xyz;

	vs_out.color = uBaseColor;

	gl_Position = uTransform.proj * cameraPos;  
}