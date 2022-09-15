#version 440

layout(location = 0) in vec3 in_vert;
layout(location = 1) in vec3 in_translation;
layout(location = 2) in vec3 in_scale;
layout(location = 3) in vec3 in_col0;
layout(location = 4) in vec3 in_col1;
layout(location = 5) in vec3 in_col2;
//layout(location = 1) in vec3 in_norm;

layout(std140, binding = 0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;

out VertexData
{
	vec3 position;
	vec3 normal;
} vs_out;

void main(void) {
	vec3 scaled = vec3(in_scale.x*in_vert.x, in_scale.y*in_vert.y, in_scale.z*in_vert.z);
	vec3 rotated = in_col0 * scaled.x + in_col1 * scaled.y + in_col2 * scaled.z;
	vec4 worldPos = transform.model * vec4(rotated + in_translation, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vs_out.position = cameraPos.xyz;
	gl_Position = transform.proj * cameraPos;
}