#version 440

layout(location = 0) in vec3 in_vert;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;

out VertexData {
	vec3 position;
	vec3 normal;
} vs_out;

void main(void) {
	vec4 worldPos = transform.model * vec4(in_vert, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vs_out.position = cameraPos.xyz;

	gl_Position = transform.proj * cameraPos;  
}