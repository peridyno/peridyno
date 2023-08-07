#version 440

layout(std140, binding = 0) uniform Transforms
{
	// transform
	mat4 model;
	mat4 view;
	mat4 proj;
	// illumination
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
	// parameters
	int width;
	int height;
	int index;
} uRenderParams;

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

	vec4 worldPos = uRenderParams.model * vec4(in_vert, 1.0);
	vec4 cameraPos = uRenderParams.view * worldPos;
	vs_out.position = cameraPos.xyz;

	vs_out.color = uBaseColor;

	gl_Position = uRenderParams.proj * cameraPos;  
}