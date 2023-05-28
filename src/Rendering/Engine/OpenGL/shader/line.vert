#version 440

/*
* Common uniform blocks
*/

layout(std140, binding = 0) uniform Transforms
{
	mat4 model;
	mat4 view;
	mat4 proj;

	// TODO: move to other place...
	int width;
	int height;
} uTransform;

layout(std140, binding = 1) uniform Lights
{
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
} uLight;

layout(std140, binding = 2) uniform Variables
{
	int index;
} uVars;


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