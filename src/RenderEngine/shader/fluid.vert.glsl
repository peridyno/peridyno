#version 440

layout(location = 0) in vec3 in_position;

out vec3 vOrigin;

uniform float uPointRadius = 0.05;
uniform float uScreenWidth;

layout(std140, binding = 0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;


void main(void)
{
	vec4 worldPos = transform.model * vec4(in_position, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vOrigin = cameraPos.xyz;

	gl_Position = transform.proj * cameraPos;

	// point size
	vec4 projCorner = transform.proj * vec4(uPointRadius, uPointRadius, cameraPos.z, cameraPos.w);
	gl_PointSize = uScreenWidth * projCorner.x / projCorner.w;
}