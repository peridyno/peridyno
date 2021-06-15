#version 440

// particle properties
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_velocity;
layout(location = 2) in vec3 in_force;

out vec3 vPosition;

uniform float uPointSize;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;

	int width;
	int height;
} transform;


void main(void) 
{
	vec4 worldPos = transform.model * vec4(in_position, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vPosition = cameraPos.xyz;
	
	gl_Position = transform.proj * cameraPos; 
		
	// point size
	vec4 projCorner = transform.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
	gl_PointSize = transform.width * projCorner.x / projCorner.w;
}