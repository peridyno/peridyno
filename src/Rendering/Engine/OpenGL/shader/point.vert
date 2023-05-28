#version 440

#extension GL_ARB_shading_language_include : require

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


// input
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

// output
out vec3 vPosition;
out vec3 vColor;

uniform float uPointSize;

vec3 ColorMapping();
void main(void) 
{
	vColor = aColor;

	vec4 worldPos = uTransform.model * vec4(aPosition, 1.0);
	vec4 cameraPos = uTransform.view * worldPos;

	vPosition = cameraPos.xyz;
	
	gl_Position = uTransform.proj * cameraPos; 
		
	// determine point size
	vec4 p1 = uTransform.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
	p1 = p1 / p1.w;
	vec4 p0 = uTransform.proj * vec4(0, 0, cameraPos.z, cameraPos.w);
	p0 = p0 / p0.w;
	gl_PointSize = uTransform.width * abs(p1.x - p0.x);
}