#version 440

#extension GL_ARB_shading_language_include : require

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

	vec4 worldPos = uRenderParams.model * vec4(aPosition, 1.0);
	vec4 cameraPos = uRenderParams.view * worldPos;

	vPosition = cameraPos.xyz;
	
	gl_Position = uRenderParams.proj * cameraPos; 
		
	// determine point size
	vec4 p1 = uRenderParams.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
	p1 = p1 / p1.w;
	vec4 p0 = uRenderParams.proj * vec4(0, 0, cameraPos.z, cameraPos.w);
	p0 = p0 / p0.w;
	gl_PointSize = uRenderParams.width * abs(p1.x - p0.x);
}