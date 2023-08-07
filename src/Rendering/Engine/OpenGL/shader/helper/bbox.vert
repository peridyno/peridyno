#version 440

layout(location=0) in vec3 aPos;

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
  
void main(void) 
{
	gl_Position = uRenderParams.proj * uRenderParams.view * uRenderParams.model * vec4(aPos, 1);
}