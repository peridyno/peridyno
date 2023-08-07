#version 440

layout(location=0) in vec3 aPos;

uniform float uPlaneScale;
uniform float uRulerScale;

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

out vec2 vTexCoord;
out vec3 vPosition;
   
void main(void) 
{
	vec3 position = aPos * uPlaneScale;
	vTexCoord = vec2(position.x, position.z) / uRulerScale;
	vPosition = vec3(uRenderParams.view * vec4(position, 1.0));
	gl_Position = uRenderParams.proj * uRenderParams.view * vec4(position, 1.0);
}