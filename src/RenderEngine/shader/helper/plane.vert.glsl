#version 440

layout(location=0) in vec3 aPos;

uniform float uScale = 5.f;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} tm;

out vec2 vTexCoord;
out vec3 vPosition;
   
void main(void) 
{
	vec3 position = aPos * uScale;
	vTexCoord = vec2(position.x, position.z);
	vPosition = vec3(tm.view * vec4(position, 1.0));
	gl_Position = tm.proj * tm.view * vec4(position, 1.0);
}