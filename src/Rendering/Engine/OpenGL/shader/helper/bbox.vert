#version 440

layout(location=0) in vec3 aPos;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;
  
void main(void) 
{
	gl_Position = transform.proj * transform.view * transform.model * vec4(aPos, 1);
}