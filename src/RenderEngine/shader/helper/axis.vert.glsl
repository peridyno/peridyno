#version 440

layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aColor;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;

out vec3 color;
   
void main(void) 
{
	color = aColor;

	vec3 p = mat3(transform.view * transform.model) * aPos;
	if(length(aPos) > 0.001)
		p = normalize(p);

	gl_Position = vec4(p, 1.0);
}