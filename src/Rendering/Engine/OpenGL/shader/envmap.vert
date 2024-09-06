#version 440

layout(location=0) in  vec3 aPos;
layout(location=0) out vec3 vPos;

layout(location=0) uniform mat4 mvp;

void main(void) 
{
	gl_Position = mvp * vec4(aPos, 1);	
	vPos = aPos;
}