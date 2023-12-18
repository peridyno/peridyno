#version 440

#extension GL_GOOGLE_include_directive: enable
#include "../common.glsl"

layout(location=0) in vec3 aPos;
 
void main(void) 
{
	gl_Position = uRenderParams.proj * uRenderParams.view * uRenderParams.model * vec4(aPos, 1);
}