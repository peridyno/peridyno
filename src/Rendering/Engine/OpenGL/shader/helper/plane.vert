#version 440

#extension GL_GOOGLE_include_directive: enable
#include "../common.glsl"

layout(location=0) in vec3 aPos;

layout(location=2) uniform float uPlaneScale;
layout(location=3) uniform float uRulerScale;

layout(location=0) out vec2 vTexCoord;
layout(location=1) out vec3 vPosition;
   
void main(void) 
{
	vec3 position = aPos * uPlaneScale;
	vTexCoord = vec2(position.x, position.z) / uRulerScale;
	vPosition = vec3(uRenderParams.view * vec4(position, 1.0));
	gl_Position = uRenderParams.proj * uRenderParams.view * vec4(position, 1.0);
}