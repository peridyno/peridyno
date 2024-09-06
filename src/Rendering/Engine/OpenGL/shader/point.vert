#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

// input
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aColor;

// output
layout(location = 0) out vec3 vPosition;
layout(location = 1) out vec3 vColor;


layout(location = 0) uniform float uPointSize;

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