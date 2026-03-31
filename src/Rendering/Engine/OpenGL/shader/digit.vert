#version 440
#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

layout(location = 0) in vec3 aPosition;
layout(location = 1) out flat int vertexIndex;


void main() {
	vec4 worldPos = uRenderParams.model * vec4(aPosition, 1.0);
	vec4 cameraPos = uRenderParams.view * worldPos;

	
	gl_Position = uRenderParams.proj * cameraPos; 
		

    vertexIndex = gl_VertexID;
}