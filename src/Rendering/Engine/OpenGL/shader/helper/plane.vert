#version 440

#extension GL_GOOGLE_include_directive: enable
#include "../common.glsl"

layout(location=0) in vec3 aPos;

layout(location=2) uniform float uPlaneScale;
layout(location=3) uniform float uRulerScale;
layout(location=4) uniform int uAxis;

layout(location=0) out vec2 vTexCoord;
layout(location=1) out vec3 vPosition;
   
void main(void) 
{
	float x = aPos.x;
	float y = aPos.y;
	float z = aPos.z;
	
	vec3 p;

	if(uAxis == 0) p = vec3(y, z, x);
	else if(uAxis == 1) p = vec3(x, y, z);
	else if(uAxis == 2) p = vec3(z, x, y);
	else if(uAxis == 3) p = vec3(-y, z, x);
	else if(uAxis == 4) p = vec3(x, -y, z);
	else if(uAxis == 5) p = vec3(z, x, -y);
	else p = vec3(x, y, z);

	vec3 position = p * uPlaneScale;
	vTexCoord = vec2(x, z) * uPlaneScale / uRulerScale;
	vPosition = vec3(uRenderParams.view * vec4(position, 1.0));
	gl_Position = uRenderParams.proj * uRenderParams.view * vec4(position, 1.0);
}