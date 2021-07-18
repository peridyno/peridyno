#version 440

// particle properties
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aVelocity;

out vec3 vPosition;
out vec3 vColor;

uniform vec3  uBaseColor;
uniform float uPointSize;

uniform int   uColorMode = 0;
uniform float uColorMin = 0.0;
uniform float uColorMax = 5.0;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
	int width;
	int height;
} transform;

vec3 ColorMapping();
void main(void) 
{
	vColor = aVelocity;

	vec4 worldPos = transform.model * vec4(aPosition, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vPosition = cameraPos.xyz;
	
	gl_Position = transform.proj * cameraPos; 
		
	// determine point size
	if (transform.proj[3][3] == 1)
	{
		// if it is orthogonal projection
		gl_PointSize = transform.width * uPointSize;
	}
	else
	{
		// perspective projection
		vec4 projCorner = transform.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
		gl_PointSize = transform.width * projCorner.x / projCorner.w;
	}
}