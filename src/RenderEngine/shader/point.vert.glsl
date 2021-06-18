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
	vColor = ColorMapping();

	vec4 worldPos = transform.model * vec4(aPosition, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vPosition = cameraPos.xyz;
	
	gl_Position = transform.proj * cameraPos; 
		
	// point size
	vec4 projCorner = transform.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
	gl_PointSize = transform.width * projCorner.x / projCorner.w;
}

// color map
vec3 JetColor(float v, float vmin, float vmax) {

	float x = clamp((v - vmin) / (vmax - vmin), 0, 1);
	float r = clamp(-4 * abs(x - 0.75) + 1.5, 0, 1);
	float g = clamp(-4 * abs(x - 0.50) + 1.5, 0, 1);
	float b = clamp(-4 * abs(x - 0.25) + 1.5, 0, 1);
	return vec3(r, g, b);
}

vec3 HeatColor(float v, float vmin, float vmax) {
	float x = clamp((v - vmin) / (vmax - vmin), 0, 1);
	float r = clamp(-4 * abs(x - 0.75) + 2, 0, 1);
	float g = clamp(-4 * abs(x - 0.50) + 2, 0, 1);
	float b = clamp(-4 * abs(x) + 2, 0, 1);
	return vec3(r, g, b);
}

vec3 ColorMapping()
{
	// color by velocity
	if (uColorMode == 0)
		return uBaseColor;
	if (uColorMode == 1)
		return JetColor(length(aVelocity), uColorMin, uColorMax);
	if (uColorMode == 2)
		return HeatColor(length(aVelocity), uColorMin, uColorMax);
	// default 
	return vec3(1, 0, 0);
}