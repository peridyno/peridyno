#version 440

in vec2 vTexCoord;
in vec3 vPosition;
out vec4 FragColor;

layout(binding = 1) uniform sampler2D uRulerTex;

layout(std140, binding = 1) uniform LightUniformBlock
{
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
} light;

/***************** ShadowMap *********************/
layout(std140, binding = 2) uniform ShadowUniformBlock
{
	// support up to 4 cascaded shadow map layers
	mat4 transform[4];
	vec4 minDepth;
	vec4 maxDepth;
// may have some other data in future
} shadow;

layout(binding = 5) uniform sampler2DArray shadowDepth;
//layout(binding = 6) uniform sampler2D shadowColor;

int GetShadowLevel(vec3 pos)
{
	float depth = abs(pos.z);
	for (int i = 0; i < 4; i++)
	{
		if (depth < shadow.maxDepth[i])
			return i;
	}
	return 0;
}

vec3 GetShadowFactor(vec3 pos)
{	
	int level = GetShadowLevel(pos);

	vec4 posLightSpace = shadow.transform[level] * vec4(pos, 1);
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;
	
	float closestDepth = texture(shadowDepth, vec3(projCoords.xy, level)).r;
	float currentDepth = min(1.0, projCoords.z);

	//float bias = max(0.05 * (1.0 - dot(normal, normalize(light.direction.xyz))), 0.005); 
	float bias = 0.005;

	// simple PCF
	vec3 shadow = vec3(0);	
	vec2 texelSize = 1.0 / textureSize(shadowDepth, 0).xy;
	for (int x = -1; x <= 1; ++x)
	{
		for (int y = -1; y <= 1; ++y)
		{
			float pcfDepth = texture(shadowDepth, vec3(projCoords.xy + vec2(x, y) * texelSize, level)).r;
			float visible = currentDepth - bias > pcfDepth ? 0.0 : 1.0;
			shadow += visible;
		}
	}
	return clamp(shadow / 9.0, 0, 1)/* * sqrt(level + 1)*/;
}


void main(void) 
{ 
	vec3 shadow = GetShadowFactor(vPosition);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0, 1);
	FragColor = vec4(shadow * f, 0.5);
}