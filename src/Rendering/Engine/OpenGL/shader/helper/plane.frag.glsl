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
layout(std140, binding = 2) uniform ShadowUniform{
	mat4	transform;
	float	bias0;
	float	bias1;
	float	radius;
	float	clamp;
} uShadowBlock;
layout(binding = 5) uniform sampler2D uTexShadow;

vec3 GetShadowFactor(vec3 pos)
{
	vec4 posLightSpace = uShadowBlock.transform * vec4(pos, 1);
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;

	float closestDepth = texture(uTexShadow, projCoords.xy).r;
	float currentDepth = min(1.0, projCoords.z);

	//float temp = 1.0 - abs(dot(normal, normalize(light.direction.xyz)));
	float bias = 0.005; // max(uShadowBlock.bias0 * temp, uShadowBlock.bias1);

	// simple PCF
	vec2 shadow = vec2(0);
	vec2 texelSize = 1.0 / textureSize(uTexShadow, 0).xy;

	for (float x = -uShadowBlock.radius; x <= uShadowBlock.radius; x += 1.f)
	{
		for (float y = -uShadowBlock.radius; y <= uShadowBlock.radius; y += 1.f)
		{
			float pcfDepth = texture(uTexShadow, projCoords.xy + vec2(x, y) * texelSize).r;
			float visible = currentDepth - bias > pcfDepth ? 0.0 : 1.0;
			shadow += vec2(visible, 1);
		}
	}
	return vec3(clamp(shadow.x / shadow.y, uShadowBlock.clamp, 1.0));
}

void main(void) 
{ 
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * light.intensity.rgb + light.ambient.rgb;
	shading = clamp(shading, 0, 1);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0.0, 1.0);
	FragColor = vec4(shading * f, 0.5);
}