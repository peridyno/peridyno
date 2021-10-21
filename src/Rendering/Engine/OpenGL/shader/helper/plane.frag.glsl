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
	float	minValue;		// patch to color bleeding
} uShadowBlock;
layout(binding = 5) uniform sampler2D uTexShadow;

vec3 GetShadowFactor(vec3 pos)
{
	vec4 posLightSpace = uShadowBlock.transform * vec4(pos, 1);
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;	// NDC
	projCoords = projCoords * 0.5 + 0.5;

	// From http://fabiensanglard.net/shadowmappingVSM/index.php
	float distance = min(1.0, projCoords.z);
	vec2  moments = texture(uTexShadow, projCoords.xy).rg;

	// Surface is fully lit. as the current fragment is before the light occluder
	if (distance <= moments.x)
		return vec3(1.0);

	// The fragment is either in shadow or penumbra. We now use chebyshev's upperBound to check
	// How likely this pixel is to be lit (p_max)
	float variance = moments.y - (moments.x * moments.x);
	variance = max(variance, 0.00001);

	float d = distance - moments.x;
	float p_max = variance / (variance + d * d);

	// simple patch to color bleeding 
	p_max = (p_max - uShadowBlock.minValue) / (1.0 - uShadowBlock.minValue);
	p_max = clamp(p_max, 0.0, 1.0);

	return vec3(p_max);
}


subroutine void RenderPass(void);
layout(location = 0) subroutine uniform RenderPass renderPass;

void main(void) {
	renderPass();
}


layout(index = 0) subroutine(RenderPass) void ColorPass(void)
{
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * light.intensity.rgb + light.ambient.rgb;
	shading = clamp(shading, 0, 1);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0.0, 1.0);
	FragColor = vec4(shading * f, 0.5);
}

layout(index = 1) subroutine(RenderPass) void ShadowPass(void)
{
	float depth = gl_FragCoord.z;
	//depth = depth * 0.5 + 0.5;

	float moment1 = depth;
	float moment2 = depth * depth;

	// Adjusting moments (this is sort of bias per pixel) using partial derivative
	float dx = dFdx(depth);
	float dy = dFdy(depth);
	moment2 += 0.25 * (dx * dx + dy * dy);

	FragColor = vec4(moment1, moment2, 0.0, 0.0);
}
