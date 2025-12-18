// common structures, functions and uniform blocks for stages

// render parameters
layout(std140, binding = 0) uniform RenderParams
{
	// transform
	mat4 model;
	mat4 view;
	mat4 proj;
	// illumination
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
	float ShadowMultiplier;
	float ShadowBrightness;
	float SamplePower;
	float ShadowContrast;
	// parameters
	int width;
	int height;
	int index;
	int mode;
	float scale;

} uRenderParams;


layout(std140, binding = 1) uniform PbrMaterial{

	int tintColor;
	int useAOTex;
	int useRoughnessTex;
	int useMetallicTex;

	int useEmissiveTex;
	int useAlphaTex;
	int tempData;
	int tempData2;

	vec3 color;
	float roughness;
	float metallic;
	float alpha;
	float ao; 

} uMtl;


vec3 ReinhardTonemap(vec3 v)
{
	return v / (1.0f + v);
}

vec3 GammaCorrect(vec3 v)
{
	float gamma = 2.2;
	return pow(v, vec3(1.0 / gamma));
}

vec3 GammaCorrectWithGamma(vec3 v, float gamma)
{
	return pow(v, vec3(1.0 / gamma));
}

float getBias(vec3 normal) 
{
	return max(0.005 * (1.0 - dot(normal, uRenderParams.direction.xyz)), 0.001);
}