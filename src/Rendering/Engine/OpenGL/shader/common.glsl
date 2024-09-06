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
	// parameters
	int width;
	int height;
	int index;
	int mode;
	float scale;
} uRenderParams;

// PBR material parameters
layout(std140, binding = 1) uniform PbrMaterial
{
	vec3  color;
	float metallic;
	float roughness;
	float alpha;
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