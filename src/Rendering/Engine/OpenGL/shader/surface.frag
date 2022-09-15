#version 460

#extension GL_ARB_shading_language_include : require

#include "common.glsl"
#include "pbr.glsl"
#include "shadow.glsl"
#include "postprocess.glsl"

in VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
} fs_in;



// PBR evaluation
uniform vec3  uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform float uAlpha;

layout(location = 0) out vec4 fragColor;

subroutine void RenderPass(void);
layout(location = 0) subroutine uniform RenderPass renderPass;

vec3 GetViewDir()
{
	// orthogonal projection
	if(uTransform.proj[3][3] == 1.0)
		return vec3(0, 0, 1);
	// perspective projection
	return normalize(-fs_in.position);
}

layout(index = 0) subroutine(RenderPass) void ColorPass(void)
{
	vec3 N = normalize(fs_in.normal);
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);

	// for main directional light
	{
		vec3 L = normalize(uLight.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, L);

		// do not consider attenuation
		vec3 radiance = uLight.intensity.rgb * uLight.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
		if (uLight.direction.w != 0)
			shadowFactor = GetShadowFactor(fs_in.position);

		Lo += shadowFactor * radiance * brdf;
	}
	
	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uLight.camera.rgb * uLight.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uLight.ambient.rgb * uLight.ambient.a * fs_in.color;

	// final color
	fragColor.rgb = ambient + Lo;
	fragColor.rgb = ReinhardTonemap(fragColor.rgb);
	fragColor.rgb = GammaCorrect(fragColor.rgb);

	// TODO: handle transparency
	fragColor.a = 1.0;
}

layout(index = 1) subroutine(RenderPass) void ShadowPass(void)
{
	fragColor = vec4(GetShadowMoments(), 0.0, 0.0);
}

void main(void) { 
	renderPass();
} 