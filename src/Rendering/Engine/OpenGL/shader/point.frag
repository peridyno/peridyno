#version 440


#extension GL_ARB_shading_language_include : require
#include "common.glsl"
#include "pbr.glsl"
#include "shadow.glsl"
#include "postprocess.glsl"

// fragment shader input
in vec3 vPosition;
in vec3 vColor;

uniform float uMetallic;
uniform float uRoughness;
uniform float uPointSize;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out ivec4 fragIndices;

subroutine void RenderPass(void);
layout(location = 0) subroutine uniform RenderPass renderPass;

vec3 fNormal;
vec3 fPosition;

void main(void) 
{
    // make sphere...
    vec2 uv = gl_PointCoord * 2.0 - vec2(1.0);
    float d = dot(uv, uv);
    if (d > 1.0)
    {
        discard;
    }
	fNormal = vec3(uv.x, -uv.y, sqrt(1.f-d));
	fPosition = vPosition + fNormal * uPointSize;

	// update depth
	vec4 clipPos = uTransform.proj * vec4(fPosition, 1);
	float ndcZ = clipPos.z / clipPos.w;

	gl_FragDepth = (gl_DepthRange.diff * ndcZ + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

	renderPass();
}

vec3 GetViewDir()
{
	// orthogonal projection
	if (uTransform.proj[3][3] == 1.0)
		return vec3(0, 0, 1);
	// perspective projection
	return normalize(-fPosition);
}

vec3 Shade()
{
	vec3 N = normalize(fNormal);
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);

	// for main directional light
	{
		vec3 L = normalize(uLight.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(vColor, uMetallic, uRoughness, N, V, L);

		// do not consider attenuation
		vec3 radiance = uLight.intensity.rgb * uLight.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
		if (uLight.direction.w != 0)
			shadowFactor = GetShadowFactor(vPosition);

		Lo += shadowFactor * radiance * brdf;
	}
	
	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(vColor, uMetallic, uRoughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uLight.camera.rgb * uLight.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uLight.ambient.rgb * uLight.ambient.a * vColor;

	// final color
	vec3 color = ambient + Lo;
	color = ReinhardTonemap(color);
	color = GammaCorrect(color);

	return color;
}

layout(index = 0) subroutine(RenderPass) void ColorPass(void)
{
	fragColor.rgb = Shade();
	fragColor.a = 1.0;
	
	// store index
	fragIndices.r = uVars.index;
	fragIndices.g = 0;
}

layout(index = 1) subroutine(RenderPass) void ShadowPass(void)
{
	fragColor = vec4(GetShadowMoments(), 0.0, 0.0);
}