#version 450

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"
#include "pbr.glsl"
#include "shadow.glsl"


// fragment shader input
layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;

layout(location = 0) uniform float uPointSize;
layout(location = 1) uniform float uMetallic;
layout(location = 2) uniform float uRoughness;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out ivec4 fragIndices;

vec3 fNormal;
vec3 fPosition;

vec3 GetViewDir()
{
	// orthogonal projection
	if (uRenderParams.proj[3][3] == 1.0)
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
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(vColor, uMetallic, uRoughness, N, V, L);

		// do not consider attenuation
		vec3 radiance = uRenderParams.intensity.rgb * uRenderParams.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
		if (uRenderParams.direction.w != 0)
			shadowFactor = GetShadowFactor(vPosition);

		Lo += shadowFactor * radiance * brdf;
	}
	
	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(vColor, uMetallic, uRoughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

// IBL
	if(true)
	{
		// convert to world space...
	    mat4 invView = inverse(uRenderParams.view);
		N = normalize(vec3(invView * vec4(N, 0))); 
		V = normalize(vec3(invView * vec4(V, 0))); 
		Lo += EvalPBR_IBL(vColor, uMetallic, uRoughness, N, V);
	}
	
	// ambient light
	{
		Lo += uRenderParams.ambient.rgb * uRenderParams.ambient.a * vColor;
	}
	 
	// final color
	Lo = ReinhardTonemap(Lo);
	Lo = GammaCorrect(Lo);

	return Lo;
}

void ColorPass(void)
{
	fragColor.rgb = Shade();
	fragColor.a = 1.0;
	
	// store index
	fragIndices.r = uRenderParams.index;
	fragIndices.g = 0;
}

void ShadowPass(void)
{
	fragColor = vec4(GetShadowMoments(), 0.0, 0.0);
}

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
	vec4 clipPos = uRenderParams.proj * vec4(fPosition, 1);
	float ndcZ = clipPos.z / clipPos.w;

	gl_FragDepth = (ndcZ + 1) / 2.0;
	//gl_FragDepth = (gl_DepthRange.diff * ndcZ + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

	if(uRenderParams.mode == 0){
		ColorPass();
	}else if(uRenderParams.mode == 1){
		ShadowPass();
	}else if(uRenderParams.mode == 2){
		discard;
	}
}
