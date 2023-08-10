#version 460

#extension GL_GOOGLE_include_directive: enable

#include "common.glsl"
#include "shadow.glsl"
#include "transparency.glsl"
#include "pbr.glsl"

layout(location=0) in VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
    flat int instanceID;
} fs_in;

layout(location = 0) out vec4  fragColor;
layout(location = 1) out ivec4 fragIndices;

// material properties
layout(location = 3) uniform float uMetallic;
layout(location = 4) uniform float uRoughness;
layout(location = 5) uniform float uAlpha;

vec3 GetViewDir()
{
	// orthogonal projection
	if(uRenderParams.proj[3][3] == 1.0)
		return vec3(0, 0, 1);
	// perspective projection
	return normalize(-fs_in.position);
}

vec3 Shade()
{
	vec3 N = normalize(fs_in.normal);
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);

	// for main directional light
	{
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, L);

		// do not consider attenuation
		vec3 radiance = uRenderParams.intensity.rgb * uRenderParams.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
		if (uRenderParams.direction.w != 0)
			shadowFactor = GetShadowFactor(fs_in.position);

		Lo += shadowFactor * radiance * brdf;
	}
	
	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uRenderParams.ambient.rgb * uRenderParams.ambient.a * fs_in.color;

	// final color
	vec3 color = ambient + Lo;
	color = ReinhardTonemap(color);
	color = GammaCorrect(color);

	return color;
}


vec3 ShadeTransparency()
{
	vec3 N = normalize(fs_in.normal);
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);

	// for main directional light
	{
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, L);

		// do not consider attenuation
		vec3 radiance = uRenderParams.intensity.rgb * uRenderParams.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
//		if (uLight.direction.w != 0)
//			shadowFactor = GetShadowFactor(fs_in.position);

		Lo += shadowFactor * radiance * brdf;
	}
	
	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(fs_in.color, uMetallic, uRoughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uRenderParams.ambient.rgb * uRenderParams.ambient.a * fs_in.color;

	// final color
	vec3 color = ambient + Lo;
	color = ReinhardTonemap(color);
	color = GammaCorrect(color);

	return color;
}

void ColorPass(void)
{
	fragColor.rgb = Shade();
	fragColor.a = 1.0;
	
	// store index
	fragIndices.r = uRenderParams.index;
	fragIndices.g = fs_in.instanceID;
}

void ShadowPass(void)
{
	fragColor = vec4(GetShadowMoments(), 0.0, 0.0);
}


void TransparencyLinkedList(void)
{
	// Get the index of the next free node in the buffer.
	uint freeNodeIndex = atomicCounterIncrement(u_freeNodeIndex);

	// Check, if still space in the buffer.
	if (freeNodeIndex < uMaxNodes)
	{
		// Replace new index as the new head and gather the previous head, which will be the next index of this entry.
		uint nextIndex = imageAtomicExchange(u_headIndex, ivec2(gl_FragCoord.xy), freeNodeIndex);

		// Store the color, depth and the next index for later resolving.
		nodes[freeNodeIndex].color = vec4(ShadeTransparency(), uAlpha);
		nodes[freeNodeIndex].depth = gl_FragCoord.z;
		nodes[freeNodeIndex].nextIndex = nextIndex;
		nodes[freeNodeIndex].geometryID = uRenderParams.index;
		nodes[freeNodeIndex].instanceID = fs_in.instanceID;
	}
	// No output to the framebuffer.
}


void main(void) { 
	if(uRenderParams.mode == 0){
		ColorPass();
	}else if(uRenderParams.mode == 1){
		ShadowPass();
	}else if(uRenderParams.mode == 2){
		TransparencyLinkedList();
	}
} 