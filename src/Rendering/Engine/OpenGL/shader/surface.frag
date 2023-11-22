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
	vec3 texCoord;    
	vec3 tangent;
    vec3 bitangent;
    flat int instanceID;
} fs_in;

layout(location = 0) out vec4  fragColor;
layout(location = 1) out ivec4 fragIndices;

layout(binding = 10) uniform sampler2D uTexColor;
layout(binding = 11) uniform sampler2D uTexBump;

layout(location = 4) uniform float uBumpScale = 1.0;

vec3 GetViewDir()
{
	// orthogonal projection
	if(uRenderParams.proj[3][3] == 1.0)
		return vec3(0, 0, 1);
	// perspective projection
	return normalize(-fs_in.position);
}

vec3 GetNormal()
{
	if(textureSize(uTexBump, 0).x > 1 && fs_in.texCoord.z > 0)
	{
		mat3 tbn = mat3(
			normalize(fs_in.tangent), 
			normalize(fs_in.bitangent), 
			normalize(fs_in.normal));

		// transform normal
		vec3 bump = texture(uTexBump, fs_in.texCoord.xy).rgb;
		bump = pow(bump, vec3(1.0/2.2));
		bump = bump * 2.0 - 1.0;

		//return vec3(bump.y);
		bump = mix(vec3(0, 0, 1), bump, uBumpScale);
		return normalize(tbn * bump); 
	}

	if(length(fs_in.normal) > 0)
		return normalize(fs_in.normal);

	// 
	vec3 X = dFdx(fs_in.position);
	vec3 Y = dFdy(fs_in.position);
	return normalize(cross(X,Y));
}

vec3 GetColor()
{
	if(fs_in.texCoord.z > 0)
		return texture(uTexColor, fs_in.texCoord.xy).rgb;
	return fs_in.color;
}

vec3 Shade()
{
	vec3 N = GetNormal();
	//return N;
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);
	vec3 baseColor = GetColor();

	// for main directional light
	{
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(baseColor, uMtl.metallic, uMtl.roughness, N, V, L);

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
		vec3 brdf = EvalPBR(baseColor, uMtl.metallic, uMtl.roughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uRenderParams.ambient.rgb * uRenderParams.ambient.a * baseColor;

	// final color
	vec3 color = ambient + Lo;
	color = ReinhardTonemap(color);
	color = GammaCorrect(color);

	return color;
}


vec3 ShadeTransparency()
{
	vec3 N = GetNormal();
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 Lo = vec3(0);	
	vec3 baseColor = GetColor();

	// for main directional light
	{
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(baseColor, uMtl.metallic, uMtl.roughness, N, V, L);

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
		vec3 brdf = EvalPBR(baseColor, uMtl.metallic, uMtl.roughness, N, V, V);

		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		Lo += radiance * brdf;
	}

	// ambient light
	vec3 ambient = uRenderParams.ambient.rgb * uRenderParams.ambient.a * baseColor;

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
		nodes[freeNodeIndex].color = vec4(ShadeTransparency(), uMtl.alpha);
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