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
layout(binding = 12) uniform sampler2D uTexORM;
layout(binding = 13) uniform sampler2D uTexEmissiveColor;

layout(location = 4) uniform float uBumpScale = 1.0;

vec3 GetViewDir()
{
	// orthogonal projection
	if(uRenderParams.proj[3][3] == 1.0)
		return vec3(0, 0, 1);
	// perspective projection
	return normalize(-fs_in.position);
}

vec3 GetEmissive()
{
	vec3 emissive = vec3(0);
	if(uMtl.useEmissiveTex == 1)
		emissive = texture(uTexEmissiveColor, fs_in.texCoord.xy).rgb * uMtl.emissiveIntensity;

	return emissive;
}

vec3 GetORM()
{
	vec3 ormTexValue = vec3(1);
	if(uMtl.useAOTex == 1 || uMtl.useRoughnessTex == 1 || uMtl.useMetallicTex == 1)
		ormTexValue = texture(uTexORM, fs_in.texCoord.xy).rgb;

	vec3 ormValue ;
	if(uMtl.useAOTex == 0)
		ormValue.x = 1.0;
	else
		ormValue.x = ormTexValue.x;

	if(uMtl.useRoughnessTex == 0)
		ormValue.y = uMtl.roughness;
	else
		ormValue.y = ormTexValue.y;

	if(uMtl.useMetallicTex == 0)
		ormValue.z = uMtl.metallic;
	else
		ormValue.z = ormTexValue.z;

	return ormValue;

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

float VSM(vec3 projCoords)
{
		// From http://fabiensanglard.net/shadowmappingVSM/index.php
		float distance = min(1.0, projCoords.z);
		vec2  moments = texture(uTexShadow, projCoords.xy).rg;

		// Surface is fully lit. as the current fragment is before the light occluder
		if (distance < moments.x)
			return 1.0;

		// The fragment is either in shadow or penumbra. We now use chebyshev's upperBound to check
		// How likely this pixel is to be lit (p_max)
		float variance = moments.y - (moments.x * moments.x);
		variance = max(variance, 0.00001);

		float d = distance - moments.x;
		float p_max = variance / (variance + d * d);

		// simple patch to color bleeding 
		p_max = (p_max - uShadowBlock.minValue) / (1.0 - uShadowBlock.minValue);
		p_max = clamp(p_max, 0.0, 1.0);

		return p_max;
}


float ShadowMap(vec3 projCoords)
{
	float bias = getBias(fs_in.normal);
	//bias = max(0.005 * (1.0 - dot(normal, normalize(lightPos))), 0.001);

	float closestDepth = texture(uTexShadow, projCoords.xy).r;
	float currentDepth = projCoords.z;

	return (currentDepth - bias <= closestDepth) ? 1.0 : 0.0;
}

float PCF(vec3 projCoords)
{
		ivec2 texSize = textureSize(uTexShadow, 0);
		vec2 texelSize = 1.0 /vec2(texSize);
		int range = 3;
		int samples = 0;
		float shadow = 0.0;

		bool UseVSM = false;

		for (int x = -range; x <= range; ++x) 
		{
			for (int y = -range; y <= range; ++y) 
			{ 
				if(UseVSM)
					shadow +=VSM(vec3(projCoords.xy + vec2(x, y) * texelSize,projCoords.z)); 
				else
					shadow += ShadowMap(vec3(projCoords.xy + vec2(x, y) * texelSize,projCoords.z));
				samples++;
			}
		}
		shadow /= float(samples);
		return shadow;
}

vec3 GetShadowFactorSM(vec3 pos)
{ 
	vec4 posLightSpace = uShadowBlock.transform * vec4(pos, 1);
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;	// NDC
	projCoords = projCoords * 0.5 + 0.5;
	
	float vsmResult = VSM(projCoords);
	return vec3(vsmResult);	
}

vec3 Shade()
{
	vec3 N = GetNormal();
	//return N;
	vec3 V = GetViewDir();

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;
	
	vec3 color = vec3(0);

	vec3 baseColor = GetColor();
	vec3 ORM = GetORM();

	vec3 ORMCorrect = GammaCorrectWithGamma(ORM,2.2);
		
	float artFactor = 1;

	// for main directional light
	{

		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(baseColor, ORMCorrect.b, ORMCorrect.g, N, V, L);

		// do not consider attenuation
		vec3 radiance = uRenderParams.intensity.rgb * uRenderParams.intensity.a;

		// shadow
		vec3 shadowFactor = vec3(1);
		if (uRenderParams.direction.w != 0)
			shadowFactor = GetShadowFactorSM(fs_in.position);

		color += shadowFactor * radiance * brdf;

		vec3 brdf_art = EvalPBR(vec3(1), 0, 1.0f, N, V, L);

		if(uRenderParams.ShadowMultiplier > 0 && uRenderParams.ShadowBrightness < 1 && uRenderParams.ShadowContrast>0.1)
		{
			artFactor = shadowFactor.x * uRenderParams.ShadowContrast * brdf_art.x;
			artFactor = pow(artFactor,uRenderParams.SamplePower);

			artFactor = mix(1.0, (artFactor*(1 - uRenderParams.ShadowBrightness) + uRenderParams.ShadowBrightness), uRenderParams.ShadowMultiplier);
			artFactor = clamp(artFactor, 0.0, 1.0);
		}

	}


	// for a simple camera light
	{
		// evaluate BRDF
		vec3 brdf = EvalPBR(baseColor,  ORMCorrect.b, ORMCorrect.g, N, V, V);
		// do not consider attenuation
		vec3 radiance = uRenderParams.camera.rgb * uRenderParams.camera.a;

		// no shadow...
		color += radiance * brdf; 
	}


	// IBL
	{
		// convert to world space...
	    mat4 invView = inverse(uRenderParams.view);
		N = normalize(vec3(invView * vec4(N, 0))); 
		V = normalize(vec3(invView * vec4(V, 0))); 

		color += EvalPBR_IBL(baseColor, ORMCorrect.b, ORMCorrect.g, N, V);
	}
	
	// ambient light
	{
		color += uRenderParams.ambient.rgb * uRenderParams.ambient.a * baseColor;
	}

	// final color
	color *= artFactor;
	color = ReinhardTonemap(color);
	color = GammaCorrect(color);

	vec3 EmissiveColor = GetEmissive();
	if(EmissiveColor.r + EmissiveColor.g + EmissiveColor.b>0.05)	
		color = color*(1 - uMtl.emissiveIntensity) + GetEmissive();

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

	vec3 ORM = GetORM();
	vec3 ORMCorrect = GammaCorrectWithGamma(ORM,2.2);

	// for main directional light
	{
		vec3 L = normalize(uRenderParams.direction.xyz);
	
		// evaluate BRDF
		vec3 brdf = EvalPBR(baseColor, ORMCorrect.b, ORMCorrect.g, N, V, L);

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
		vec3 brdf = EvalPBR(baseColor, ORMCorrect.b, ORMCorrect.g, N, V, V);

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