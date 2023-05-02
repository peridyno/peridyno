#version 440


/*
* Common uniform blocks
*/

layout(std140, binding = 0) uniform Transforms
{
	mat4 model;
	mat4 view;
	mat4 proj;

	// TODO: move to other place...
	int width;
	int height;
} uTransform;

layout(std140, binding = 1) uniform Lights
{
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
} uLight;

layout(std140, binding = 2) uniform Variables
{
	int index;
} uVars;


/*
* PBR shading
* refer to https://learnopengl.com
*/

const float PI = 3.14159265359;
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / max(denom, 0.001); // prevent divide by zero for roughness=0.0 and NdotH=1.0
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}


vec3 EvalPBR(vec3 color, float metallic, float roughness, vec3 N, vec3 V, vec3 L)
{
	// calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
	// of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, color, metallic);

	// calculate per-light radiance
	vec3 H = normalize(V + L);

	// Cook-Torrance BRDF
	float NDF = DistributionGGX(N, H, roughness);
	float G = GeometrySmith(N, V, L, roughness);
	vec3 F = FresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

	vec3 nominator = NDF * G * F;
	float denominator = 4 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
	vec3 specular = nominator / max(denominator, 0.001); // prevent divide by zero for NdotV=0.0 or NdotL=0.0

	// kS is equal to Fresnel
	vec3 kS = F;
	// for energy conservation, the diffuse and specular light can't
	// be above 1.0 (unless the surface emits light); to preserve this
	// relationship the diffuse component (kD) should equal 1.0 - kS.
	vec3 kD = vec3(1.0) - kS;
	// multiply kD by the inverse metalness such that only non-metals 
	// have diffuse lighting, or a linear blend if partly metal (pure metals
	// have no diffuse light).
	kD *= 1.0 - metallic;

	// scale light by NdotL
	float NdotL = max(dot(N, L), 0.0);

	return (kD * color / PI + specular) * NdotL;
}

/*
*  ShadowMap
*/

layout(std140, binding = 3) uniform ShadowUniform{
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

vec2 GetShadowMoments()
{
	float depth = gl_FragCoord.z;
	//depth = depth * 0.5 + 0.5;

	float moment1 = depth;
	float moment2 = depth * depth;

	// Adjusting moments (this is sort of bias per pixel) using partial derivative
	float dx = dFdx(depth);
	float dy = dFdy(depth);
	moment2 += 0.25 * (dx * dx + dy * dy);

	return vec2(moment1, moment2);
}


/*
* Post Processing...
*/

vec3 ReinhardTonemap(vec3 v)
{
	return v / (1.0f + v);
}

vec3 GammaCorrect(vec3 v)
{
	float gamma = 2.2;
	return pow(v, vec3(1.0 / gamma));
}



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