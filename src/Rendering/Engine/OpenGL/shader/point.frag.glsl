#version 440

in vec3 vPosition;
in vec3 vColor;

layout(std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
	int width;
	int height;
} transform;

layout(std140, binding = 1) uniform LightUniformBlock
{
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
} light;

uniform float uMetallic;
uniform float uRoughness;
uniform float uAlpha;

uniform float uPointSize;

layout(location = 0) out vec4 fragColor;

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
	vec4 clipPos = transform.proj * vec4(fPosition, 1);
	float ndcZ = clipPos.z / clipPos.w;

	gl_FragDepth = (gl_DepthRange.diff * ndcZ + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

	renderPass();
}

vec3 reinhard_tonemap(vec3 v)
{
	return v / (1.0f + v);
}

vec3 gamma_correct(vec3 v)
{
	float gamma = 2.2;
	return pow(v, vec3(1.0 / gamma));
}

vec3 pbr();
layout(index = 0) subroutine(RenderPass) void ColorPass(void)
{
	vec3 color = pbr();
	color = reinhard_tonemap(color);
	color = gamma_correct(color);
	fragColor.rgb = color;
	fragColor.a = 1.0;
}

layout(index = 1) subroutine(RenderPass) void DepthPass(void)
{
	fragColor = vec4(1);
}

/***************** ShadowMap *********************/
layout(std140, binding = 2) uniform ShadowUniform{
	mat4	transform;
	float	bias0;
	float	bias1;
	float	radius;
	float	clamp;
} uShadowBlock;
layout(binding = 5) uniform sampler2D uTexShadow;

vec3 GetShadowFactor(vec3 pos)
{
	vec4 posLightSpace = uShadowBlock.transform * vec4(pos, 1);
	vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;

	float closestDepth = texture(uTexShadow, projCoords.xy).r;
	float currentDepth = min(1.0, projCoords.z);

	//float temp = 1.0 - abs(dot(normal, normalize(light.direction.xyz)));
	float bias = 0.005; // max(uShadowBlock.bias0 * temp, uShadowBlock.bias1);

	// simple PCF
	vec2 shadow = vec2(0);
	vec2 texelSize = 1.0 / textureSize(uTexShadow, 0).xy;

	for (float x = -uShadowBlock.radius; x <= uShadowBlock.radius; x += 1.f)
	{
		for (float y = -uShadowBlock.radius; y <= uShadowBlock.radius; y += 1.f)
		{
			float pcfDepth = texture(uTexShadow, projCoords.xy + vec2(x, y) * texelSize).r;
			float visible = currentDepth - bias > pcfDepth ? 0.0 : 1.0;
			shadow += vec2(visible, 1);
		}
	}
	return vec3(clamp(shadow.x / shadow.y, uShadowBlock.clamp, 1.0));
}

// refer to https://learnopengl.com
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness * roughness;
	float a2 = a * a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH * NdotH;

	float nom = a2;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;

	return nom / max(denom, 0.001); // prevent divide by zero for uRoughness=0.0 and NdotH=1.0
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r * r) / 8.0;

	float nom = NdotV;
	float denom = NdotV * (1.0 - k) + k;

	return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
	float NdotV = max(dot(N, V), 0.0);
	float NdotL = max(dot(N, L), 0.0);
	float ggx2 = GeometrySchlickGGX(NdotV, roughness);
	float ggx1 = GeometrySchlickGGX(NdotL, roughness);

	return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
// ----------------------------------------------------------------------------
vec3 pbr()
{
	vec3 N = normalize(fNormal);
	vec3 V = normalize(-fPosition);

	float dotNV = dot(N, V);
	if (dotNV < 0.0)	N = -N;

	// calculate reflectance at fNormal incidence; if dia-electric (like plastic) use F0 
	// of 0.04 and if it's a metal, use the vColor color as F0 (uMetallic workflow)    
	vec3 F0 = vec3(0.04);
	F0 = mix(F0, vColor, uMetallic);

	// reflectance equation
	vec3 Lo = vec3(0.0);
	//for(int i = 0; i < 4; ++i) 
	{
		// calculate per-light radiance
		//vec3 L = normalize(lightPositions[i] - WorldPos);
		vec3 L = normalize(light.direction.xyz);
		vec3 H = normalize(V + L);
		//float distance = length(lightPositions[i] - WorldPos);
		//float attenuation = 1.0 / (distance * distance);
		//vec3 radiance = lightColors[i] * attenuation;
		vec3 radiance = light.intensity.rgb * light.intensity.a;
				
		// Cook-Torrance BRDF
		float NDF = DistributionGGX(N, H, uRoughness);
		float G = GeometrySmith(N, V, L, uRoughness);
		vec3 F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);

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
		kD *= 1.0 - uMetallic;

		// scale light by NdotL
		float NdotL = max(dot(N, L), 0.0);

		// add to outgoing radiance Lo
		//Lo += (kD * vColor / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again

		Lo += GetShadowFactor(fPosition) * (kD * vColor / PI + specular) * radiance * NdotL;
	}

	vec3 ambient = light.ambient.rgb * light.ambient.a * vColor.rgb;

	return ambient + Lo;
}