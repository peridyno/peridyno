#define M_PI 3.14159265358979323846

float clampedDot(vec3 x, vec3 y)
{
    return clamp(dot(x, y), 0.0, 1.0);
}

vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
    float alphaRoughnessSq = alphaRoughness * alphaRoughness;

    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

    float GGX = GGXV + GGXL;
    if (GGX > 0.0)
    {
        return 0.5 / GGX;
    }
    return 0.0;
}

// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
float D_GGX(float NdotH, float alphaRoughness)
{
    float alphaRoughnessSq = alphaRoughness * alphaRoughness;
    float f = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
    return alphaRoughnessSq / (M_PI * f * f);
}

//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
vec3 BRDF_lambertian(vec3 f0, vec3 f90, vec3 diffuseColor, float specularWeight, float VdotH)
{
    // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
    return (1.0 - specularWeight * F_Schlick(f0, f90, VdotH)) * (diffuseColor / M_PI);
}

//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
vec3 BRDF_specularGGX(vec3 f0, vec3 f90, float alphaRoughness, float specularWeight, float VdotH, float NdotL, float NdotV, float NdotH)
{
    vec3 F = F_Schlick(f0, f90, VdotH);
    float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
    float D = D_GGX(NdotH, alphaRoughness);

    return specularWeight * F * Vis * D;
}

vec3 EvalPBR(vec3 color, float metallic, float roughness, vec3 N, vec3 V, vec3 L)
{
    float ior = 1.5;
    vec3  f0 = vec3(0.04);
    float specularWeight = 1.0;

    metallic = clamp(metallic, 0.0, 1.0);

	vec3 c_diff = mix(color,  vec3(0), metallic);
	f0 = mix(f0, color, metallic);
	
	float perceptualRoughness = clamp(roughness, 0.0, 1.0);
	float alphaRoughness = perceptualRoughness * perceptualRoughness;

    float reflectance = max(max(f0.r, f0.g), f0.b);
    vec3 f90 = vec3(1.0);

	vec3 H = normalize(V + L);

	float NdotL = clampedDot(N, L);
    float NdotV = clampedDot(N, V);
    float NdotH = clampedDot(N, H);
    float LdotH = clampedDot(L, H);
    float VdotH = clampedDot(V, H);

	if (NdotL > 0.0 || NdotV > 0.0)
    {
		vec3 l_diffuse = NdotL *  BRDF_lambertian(f0, f90, c_diff, specularWeight, VdotH);
		vec3 l_specular = NdotL * BRDF_specularGGX(f0, f90, alphaRoughness, specularWeight, VdotH, NdotL, NdotV, NdotH);

		return l_diffuse + l_specular;
	}
    return vec3(0);
}

layout(std140, binding = 6) uniform IBLParams {
    float scale;
} iblParams;
layout(binding = 7) uniform samplerCube irradianceCube;
layout(binding = 8) uniform samplerCube prefilteredCube;
layout(binding = 9) uniform sampler2D   brdfLut;

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}   

// in world space...
vec3 EvalPBR_IBL(vec3 albedo, float metallic, float roughness, vec3 N, vec3 V)
{
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;	

    vec3 irradiance = texture(irradianceCube, N).rgb;
    vec3 diffuse = irradiance * albedo;
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 R = reflect(-V, N); 
    vec3 prefilteredColor = textureLod(prefilteredCube, R,  roughness * MAX_REFLECTION_LOD).rgb;    
    vec2 brdf  = texture(brdfLut, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * brdf.x + brdf.y);

    vec3 ambient = kD * diffuse + specular;
   
    return ambient * iblParams.scale;
}
