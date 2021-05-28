#version 440

layout(binding = 0) uniform sampler2D texture0;
layout(binding = 1) uniform sampler2D texture1;
layout(binding = 2) uniform sampler2D texture2;
layout(binding = 3) uniform sampler2D texture3;

layout(binding = 4) uniform sampler2D SSAOTex;

in  vec2 texcoord;
out vec4 FragColor;

vec3 albedo;
vec3 normal;
vec3 position;
float metallic;
float roughness;

layout (std140, binding=1) uniform LightUniformBlock
{
	vec4 ambient;

	// main directional light
	vec4 intensity;
	vec4 direction;

	mat4 transform;
} light;

/***************** ShadowMap *********************/
layout(binding = 5) uniform sampler2D shadowDepth;
layout(binding = 6) uniform sampler2D shadowColor;

vec3 GetShadowFactor(vec3 pos)
{
    vec4 posLightSpace = light.transform * vec4(pos, 1);
    vec3 projCoords = posLightSpace.xyz / posLightSpace.w;
    projCoords = projCoords * 0.5 + 0.5;

    float closestDepth = texture(shadowDepth, projCoords.xy).r;
    float currentDepth = min(1.0, projCoords.z);

    //float bias = max(0.05 * (1.0 - dot(normal, normalize(light.direction.xyz))), 0.005); 
    float bias = 0.005;

    // simple PCF
    vec3 shadow = vec3(0);
    vec2 texelSize = 1.0 / textureSize(shadowDepth, 0);
    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            float pcfDepth = texture(shadowDepth, projCoords.xy + vec2(x, y) * texelSize).r;
            float visible = currentDepth - bias > pcfDepth ? 0.0 : 1.0;
            shadow += texture(shadowColor, projCoords.xy + vec2(x, y) * texelSize).rgb * visible;
        }
    }
    return clamp(shadow / 9.0, 0, 1);
}


float SSAO()
{
	vec2 texelSize = 1.0 / vec2(textureSize(SSAOTex, 0));
    float result = 0.0;
    for (int x = -2; x < 2; ++x) 
    {
        for (int y = -2; y < 2; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(SSAOTex, texcoord + offset).r;
        }
    }
    return result / (4.0 * 4.0);
}

vec3 pbr();
void main(void) { 

	// lookup textures
	vec4 v0 = texture(texture0, texcoord);
    // discard invalid pixel...
    if (v0.r == 1.0)
        discard;
    // frag depth
    gl_FragDepth = v0.r;

	vec4 v1 = texture(texture1, texcoord);
	vec4 v2 = texture(texture2, texcoord);
	vec4 v3 = texture(texture3, texcoord);
			
	// unpack variables
	albedo   = v1.xyz;
	normal   = v2.xyz;
	position = v3.xyz;
	metallic  = v2.w;
	roughness = v3.w;

	// frag color
	vec3 color = pbr();
    
    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correct
    color = pow(color, vec3(1.0/2.2)); 

    FragColor = vec4(color, 1.0);
} 

// refer to https://learnopengl.com
const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.001); // prevent divide by zero for roughness=0.0 and NdotH=1.0
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
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
    vec3 N = normalize(normal);
    vec3 V = normalize(-position);
	
	float dotNV = dot(N, V);
	if(dotNV < 0.0)	N = -N;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo, metallic);

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
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        vec3 nominator    = NDF * G * F; 
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

        // add to outgoing radiance Lo
        //Lo += (kD * albedo / PI + specular) * radiance * NdotL;  // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    		
		Lo += GetShadowFactor(position) * (kD * albedo / PI + specular) * radiance * NdotL;
	}   
    
    // ambient lighting (note that the next IBL tutorial will replace 
    // this ambient lighting with environment lighting).
	float ao = SSAO();
    vec3 ambient = light.ambient.rgb * light.ambient.a * albedo * ao;

	return ambient + Lo;
}
