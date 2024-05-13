#version 440

layout(location=0) in vec3 vPos;

layout(binding =1) uniform sampler2D envImage;
layout(location=2) uniform float roughness;
layout(location=3) uniform int   mode;

layout(location=0) out vec4 FragColor;

#define PI 3.1415926535897932384626433832795
const vec2 invAtan = vec2(0.5 / PI, 1.0 / PI);

vec2 get_uv(vec3 v){
    // convert coordinates
    vec2 uv = vec2(atan(-v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
	uv.y = 1.0 - uv.y;
    return uv;
}


float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}  

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVec);
}  


void main(void) { 
    vec3 N = normalize(vPos);

    if(mode == 0) 
    {    
        vec2 uv = get_uv(N);
        vec3 color = textureLod(envImage, uv, 0).rgb;
        color = pow(color, vec3(1/2.2));
        FragColor = vec4(color, 1);
    }
    else if(mode == 1)
    {
        // irradiance
        vec3 irradiance = vec3(0.0);  
        vec3 up    = vec3(0.0, 1.0, 0.0);
        vec3 right = normalize(cross(up, N));
        up         = normalize(cross(N, right));

        float sampleDelta = 0.025;
        float nrSamples = 0.0; 
        for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
        {
            for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
            {
                // spherical to cartesian (in tangent space)
                vec3 tangentSample = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
                // tangent space to world
                vec3 sampleVec = tangentSample.x * right + tangentSample.y * up + tangentSample.z * N; 
                
                //irradiance += texture(environmentMap, sampleVec).rgb * cos(theta) * sin(theta);
                vec2 uv = get_uv(sampleVec);
                irradiance += textureLod(envImage, uv, 0).rgb;
                nrSamples++;
            }
        }
        FragColor = vec4(PI * irradiance * (1.0 / float(nrSamples)), 1);
    }
    else if(mode == 2)
    {
        // specular
        vec3 R = N;
        vec3 V = R;
        const uint SAMPLE_COUNT = 2048;
        float totalWeight = 0.0;   
        vec3 prefilteredColor = vec3(0.0);    
        
        // simple hack for mip level
        vec2 size = textureSize(envImage, 0);
        float maxMipLevel = max(0, log2(min(size.x, size.y)) - 4.0);
        float minMipLevel = max(0, maxMipLevel - 4.0);
        float mipLevel = mix(minMipLevel,  maxMipLevel, roughness);

        for(uint i = 0u; i < SAMPLE_COUNT; ++i)
        {
            vec2 Xi = Hammersley(i, SAMPLE_COUNT);
            vec3 H  = ImportanceSampleGGX(Xi, N, roughness);
            vec3 L  = normalize(2.0 * dot(V, H) * H - V);
            float NdotL = max(dot(N, L), 0.0);
            if(NdotL > 0.0)
            {
                vec2 uv = get_uv(L);
                prefilteredColor += textureLod(envImage, uv, mipLevel).rgb * NdotL;
                //prefilteredColor += texture(environmentMap, L).rgb * NdotL;
                totalWeight += NdotL;
            }
        }
        prefilteredColor = prefilteredColor / totalWeight;
        FragColor = vec4(prefilteredColor, 1.0);
    }
}  