#version 440

const float radius	= 0.05;
const float bias	= 0.001;
const int kernelSize = 64;

layout(binding=0) uniform sampler2D texture0;
layout(binding=1) uniform sampler2D texture1;
layout(binding=2) uniform sampler2D texture2;
layout(binding=3) uniform sampler2D texture3;

// SSAO noise tex
layout(binding=4) uniform sampler2D texNoise;

in  vec2 texcoord;
out vec3 FragOut;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} transform;

// SSAO samples
layout (std140, binding=3) uniform SSAOKernel
{
	vec3 samples[64];
};

void main(void) { 

	vec4 v0 = texture(texture0, texcoord);
	

	// discard invalid pixel...
	if(v0.r == 1.0)	
		discard;
		
	vec3 normal = texture(texture2, texcoord).xyz;
	vec3 fragPos = texture(texture3, texcoord).xyz;	
		
	vec2 noiseScale = textureSize(texture0, 0) * vec2(1.0 / 4.0);	
	vec3 randomVec = texture(texNoise, texcoord * noiseScale).xyz;

	vec3 tangent   = normalize(randomVec - normal * dot(randomVec, normal));
	vec3 bitangent = cross(normal, tangent);
	mat3 TBN       = mat3(tangent, bitangent, normal);  
	
	float occlusion = 0.0;
	for(int i = 0; i < kernelSize; ++i)
	{
		// get sample position
		vec3 samplePos = TBN * samples[i]; // from tangent to view-space
		samplePos = fragPos + samplePos * radius; 
    
		vec4 offset = vec4(samplePos, 1.0);
		offset      = transform.proj * offset;    // from view to clip-space
		offset.xyz /= offset.w;               // perspective divide
		offset.xyz  = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0  
				
		float sampleDepth = texture(texture3, offset.xy).z; 

		float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
		occlusion += (sampleDepth >= samplePos.z + bias ? 1.0 : 0.0) * rangeCheck; 
	}  

	occlusion = 1.0 - (occlusion / kernelSize);
	FragOut = vec3(occlusion);
} 
