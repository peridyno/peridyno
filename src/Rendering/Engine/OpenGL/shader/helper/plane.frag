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
* Order-Independent Transparency with Linked-List
*/

#define BINDING_ATOMIC_FREE_INDEX 0
#define BINDING_IMAGE_HEAD_INDEX 0
#define BINDING_BUFFER_LINKED_LIST 0

uniform uint uMaxNodes = 1024 * 1024 * 8;

// OIT - Linked List
struct TransparentNode
{
	vec4	color;
	float	depth;
	uint	nextIndex;
	int		geometryID;
	int     instanceID;
};

// enable early-z
layout(early_fragment_tests) in;

layout(binding = BINDING_ATOMIC_FREE_INDEX) uniform atomic_uint u_freeNodeIndex;
layout(binding = BINDING_IMAGE_HEAD_INDEX, r32ui) uniform uimage2D u_headIndex;
layout(binding = BINDING_BUFFER_LINKED_LIST, std430) buffer LinkedList
{
	TransparentNode nodes[];
};


in vec2 vTexCoord;
in vec3 vPosition;

//layout(location = 0) out vec4 fragColor;
//layout(location = 1) out ivec4 fragIndices;

layout(binding = 1) uniform sampler2D uRulerTex;

void main(void) {
	vec3 shadow = GetShadowFactor(vPosition);
	vec3 shading = shadow * uLight.intensity.rgb + uLight.ambient.rgb;
	shading = clamp(shading, 0, 1);
	float f = texture(uRulerTex, vTexCoord).r;
	f = clamp(0.5 - f, 0.0, 1.0);

//	fragColor = vec4(shading * f, 0.5);	
//    fragIndices = ivec4(-1);

	// Get the index of the next free node in the buffer.
	uint freeNodeIndex = atomicCounterIncrement(u_freeNodeIndex);

	// Check, if still space in the buffer.
	if (freeNodeIndex < uMaxNodes)
	{
		// Replace new index as the new head and gather the previous head, which will be the next index of this entry.
		uint nextIndex = imageAtomicExchange(u_headIndex, ivec2(gl_FragCoord.xy), freeNodeIndex);

		// Store the color, depth and the next index for later resolving.
		nodes[freeNodeIndex].color = vec4(shading * f, 0.5);
		nodes[freeNodeIndex].depth = gl_FragCoord.z;
		nodes[freeNodeIndex].nextIndex = nextIndex;
		nodes[freeNodeIndex].geometryID = -1;
		nodes[freeNodeIndex].instanceID = -1;
	}
}

