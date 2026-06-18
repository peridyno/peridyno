// shader for Order-Independent-Transparency blending
#version 440

#extension GL_GOOGLE_include_directive: enable
#include "transparency.glsl"

layout(location = 0) out vec4  fragColor;
layout(location = 1) out ivec4 fragIndices;

#define MAX_FRAGMENTS 128
uint fragmentIndices[MAX_FRAGMENTS];

void main(void)
{
	uint walkerIndex = imageLoad(u_headIndex, ivec2(gl_FragCoord.xy)).r;

	// Check, if a fragment was written.
	if (walkerIndex != 0xffffffff)
	{
		uint numberFragments = 0;
		uint tempIndex;

		// Copy the fragment indices of this pixel.
		while (walkerIndex != 0xffffffff && numberFragments < MAX_FRAGMENTS)
		{
			fragmentIndices[numberFragments++] = walkerIndex;
			walkerIndex = nodes[walkerIndex].nextIndex;
		}

		// Step 1: Pre-fetch depths to avoid repeated SSBO access
		float fragmentDepths[MAX_FRAGMENTS];
		for (uint i = 0; i < numberFragments; i++) {
			fragmentDepths[i] = nodes[fragmentIndices[i]].depth;
		}

		// Step 2: Insertion sort using cached depths
		for (uint i = 1; i < numberFragments; i++)
		{
			float keyDepth = fragmentDepths[i];
			uint keyIdx = fragmentIndices[i];
			int j = int(i) - 1;
    
			while (j >= 0 && fragmentDepths[uint(j)] < keyDepth)
			{
				fragmentDepths[uint(j + 1)] = fragmentDepths[uint(j)];
				fragmentIndices[uint(j + 1)] = fragmentIndices[uint(j)];
				j--;
			}
			fragmentDepths[uint(j + 1)] = keyDepth;
			fragmentIndices[uint(j + 1)] = keyIdx;
		}

		vec3 color = vec3(0, 0, 0);
		float factor = 1.f;
		float depth = 1.f;
		for (uint i = 0; i < numberFragments; i++)
		{
			uint idx = fragmentIndices[i];
    
			// Skip if fully transparent
			if (nodes[idx].color.a < 0.001) continue;
			if (nodes[idx].depth == depth ) continue;
			
			depth = nodes[fragmentIndices[i]].depth;
			color = mix(color, nodes[fragmentIndices[i]].color.rgb, nodes[fragmentIndices[i]].color.a);
			factor = factor * (1 - nodes[fragmentIndices[i]].color.a);
			
			if (factor < 0.01) break;
		}
		float alpha = 1 - factor;

		fragColor = vec4(color / alpha, alpha);

		fragIndices.r = nodes[fragmentIndices[numberFragments - 1]].geometryID;
		fragIndices.g = nodes[fragmentIndices[numberFragments - 1]].instanceID;

		gl_FragDepth = nodes[fragmentIndices[numberFragments - 1]].depth;
	}
	else
	{
		discard;
	}
}
