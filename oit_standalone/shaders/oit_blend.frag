#version 430

/*
 * OIT pass 2 - resolve the per-pixel linked list (sort + back-to-front composite).
 * This is a FAITHFUL copy of src/Rendering/Engine/OpenGL/shader/blend.frag (the
 * buggy baseline) so the standalone harness reproduces the engine behaviour.
 * Only the final output line composites over a background instead of emitting
 * premultiplied color, so the PNG is directly viewable.
 */

out vec4 fragColor;

struct TransparentNode
{
	vec4  color;
	float depth;
	uint  nextIndex;
	int   geometryID;
	int   instanceID;
};

layout(binding = 0, r32ui)  uniform uimage2D     u_headIndex;
layout(binding = 0, std430) buffer  LinkedList { TransparentNode nodes[]; };

uniform vec3 uBackground;

#define MAX_FRAGMENTS 128
uint fragmentIndices[MAX_FRAGMENTS];

void main(void)
{
	uint walkerIndex = imageLoad(u_headIndex, ivec2(gl_FragCoord.xy)).r;

	if (walkerIndex != 0xffffffff)
	{
		uint numberFragments = 0;

		// Copy the fragment indices of this pixel.
		while (walkerIndex != 0xffffffff && numberFragments < MAX_FRAGMENTS)
		{
			fragmentIndices[numberFragments++] = walkerIndex;
			walkerIndex = nodes[walkerIndex].nextIndex;
		}

		// Pre-fetch depths
		float fragmentDepths[MAX_FRAGMENTS];
		for (uint i = 0; i < numberFragments; i++)
			fragmentDepths[i] = nodes[fragmentIndices[i]].depth;

		// Insertion sort, far -> near
		for (uint i = 1; i < numberFragments; i++)
		{
			float keyDepth = fragmentDepths[i];
			uint  keyIdx   = fragmentIndices[i];
			int j = int(i) - 1;
			while (j >= 0 && fragmentDepths[uint(j)] < keyDepth)
			{
				fragmentDepths[uint(j + 1)]  = fragmentDepths[uint(j)];
				fragmentIndices[uint(j + 1)] = fragmentIndices[uint(j)];
				j--;
			}
			fragmentDepths[uint(j + 1)]  = keyDepth;
			fragmentIndices[uint(j + 1)] = keyIdx;
		}

		vec3  color  = vec3(0, 0, 0);
		float factor = 1.0;
		float depth  = 1.0;
		for (uint i = 0; i < numberFragments; i++)
		{
			uint idx = fragmentIndices[i];
			if (nodes[idx].color.a < 0.001) continue;
			if (nodes[idx].depth == depth)  continue;

			depth  = nodes[idx].depth;
			color  = mix(color, nodes[idx].color.rgb, nodes[idx].color.a);
			factor = factor * (1.0 - nodes[idx].color.a);

			if (factor < 0.01) break;
		}

		// composite over background (standalone-only change)
		fragColor = vec4(color + uBackground * factor, 1.0);
	}
	else
	{
		fragColor = vec4(uBackground, 1.0);
	}
}
