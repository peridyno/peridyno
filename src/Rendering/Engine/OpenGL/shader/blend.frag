// shader for Order-Independent-Transparency blending
#version 440

#extension GL_GOOGLE_include_directive: enable
#include "transparency.glsl"

layout(location = 0) out vec4  fragColor;
layout(location = 1) out ivec4 fragIndices;

#define MAX_FRAGMENTS 128
uint  fragmentIndices[MAX_FRAGMENTS];
float fragmentDepths[MAX_FRAGMENTS];

// Strict total order "fragment a is FARTHER than b": depth descending, with
// (geometryID, instanceID) as stable tie-breaks. Two fragments at exactly the
// same depth (touching faces, coplanar geometry) would otherwise be kept/ordered
// according to draw order, leaving a tiny residual order-dependence; the stable
// tie-break removes it so the resolve is fully deterministic.
bool farther(float da, uint ia, float db, uint ib)
{
	if (da != db) return da > db;
	if (nodes[ia].geometryID != nodes[ib].geometryID) return nodes[ia].geometryID > nodes[ib].geometryID;
	return nodes[ia].instanceID > nodes[ib].instanceID;
}

void main(void)
{
	uint walkerIndex = imageLoad(u_headIndex, ivec2(gl_FragCoord.xy)).r;

	// Check, if a fragment was written.
	if (walkerIndex != 0xffffffff)
	{
		uint numberFragments = 0;
		uint farthestSlot    = 0;	// kept slot that is farthest in the total order -> first to be evicted

		// Walk the WHOLE per-pixel list, keeping the nearest MAX_FRAGMENTS
		// fragments. The list is head-insertion ordered, so a plain
		// "first MAX_FRAGMENTS walked" cap keeps a draw-order-dependent subset
		// (not the nearest), which makes the resolve order-DEPENDENT and breaks
		// the front/back ordering whenever a pixel has > MAX_FRAGMENTS layers.
		while (walkerIndex != 0xffffffff)
		{
			float d = nodes[walkerIndex].depth;

			if (numberFragments < MAX_FRAGMENTS)
			{
				uint slot = numberFragments;
				fragmentIndices[slot] = walkerIndex;
				fragmentDepths[slot]  = d;
				if (slot == 0 || farther(d, walkerIndex, fragmentDepths[farthestSlot], fragmentIndices[farthestSlot]))
					farthestSlot = slot;
				numberFragments++;
			}
			else if (farther(fragmentDepths[farthestSlot], fragmentIndices[farthestSlot], d, walkerIndex))
			{
				// buffer full and the new fragment is nearer than the farthest
				// kept one: replace it, then rescan to find the new farthest.
				fragmentIndices[farthestSlot] = walkerIndex;
				fragmentDepths[farthestSlot]  = d;
				farthestSlot = 0;
				for (uint k = 1; k < MAX_FRAGMENTS; k++)
					if (farther(fragmentDepths[k], fragmentIndices[k], fragmentDepths[farthestSlot], fragmentIndices[farthestSlot]))
						farthestSlot = k;
			}

			walkerIndex = nodes[walkerIndex].nextIndex;
		}

		// Insertion sort using the total order, far -> near (index 0 = farthest).
		for (uint i = 1; i < numberFragments; i++)
		{
			float keyDepth = fragmentDepths[i];
			uint keyIdx = fragmentIndices[i];
			int j = int(i) - 1;

			while (j >= 0 && farther(keyDepth, keyIdx, fragmentDepths[uint(j)], fragmentIndices[uint(j)]))
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
		for (uint i = 0; i < numberFragments; i++)
		{
			uint idx = fragmentIndices[i];

			// Skip if fully transparent
			if (nodes[idx].color.a < 0.001) continue;

			color = mix(color, nodes[idx].color.rgb, nodes[idx].color.a);
			factor = factor * (1 - nodes[idx].color.a);

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
