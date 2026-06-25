#version 430

/*
 * OIT pass 2 - FIXED resolve.
 *
 * Root cause of the "can't tell box front/back when rotating" bug:
 * the original blend.frag truncates the per-pixel list to the FIRST
 * MAX_FRAGMENTS nodes walked. The list is head-insertion ordered, so that is
 * the LAST MAX_FRAGMENTS fragments *drawn* - a draw-order-dependent subset, not
 * the nearest ones. Once a pixel has >MAX_FRAGMENTS transparent fragments (easy
 * with many stacked boxes) the kept subset, and therefore the sort, changes with
 * the view -> popping / wrong z-order.
 *
 * Fix: while walking the full list, retain the MAX_FRAGMENTS *nearest* fragments
 * (smallest depth). Nearest layers dominate a back-to-front composite, so this is
 * both stable across views and visually ~exact. Also drops the fragile
 * `depth == previous` skip that discarded legitimate equal-depth fragments.
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
uint  fragmentIndices[MAX_FRAGMENTS];
float fragmentDepths[MAX_FRAGMENTS];

// Strict total order "a is FARTHER than b": depth desc, then (geometryID,
// instanceID) as stable tie-breaks so equal-depth fragments are kept/ordered
// deterministically (not by draw order).
bool farther(float da, uint ia, float db, uint ib)
{
	if (da != db) return da > db;
	if (nodes[ia].geometryID != nodes[ib].geometryID) return nodes[ia].geometryID > nodes[ib].geometryID;
	return nodes[ia].instanceID > nodes[ib].instanceID;
}

void main(void)
{
	uint walkerIndex = imageLoad(u_headIndex, ivec2(gl_FragCoord.xy)).r;
	if (walkerIndex == 0xffffffff) { fragColor = vec4(uBackground, 1.0); return; }

	uint numberFragments = 0;
	uint farthestSlot    = 0;   // kept slot that is farthest in the total order

	// Walk the WHOLE list, keeping the nearest MAX_FRAGMENTS fragments.
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
			// buffer full and new fragment is nearer than the farthest kept one:
			// replace it, then recompute which kept slot is now farthest.
			fragmentIndices[farthestSlot] = walkerIndex;
			fragmentDepths[farthestSlot]  = d;
			farthestSlot = 0;
			for (uint k = 1; k < MAX_FRAGMENTS; k++)
				if (farther(fragmentDepths[k], fragmentIndices[k], fragmentDepths[farthestSlot], fragmentIndices[farthestSlot]))
					farthestSlot = k;
		}

		walkerIndex = nodes[walkerIndex].nextIndex;
	}

	// Insertion sort using the total order, far -> near.
	for (uint i = 1; i < numberFragments; i++)
	{
		float keyDepth = fragmentDepths[i];
		uint  keyIdx   = fragmentIndices[i];
		int j = int(i) - 1;
		while (j >= 0 && farther(keyDepth, keyIdx, fragmentDepths[uint(j)], fragmentIndices[uint(j)]))
		{
			fragmentDepths[uint(j + 1)]  = fragmentDepths[uint(j)];
			fragmentIndices[uint(j + 1)] = fragmentIndices[uint(j)];
			j--;
		}
		fragmentDepths[uint(j + 1)]  = keyDepth;
		fragmentIndices[uint(j + 1)] = keyIdx;
	}

	// Back-to-front composite.
	vec3  color  = vec3(0.0);
	float factor = 1.0;            // remaining transmittance
	for (uint i = 0; i < numberFragments; i++)
	{
		vec4 c = nodes[fragmentIndices[i]].color;
		if (c.a < 0.001) continue;
		color  = mix(color, c.rgb, c.a);
		factor *= (1.0 - c.a);
		if (factor < 0.003) break;
	}

	fragColor = vec4(color + uBackground * factor, 1.0);
}
