#version 440

#define BINDING_IMAGE_HEAD_INDEX 0
#define BINDING_BUFFER_LINKED_LIST 0

// This value is enough for the dragon. Increase for a more complex scene.
#define MAX_FRAGMENTS 8

struct NodeType
{
	vec4 color;
	float depth;
	uint nextIndex;
};

layout(binding = BINDING_IMAGE_HEAD_INDEX, r32ui) uniform uimage2D u_headIndex;

layout(binding = BINDING_BUFFER_LINKED_LIST, std430) buffer LinkedList
{
	NodeType nodes[];
	// Padding[]
};

out vec4 fragColor;

void main(void)
{
	const vec4 clearColor = vec4(0.0, 0.0, 0.0, 0.0);

	uint fragmentIndices[MAX_FRAGMENTS];

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

		// Sort the indices: Highest to lowest depth.
		for (uint i = 0; i < numberFragments; i++)
		{
			for (uint j = i + 1; j < numberFragments; j++)
			{
				if (nodes[fragmentIndices[j]].depth > nodes[fragmentIndices[i]].depth)
				{
					tempIndex = fragmentIndices[i];

					fragmentIndices[i] = fragmentIndices[j];

					fragmentIndices[j] = tempIndex;
				}
			}
		}

		vec3 color = vec3(0, 0, 0);
		float factor = 1.f;
		for (uint i = 0; i < numberFragments; i++)
		{
			color = mix(color, nodes[fragmentIndices[i]].color.rgb, nodes[fragmentIndices[i]].color.a);
			factor = factor * (1 - nodes[fragmentIndices[i]].color.a);
		}
		float alpha = 1 - factor;

		fragColor = vec4(color / alpha, alpha);
	}
	else
	{
		discard;
	}
}
