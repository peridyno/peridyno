
/*
* Order-Independent Transparency with Linked-List
*/

#define BINDING_ATOMIC_FREE_INDEX 0
#define BINDING_IMAGE_HEAD_INDEX 0
#define BINDING_BUFFER_LINKED_LIST 0

// Fragment-node budget. MUST match MAX_OIT_NODES in GLRenderEngine.h (the CPU-side
// SSBO allocation). 32 bytes/node -> 32M nodes = 1 GB. On overflow the append guard
// below drops fragments; GLRenderEngine Step 4 reads the free-node counter and warns.
const uint uMaxNodes = 1024 * 1024 * 32;

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