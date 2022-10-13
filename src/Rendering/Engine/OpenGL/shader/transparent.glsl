
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