#version 430

/*
 * OIT pass 1 - append each transparent fragment into a per-pixel linked list.
 * Faithful reproduction of PeriDyno's surface.frag::TransparencyLinkedList +
 * transparency.glsl (see src/Rendering/Engine/OpenGL/shader/).
 */

#extension GL_ARB_shader_atomic_counters : enable

in vec3 vNormalView;

// must match src/Rendering/Engine/OpenGL/shader/transparency.glsl layout
struct TransparentNode
{
	vec4  color;
	float depth;
	uint  nextIndex;
	int   geometryID;
	int   instanceID;
};

uniform uint uMaxNodes;   // node-buffer capacity (set from host)

layout(early_fragment_tests) in;

layout(binding = 0, offset = 0) uniform atomic_uint u_freeNodeIndex;
layout(binding = 0, r32ui)      uniform uimage2D     u_headIndex;
layout(binding = 0, std430)     buffer  LinkedList { TransparentNode nodes[]; };

uniform vec3  uColor;
uniform float uAlpha;
uniform int   uGeomID;

void main(void)
{
	// trivial shading just so faces are visually distinguishable
	vec3 N = normalize(vNormalView);
	float ndl = abs(normalize(vec3(0.3, 0.4, 1.0)).x * N.x +
	                 0.4 * N.y + N.z) * 0.45 + 0.55;
	vec3 shaded = uColor * ndl;

	uint freeNodeIndex = atomicCounterIncrement(u_freeNodeIndex);
	if (freeNodeIndex < uMaxNodes)
	{
		uint nextIndex = imageAtomicExchange(u_headIndex, ivec2(gl_FragCoord.xy), freeNodeIndex);

		nodes[freeNodeIndex].color      = vec4(shaded, uAlpha);
		nodes[freeNodeIndex].depth      = gl_FragCoord.z;
		nodes[freeNodeIndex].nextIndex  = nextIndex;
		nodes[freeNodeIndex].geometryID = uGeomID;
		nodes[freeNodeIndex].instanceID = 0;
	}
	// no color output
}
