#version 440

#extension GL_ARB_shading_language_include : require

#include "common.glsl"
#include "shadow.glsl"
#include "transparent.glsl"

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
		nodes[freeNodeIndex].geometryID = uVars.index;
		nodes[freeNodeIndex].instanceID = -1;
	}
}

