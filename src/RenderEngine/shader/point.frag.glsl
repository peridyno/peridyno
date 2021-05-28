#version 440

in vec3 vPosition;
in vec3 vColor;

uniform float uPointSize;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;

	int width;
	int height;
} transform;

layout(std140, binding = 2) uniform MaterialUniformBlock
{
	vec4  albedo;
	float metallic;
	float roughness;

	int   colorMode;
	float colorMin;
	float colorMax;

	int   shadowMode;
};

layout(location = 0) out vec4 fOutput0;
layout(location = 1) out vec4 fOutput1;
layout(location = 2) out vec4 fOutput2;

subroutine void RenderPass(void);
layout(location = 0) subroutine uniform RenderPass renderPass;

vec3 fNormal;
vec3 fPosition;

void main(void) 
{
    // make sphere...
    vec2 uv = gl_PointCoord * 2.0 - vec2(1.0);
    float d = dot(uv, uv);
    if (d > 1.0)
    {
        discard;
    }
    fNormal = vec3(uv.x, -uv.y, sqrt(1.f-d));
    fPosition = vPosition + fNormal * uPointSize;

	// update depth
	vec4 clipPos = transform.proj * vec4(fPosition, 1);
	float ndcZ = clipPos.z / clipPos.w;

	gl_FragDepth = (gl_DepthRange.diff * ndcZ + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
	renderPass();
}


layout(index = 0) subroutine(RenderPass) void GBufferPass(void)
{

	// pack g-buffer outputs
	fOutput0.rgb = vColor;
	fOutput0.w = 1.0;
	fOutput1.xyz = fNormal;
	fOutput1.w = metallic;
	fOutput2.xyz = fPosition;
	fOutput2.w = roughness;
}

layout(index = 1) subroutine(RenderPass) void ShadowMapPass(void)
{
	// transparent shadow map
	fOutput0 = albedo * (1 - albedo.a);
}

// OIT - Linked List
struct NodeType
{
	vec4 color;
	float depth;
	uint nextIndex;
};
