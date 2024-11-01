#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

// draw call input
layout(location = 0) in int pIndex;    
layout(location = 1) in int nIndex;
layout(location = 2) in int tIndex;

// shader storage buffer
layout(std430, binding = 8) buffer _Position { float positions[]; };
layout(std430, binding = 9) buffer _Normal   { float normals[];   };
layout(std430, binding = 10) buffer _TexCoord { float texCoords[]; };
layout(std430, binding = 11) buffer _Color    { float colors[]; };
layout(std430, binding = 12) buffer _Tangent    { float tangent[]; };
layout(std430, binding = 13) buffer _Bitangent    { float bitangent[]; };

// instance transform
layout(location = 3) in vec3 instance_translation;
layout(location = 4) in vec3 instance_scaling;
layout(location = 5) in mat3 instance_rotation; // location 5-7 is used for rotation matrix
layout(location = 8) in vec3 instance_color;	  // instance color

// uniform
// use vertex normal
layout(location = 1) uniform bool uVertexNormal = false;
// is instance rendering?
layout(location = 2) uniform bool uInstanced = false;
// color mode: 0 - object, 1 - vertex, 2 - texture
layout(location = 3) uniform int  uColorMode = 0;

// output
layout(location = 0) out VertexData {
	vec3 position;
	vec3 normal;
	vec3 color;
	vec3 texCoord;
	vec3 tangent;
    vec3 bitangent;
	int  instanceID;
} vs;

void main(void) {

	int offset = pIndex * 3;

	vec3 position = vec3(
        positions[offset], 
        positions[offset + 1], 
        positions[offset + 2]);

	// get color
	if(uColorMode == 0) {	
		vs.color = uMtl.color;
	}
	else if(uColorMode == 1) {
		vs.color = vec3(colors[offset], colors[offset + 1], colors[offset + 2]);
	}
	else {
		vs.color = vec3(1);
	}

	vs.instanceID = -1;

	mat4 modelMat = uRenderParams.model;
	if(uInstanced) {

		mat4 instanceTransform = mat4(1.0);
		// scale
		instanceTransform[0][0] = instance_scaling.x;
		instanceTransform[1][1] = instance_scaling.y;
		instanceTransform[2][2] = instance_scaling.z;
		// rotation
		instanceTransform = mat4(instance_rotation) * instanceTransform;
		// translate
		instanceTransform[3] = vec4(instance_translation, 1);
		// apply to model transform matrix
		modelMat = modelMat * instanceTransform;

		vs.instanceID = gl_InstanceID;

		if(uColorMode == 0) {	
			vs.color = instance_color;
		}
		if(uColorMode == 1) {	
			vs.color = uMtl.color;
		}
	}
	mat4 MV = uRenderParams.view * modelMat;

	//vec4 worldPos = modelMat * vec4(position, 1.0);
	//vec4 cameraPos = uRenderParams.view * worldPos;
	vec4 cameraPos = MV * vec4(position, 1.0);

	vs.position = cameraPos.xyz;

	// if use vertex normal, than Normal data should be valid
	if(uVertexNormal)
	{
		mat4 N = transpose(inverse(MV));
		// use separate normal index
		if(nIndex != -1) 
			offset = nIndex * 3;

		vec4 n = vec4(normals[offset], normals[offset + 1], normals[offset + 2], 0.0);
		vec4 tn = vec4(tangent[offset], tangent[offset + 1], tangent[offset + 2], 0.0);
		vec4 tb = vec4(bitangent[offset], bitangent[offset + 1], bitangent[offset + 2], 0.0);
		vs.normal = (N * n).xyz;
		vs.tangent = (N * tn).xyz;
		vs.bitangent = (N * tb).xyz;
	}
	else{
		vs.normal = vec3(0);
		vs.tangent = vec3(0);
		vs.bitangent = vec3(0);
	}

	// texture coordinates
	if(tIndex != -1 && uColorMode == 2)
	{
		offset = tIndex * 2;
		vs.texCoord = vec3(texCoords[offset], texCoords[offset+1], 1);
	}
	else
	{
		vs.texCoord = vec3(-1);
	}
	

	gl_Position = uRenderParams.proj * cameraPos;
}