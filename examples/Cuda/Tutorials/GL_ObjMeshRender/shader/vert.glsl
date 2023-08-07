#version 430

layout(std140, binding = 0) uniform Transforms
{
	// transform
	mat4 model;
	mat4 view;
	mat4 proj;
	// illumination
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
	// parameters
	int width;
	int height;
	int index;
} uRenderParams;

// shader storage buffer
layout(std430, binding = 0) buffer _Position { float positions[]; };
layout(std430, binding = 1) buffer _Normal   { float normals[];   };
layout(std430, binding = 2) buffer _TexCoord { float texCoords[]; };
layout(std430, binding = 2) buffer _Color    { float colors[]; };

// in
layout(location = 0) in int pIndex;    
layout(location = 1) in int nIndex;
layout(location = 2) in int tIndex;

// vertex shader output
out VertexData
{
	vec3 vPosition;
	vec3 vNormal;
    vec3 vTexCoord;
    vec3 vColor;
};

void main(void) {

	vec4 position = vec4(
        positions[pIndex * 3], 
        positions[pIndex * 3 + 1], 
        positions[pIndex * 3 + 2],
        1.0);
    position = uRenderParams.view * uRenderParams.model * position;
	vPosition= position.xyz;
    
    vNormal   = vec3(0);
    vTexCoord = vec3(0);

    if(nIndex >= 0) {
        vec4 normal = vec4(
            normals[nIndex * 3], 
            normals[nIndex * 3 + 1], 
            normals[nIndex * 3 + 2],
            0.0);

        mat4 MV = uRenderParams.view * uRenderParams.model;
        mat4 N = transpose(inverse(MV));
        vNormal = (N * normal).xyz;
    }        

    if(tIndex >= 0) {
        vec3 texCoord = vec3(
            texCoords[tIndex * 2],
            texCoords[tIndex * 2 + 1],
            1
        );
        vTexCoord = texCoord;
    }

    vColor = vec3(1);
        

	gl_Position = uRenderParams.proj * position;
}