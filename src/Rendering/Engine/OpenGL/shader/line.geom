#version 440

#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

layout(lines) in;
layout(triangle_strip, max_vertices = 72) out;

layout(location = 0) uniform int	uEdgeMode = 1;
layout(location = 1) uniform float	uRadius = 0.003;

layout(location = 0) in VertexData {
	vec3 position;
	vec3 normal;
	vec3 color;
	int  instanceID; 	// not used for this type
} gs_in[];

layout(location = 0) out VertexData {
	vec3 position;
	vec3 normal;
	vec3 color;
	vec3 texCoord;
	int  instanceID; 	// not used for this type
};

void line() { 

    // get line normal
    vec3 a = gs_in[0].position - gs_in[1].position;
    vec3 b = vec3(0, 0, 1);
    vec3 t = cross(a, b);
    normal = normalize(cross(t, a));
    
    // emit primitives
    color = gs_in[0].color;
    position = gs_in[0].position;
    gl_Position = gl_in[0].gl_Position;  
    EmitVertex();
    
    color = gs_in[1].color;
	
    position = gs_in[1].position;
    gl_Position = gl_in[1].gl_Position;  
    EmitVertex();

    position = gs_in[1].position;
    gl_Position = gl_in[1].gl_Position;  
    EmitVertex();
    
    EndPrimitive();
}


const uint nSectors = 6;

#define M_PI 3.14159265358979323846

void cylinder() { 

	vec3 p0 = gs_in[0].position;
	vec3 p1 = gs_in[1].position;

	// get transform
	vec3 d = normalize(p1 - p0);

	// build rotation matrix
	mat3 R;

	// hard code here...
	const vec3 v = vec3(0, 0, 1);
	
	// rotation axis
	if(dot(d, v) == -1.0) {
		R = mat3(-1, 0, 0,  0, -1, 0,  0, 0, -1);
	}
	else {
		vec3 A = normalize(d + v);

		R = mat3(	A.x * A.x * 2 - 1, A.x * A.y * 2,     A.x * A.z * 2, 
					A.x * A.y * 2,     A.y * A.y * 2 - 1, A.y * A.z * 2, 
					A.x * A.z * 2,     A.y * A.z * 2,     A.z * A.z * 2 - 1);
	}

	const float	len = length(p1 - p0);
	const vec3	scale = vec3(uRadius, uRadius, len);

	const float fSectors = nSectors;
	const float sectorStep = 2 * M_PI / fSectors;

	for (int i = 0; i < nSectors; i++) {

		float x0 = cos(i * sectorStep);
		float y0 = sin(i * sectorStep);
		float x1 = cos((i + 1) * sectorStep);
		float y1 = sin((i + 1) * sectorStep);

		vec3 n0 = R * vec3(x0, y0, 0.f);
		vec3 n1 = R * vec3(x1, y1, 0.f);
		
		vec3 p00 = p0 + R * (vec3(x0, y0, 0.f) * scale);
		vec3 p01 = p0 + R * (vec3(x0, y0, 1.f) * scale);
		vec3 p10 = p0 + R * (vec3(x1, y1, 0.f) * scale);
		vec3 p11 = p0 + R * (vec3(x1, y1, 1.f) * scale);

		vec4 p00p = uRenderParams.proj * vec4(p00, 1);
		vec4 p01p = uRenderParams.proj * vec4(p01, 1);
		vec4 p10p = uRenderParams.proj * vec4(p10, 1);
		vec4 p11p = uRenderParams.proj * vec4(p11, 1);
		
		color = gs_in[0].color;

		// 0
		normal = -d;		
		position = p0;
		gl_Position = gl_in[0].gl_Position;
		EmitVertex();
		// 00
		position = p00;
		gl_Position = p00p;
		EmitVertex();
		// 10
		position = p10;
		gl_Position = p10p;
		EmitVertex();
		//00
		position = p00;
		normal = n0;
		gl_Position = p00p;
		EmitVertex();
		//10
		position = p10;
		normal = n1;
		gl_Position = p10p;
		EmitVertex();

		
		color = gs_in[1].color;
		//01
		position = p01;
		normal = n0;
		gl_Position = p01p;
		EmitVertex();
		// 11
		position = p11;
		normal = n1;
		gl_Position = p11p;
		EmitVertex();
		// 01		
		normal = d;
		position = p01;
		gl_Position = p01p;
		EmitVertex();
		// 11
		position = p11;
		gl_Position = p11p;
		EmitVertex();
		// 1
		position = p1;
		gl_Position = gl_in[1].gl_Position;
		EmitVertex();
		EndPrimitive();
	}
}

void main()
{
	if(uEdgeMode == 1)
		cylinder();
	else
		line();
}