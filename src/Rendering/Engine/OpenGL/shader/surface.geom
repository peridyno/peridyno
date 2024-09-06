// We use a geometry shader to handle normals
#version 460

#extension GL_GOOGLE_include_directive: enable

#include "common.glsl"

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

// whether to use vertex normal
layout(location = 1) uniform bool uVertexNormal = false;

layout(location=0) in VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;    
	vec3 texCoord;
    vec3 tangent;
    vec3 bitangent;
    int  instanceID;
} gs_in[];

layout(location=0) out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;    
	vec3 texCoord;
    vec3 tangent;
    vec3 bitangent;
    int  instanceID;
} gs_out;


void main() {   
    if(uVertexNormal)
    {
        gs_out.position = gs_in[0].position;
        gs_out.normal = gs_in[0].normal;
        gs_out.color = gs_in[0].color;
        gs_out.texCoord = gs_in[0].texCoord;
        gs_out.tangent = gs_in[0].tangent;
        gs_out.bitangent = gs_in[0].bitangent;
        gs_out.instanceID = gs_in[0].instanceID;
        gl_Position = gl_in[0].gl_Position;  
        EmitVertex();
        
        gs_out.position = gs_in[1].position;
        gs_out.normal = gs_in[1].normal;
        gs_out.color = gs_in[1].color;
        gs_out.texCoord = gs_in[1].texCoord;
        gs_out.tangent = gs_in[1].tangent;
        gs_out.bitangent = gs_in[1].bitangent;
        gs_out.instanceID = gs_in[1].instanceID;
        gl_Position = gl_in[1].gl_Position;  
        EmitVertex();
        
        gs_out.position = gs_in[2].position;
        gs_out.normal = gs_in[2].normal;
        gs_out.color = gs_in[2].color;
        gs_out.texCoord = gs_in[2].texCoord;
        gs_out.tangent = gs_in[2].tangent;
        gs_out.bitangent = gs_in[2].bitangent;
        gs_out.instanceID = gs_in[2].instanceID;
        gl_Position = gl_in[2].gl_Position;  
        EmitVertex();
    }
    else 
    {
        vec3 v0 = gs_in[0].position;
        vec3 v1 = gs_in[1].position;
        vec3 v2 = gs_in[2].position;

        vec2 uv0 = gs_in[0].texCoord.st;
        vec2 uv1 = gs_in[1].texCoord.st;
        vec2 uv2 = gs_in[2].texCoord.st;

        // Edges of the triangle : position delta
        vec3 deltaPos1 = v1-v0;
        vec3 deltaPos2 = v2-v0;

        // UV delta
        vec2 deltaUV1 = uv1-uv0;
        vec2 deltaUV2 = uv2-uv0;

        // Tangent space
        float r = 1.0 / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
        gs_out.tangent = (deltaPos1 * deltaUV2.y   - deltaPos2 * deltaUV1.y)*r;
        gs_out.bitangent = (deltaPos2 * deltaUV1.x   - deltaPos1 * deltaUV2.x)*r;

        // CCW
        vec3 a = vec3(gs_in[0].position) - vec3(gs_in[1].position);
        vec3 b = vec3(gs_in[2].position) - vec3(gs_in[1].position);
        gs_out.normal = normalize(cross(b, a));

        gs_out.position = gs_in[0].position;
        gs_out.color = gs_in[0].color;
        gs_out.texCoord = gs_in[0].texCoord;
        gs_out.instanceID = gs_in[0].instanceID;
        gl_Position = gl_in[0].gl_Position;  
        EmitVertex();
    
        gs_out.position = gs_in[1].position;
        gs_out.color = gs_in[1].color;
        gs_out.texCoord = gs_in[1].texCoord;
        gs_out.instanceID = gs_in[1].instanceID;
        gl_Position = gl_in[1].gl_Position;  
        EmitVertex();
    
        gs_out.position = gs_in[2].position;
        gs_out.color = gs_in[2].color;
        gs_out.texCoord = gs_in[2].texCoord;
        gs_out.instanceID = gs_in[2].instanceID;
        gl_Position = gl_in[2].gl_Position;  
        EmitVertex();
    }

    EndPrimitive();
}

