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
    int  instanceID;
} gs_in[];

layout(location=0) out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;    
	vec3 texCoord;
    int  instanceID;
} gs_out;


void main() {   
    
    if(uVertexNormal)
    {
        gs_out.position = gs_in[0].position;
        gs_out.normal = gs_in[0].normal;
        gs_out.color = gs_in[0].color;
        gs_out.texCoord = gs_in[0].texCoord;
        gs_out.instanceID = gs_in[0].instanceID;
        gl_Position = gl_in[0].gl_Position;  
        EmitVertex();
        
        gs_out.position = gs_in[1].position;
        gs_out.normal = gs_in[1].normal;
        gs_out.color = gs_in[1].color;
        gs_out.texCoord = gs_in[1].texCoord;
        gs_out.instanceID = gs_in[1].instanceID;
        gl_Position = gl_in[1].gl_Position;  
        EmitVertex();
        
        gs_out.position = gs_in[2].position;
        gs_out.normal = gs_in[2].normal;
        gs_out.color = gs_in[2].color;
        gs_out.texCoord = gs_in[2].texCoord;
        gs_out.instanceID = gs_in[2].instanceID;
        gl_Position = gl_in[2].gl_Position;  
        EmitVertex();
    }
    else 
    {        
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

