#version 460

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VertexData
{
	vec3 position;
	vec3 color;
} gs_in[];

out VertexData
{
	vec3 position;
	vec3 normal;
	vec3 color;
} gs_out;

vec3 GetFaceNormal()
{
   vec3 a = vec3(gs_in[0].position) - vec3(gs_in[1].position);
   vec3 b = vec3(gs_in[2].position) - vec3(gs_in[1].position);
   return normalize(cross(a, b));
}  

void main() {    
    
    gs_out.normal = GetFaceNormal();

    gs_out.position = gs_in[0].position;
    gs_out.color = gs_in[0].color;
    gl_Position = gl_in[0].gl_Position;  
    EmitVertex();
    
    gs_out.position = gs_in[1].position;
    gs_out.color = gs_in[1].color;
    gl_Position = gl_in[1].gl_Position;  
    EmitVertex();
    
    gs_out.position = gs_in[2].position;
    gs_out.color = gs_in[2].color;
    gl_Position = gl_in[2].gl_Position;  
    EmitVertex();

    EndPrimitive();
}

