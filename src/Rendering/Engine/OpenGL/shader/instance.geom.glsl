#version 440

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VertexData
{
	vec3 position;
	vec3 normal;
} gs_in[];

out VertexData
{
	vec3 position;
	vec3 normal;
};

vec3 GenerateSurfaceNormal()
{
   vec3 a = vec3(gs_in[0].position) - vec3(gs_in[1].position);
   vec3 b = vec3(gs_in[2].position) - vec3(gs_in[1].position);
   return normalize(cross(a, b));
}  

void main() {    
    
    normal = GenerateSurfaceNormal();

    position = gs_in[0].position;
    gl_Position = gl_in[0].gl_Position;  
    EmitVertex();
    
    position = gs_in[1].position;
    gl_Position = gl_in[1].gl_Position;  
    EmitVertex();
    
    position = gs_in[2].position;
    gl_Position = gl_in[2].gl_Position;  
    EmitVertex();

    EndPrimitive();
}

