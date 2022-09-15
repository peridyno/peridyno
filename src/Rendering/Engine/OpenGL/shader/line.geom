#version 440

layout (lines) in;
layout (line_strip, max_vertices = 2) out;

in VertexData {
	vec3 position;
	vec3 normal;
} gs_in[];

out VertexData {
	vec3 position;
	vec3 normal;
};

vec3 GenerateLineNormal()
{
   vec3 a = gs_in[0].position - gs_in[1].position;
   vec3 b = vec3(0, 0, 1);
   vec3 t = cross(a, b);
   return normalize(cross(t, a));
}  

void main() {    
    
    normal = GenerateLineNormal();

    position = gs_in[0].position;
    gl_Position = gl_in[0].gl_Position;  
    EmitVertex();
    
    position = gs_in[1].position;
    gl_Position = gl_in[1].gl_Position;  
    EmitVertex();
    
    EndPrimitive();
}

