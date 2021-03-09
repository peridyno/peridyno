#version 330 compatibility
layout (location = 0) in vec2 corner;
layout (location = 2) in vec3 instance_pos;
layout (location = 3) in vec3 instance_color;

uniform float sprite_size;

out vec3 frag_color;
out vec2 tex_coord;
out vec3 frag_pos;
out float out_sprite_size;

void main()
{
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(instance_pos, 1.0));

    frag_pos = vec3(posEye.xy + corner * sprite_size, posEye.z);

    frag_color = instance_color;
    gl_Position = gl_ProjectionMatrix*vec4(frag_pos, 1.0);

    tex_coord = corner;
    out_sprite_size = sprite_size;
}