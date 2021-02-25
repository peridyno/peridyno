#version 330 compatibility
out vec4 FragColor;

in vec3 frag_color;
in vec3 frag_pos;
in vec2 tex_coord;
in float out_sprite_size;

void main()
{
    vec3 normal;
    //normal.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    normal.xy = tex_coord;
    float radius = dot(normal.xy, normal.xy);
    if (radius > 1.0) discard;
    normal.z = sqrt(1.0 - radius);

    vec3 light_dir = vec3(0, 1, 0);
    float diffuse_factor = max(dot(light_dir, normal), 0);
    vec3 diffuse = diffuse_factor * frag_color;

    diffuse = vec3(normal.z) * frag_color;
    float transparency = 1.0;
    if(radius > 0.8) transparency = 1.0 - 5 * (radius - 0.8);
    gl_FragColor = vec4(diffuse, transparency);

    float far=gl_DepthRange.far; 
    float near=gl_DepthRange.near;

    vec4 clip_space_pos = gl_ProjectionMatrix * vec4(frag_pos.xy, frag_pos.z + out_sprite_size * normal.z, 1.0);

    float ndc_depth = clip_space_pos.z / clip_space_pos.w;

    float depth = (((far-near) * ndc_depth) + near + far) / 2.0;
    gl_FragDepth = depth;
}