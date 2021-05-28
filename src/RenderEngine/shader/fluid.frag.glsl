#version 440

// center position
in vec3 vOrigin;

uniform float uPointRadius = 0.05;

out vec4 fragOut;

void main(void)
{
    // sphere...
    vec2 temp = gl_PointCoord * 2.0 - vec2(1.0);

    float z2 = dot(temp, temp);
    if (z2 > 1.0)
    {
        discard;
    }

    vec3 N = normalize(vec3(temp.x, -temp.y, sqrt(1-z2)));
    vec3 P = N * uPointRadius;

    // position
    vec3 position = vOrigin + P;
    
    // near, -far, 1
    fragOut.xyz = vec3(vOrigin.z + P.z, -(vOrigin.z - P.z), uPointRadius);

    // thickness
    fragOut.w = P.z * 2;
}