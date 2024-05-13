#version 440

layout(location = 0) in vec2 texCoord;

layout(location = 0) uniform vec3 uColor0;
layout(location = 1) uniform vec3 uColor1;

layout(location = 0) out vec4 fragColor;

void main(void) 
{
    fragColor.rgb = mix(uColor0, uColor1, texCoord.y);
    fragColor.a = 1.0;
}

