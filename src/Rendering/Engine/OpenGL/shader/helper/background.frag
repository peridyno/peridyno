#version 440

in vec2 texCoord;

uniform vec3 uColor0;
uniform vec3 uColor1;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out int  fragIndex;

void main(void) 
{
    fragColor.rgb = mix(uColor0, uColor1, texCoord.y);
    fragColor.a = 1.0;
    fragIndex = -1;
}

