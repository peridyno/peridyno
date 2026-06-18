#version 440
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"

layout(location = 2) in vec2 fragTexCoord;
layout(location = 0) out vec4 FragColor;

layout(binding = 10) uniform sampler2D uDigitSampler;

void main() {
    vec4 color = texture(uDigitSampler, fragTexCoord);

    if (color.a < 0.1) discard;

    FragColor = color;
}