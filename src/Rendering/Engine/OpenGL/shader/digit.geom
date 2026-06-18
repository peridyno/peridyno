#version 440
#extension GL_GOOGLE_include_directive: enable
#include "common.glsl"

layout (points) in;
layout (triangle_strip, max_vertices = 40) out;

layout(location = 1) in flat int vertexIndex[];
layout(location = 2) out vec2 fragTexCoord;

layout(location = 0) uniform float uDigitScale;
layout(location = 1) uniform vec2 uDigitOffset;
layout(location = 2) uniform float uDigitWidth;

void generateDigits() {
    int index = vertexIndex[0];
    vec4 basePos = gl_in[0].gl_Position;

    int digits[10]; 
    int numDigits = 0;
    
    if (index == 0) {
        digits[numDigits++] = 0;
    } else {
        while (index > 0 && numDigits < 10) {
            digits[numDigits] = index % 10;
            index = index / 10;
            numDigits++;
        }
    }

    for (int i = 0; i < numDigits / 2; i++) {
        int temp = digits[i];
        digits[i] = digits[numDigits - 1 - i];
        digits[numDigits - 1 - i] = temp;
    }

    float width = 0.4 * uDigitScale;
    float half_width = 0.5 * width;
    float height = uDigitScale;

    for (int i = 0; i < numDigits; i++) {
        int digit = digits[i];
        //vec4 pos = basePos + vec4(i * digitWidth * scale, uDigitOffset.y * uDigitScale, 0.0, 0.0);
        vec4 pos = basePos + vec4(uDigitOffset.x * uDigitScale + i * width, height, -uDigitScale, 0.0);

        float uvStart = digit * uDigitWidth;
        float uvEnd = uvStart + uDigitWidth;

        gl_Position = pos + vec4(-half_width, -height, 0.0, 0.0);
        fragTexCoord = vec2(uvStart, 0.0);
        EmitVertex();

        gl_Position = pos + vec4(half_width, -height, 0.0, 0.0);
        fragTexCoord = vec2(uvEnd, 0.0);
        EmitVertex();

        gl_Position = pos + vec4(-half_width, height, 0.0, 0.0);
        fragTexCoord = vec2(uvStart, 1.0);
        EmitVertex();

        gl_Position = pos + vec4(half_width, height, 0.0, 0.0);
        fragTexCoord = vec2(uvEnd, 1.0);
        EmitVertex();

        EndPrimitive();
    }
}

void main() {
    generateDigits();
}