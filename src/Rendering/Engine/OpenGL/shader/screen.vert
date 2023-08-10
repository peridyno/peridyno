#version 440

layout(location = 0) in  vec3 position;
layout(location = 0) out vec2 texCoord;

void main(void) {	
	gl_Position = vec4(position, 1);
	texCoord = position.xy * 0.5 + 0.5;
}