#version 440

layout(location = 0) out vec4 FragColor;

layout(location = 0) uniform vec4 uColor = vec4(0.75);

void main(void) {  
	FragColor = uColor;
}  