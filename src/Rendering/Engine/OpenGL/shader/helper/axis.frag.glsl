#version 440

in  vec3 color;
out vec4 FragColor;

void main(void) {  
	FragColor.rgb = color;
	FragColor.a = 1.0;
}  