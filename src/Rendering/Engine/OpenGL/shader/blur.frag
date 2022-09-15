#version 440

in vec2 texcoord;

layout(binding = 5) uniform sampler2D uTexSource;
uniform vec2 uScale;

out vec4 outColor;

void main()
{
	vec4 color = vec4(0.0);
	color += texture2D(uTexSource, texcoord.st + vec2( -3.0 * uScale.x, -3.0 * uScale.y)) * 0.015625;
	color += texture2D(uTexSource, texcoord.st + vec2( -2.0 * uScale.x, -2.0 * uScale.y)) * 0.09375;
	color += texture2D(uTexSource, texcoord.st + vec2( -1.0 * uScale.x, -1.0 * uScale.y)) * 0.234375;
	color += texture2D(uTexSource, texcoord.st + vec2( 0.0, 0.0)) * 0.3125;
	color += texture2D(uTexSource, texcoord.st + vec2( 1.0 * uScale.x,  1.0 * uScale.y)) * 0.234375;
	color += texture2D(uTexSource, texcoord.st + vec2( 2.0 * uScale.x,  2.0 * uScale.y)) * 0.09375;
	color += texture2D(uTexSource, texcoord.st + vec2( 3.0 * uScale.x, -3.0 * uScale.y)) * 0.015625;
	outColor = vec4(color.xyz, 1.0);
};