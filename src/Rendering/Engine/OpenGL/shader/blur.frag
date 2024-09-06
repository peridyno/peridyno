#version 440

layout(location = 0) in  vec2 texCoord;
layout(location = 0) out vec4 outColor;

layout(location = 0) uniform vec2		uScale;
layout(binding = 5)  uniform sampler2D	uTexSource;

void main()
{
	vec4 color = vec4(0.0);
	color += texture(uTexSource, texCoord.st + vec2( -3.0 * uScale.x, -3.0 * uScale.y)) * 0.015625;
	color += texture(uTexSource, texCoord.st + vec2( -2.0 * uScale.x, -2.0 * uScale.y)) * 0.09375;
	color += texture(uTexSource, texCoord.st + vec2( -1.0 * uScale.x, -1.0 * uScale.y)) * 0.234375;
	color += texture(uTexSource, texCoord.st + vec2( 0.0, 0.0)) * 0.3125;
	color += texture(uTexSource, texCoord.st + vec2( 1.0 * uScale.x,  1.0 * uScale.y)) * 0.234375;
	color += texture(uTexSource, texCoord.st + vec2( 2.0 * uScale.x,  2.0 * uScale.y)) * 0.09375;
	color += texture(uTexSource, texCoord.st + vec2( 3.0 * uScale.x, -3.0 * uScale.y)) * 0.015625;
	outColor = vec4(color.xyz, 1.0);
}
