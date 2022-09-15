layout(std140, binding = 0) uniform TransformUBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;
} uTransform;

layout(std140, binding = 1) uniform LightUBlock
{
	vec4 ambient;
	vec4 intensity;
	vec4 direction;
	vec4 camera;
} uLight;