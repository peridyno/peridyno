#version 440

// particle properties
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_velocity;
layout(location = 2) in vec3 in_force;

out vec3 vPosition;
out vec3 vColor;

uniform float uPointSize;

layout (std140, binding=0) uniform TransformUniformBlock
{
	mat4 model;
	mat4 view;
	mat4 proj;

	int width;
	int height;
} transform;

layout(std140, binding = 2) uniform MaterialUniformBlock
{
	vec4  albedo;
	float metallic;
	float roughness;

	int   colorMode;
	float colorMin;
	float colorMax;

	int   shadowMode;
};

// color map
vec3 JetColor(float v, float vmin, float vmax){

    float x = clamp((v-vmin)/(vmax-vmin), 0, 1);
    float r = clamp(-4*abs(x-0.75) + 1.5, 0, 1);
    float g = clamp(-4*abs(x-0.50) + 1.5, 0, 1);
    float b = clamp(-4*abs(x-0.25) + 1.5, 0, 1);
    return vec3(r,g,b);
}

vec3 HeatColor(float v,float vmin,float vmax){
    float x = clamp((v-vmin)/(vmax-vmin), 0, 1);
    float r = clamp(-4*abs(x-0.75) + 2, 0, 1);
    float g = clamp(-4*abs(x-0.50) + 2, 0, 1);
    float b = clamp(-4*abs(x) + 2, 0, 1);
    return vec3(r,g,b);
}

vec4 get_color()
{
	// color by velocity
	if(colorMode == 1)
		return vec4(JetColor(length(in_velocity), colorMin, colorMax), 1);
	if(colorMode == 2)
		return vec4(HeatColor(length(in_velocity), colorMin, colorMax), 1);
	if(colorMode == 3)
		return vec4(JetColor(length(in_force), colorMin, colorMax), 1);
	if(colorMode == 4)
		return vec4(HeatColor(length(in_force), colorMin, colorMax), 1);
	// default 
	return albedo;
}


void main(void) 
{
	vec4 worldPos = transform.model * vec4(in_position, 1.0);
	vec4 cameraPos = transform.view * worldPos;

	vPosition = cameraPos.xyz;
	vColor = get_color().rgb;
	
	gl_Position = transform.proj * cameraPos; 
		
	// point size
	vec4 projCorner = transform.proj * vec4(uPointSize, uPointSize, cameraPos.z, cameraPos.w);
	gl_PointSize = transform.width * projCorner.x / projCorner.w;
}