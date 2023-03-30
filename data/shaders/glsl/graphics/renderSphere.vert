#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec4 inTransRotation;
layout (location = 4) in vec3 inTransPosition;
layout (location = 5) in float radius;


layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec3 outViewVec;
layout (location = 3) out vec3 outLightVec;


layout (binding = 0) uniform UBO 
{
	mat4 projection;
	mat4 modelview;
	vec4 lightPos;
} ubo;

out gl_PerVertex
{
	vec4 gl_Position;
};

vec3 quat_rotate(vec4 quat, vec3 v)
{
  // Extract the vector part of the quaternion
  vec3 u = quat.xyz;

  // Extract the scalar part of the quaternion
  float s = quat.w;

  // Do the math
  return    2.0f * dot(u, v) * u
      + (s*s - dot(u, u)) * v
      + 2.0 * s * cross(u, v);
}

void main () 
{
	outUV = inUV;
	outNormal = quat_rotate(inTransRotation, inNormal).xyz;
	vec3 iPos = quat_rotate(inTransRotation, inPos * vec3(radius, radius, radius)).xyz + inTransPosition;
	vec4 eyePos = ubo.modelview * vec4(iPos, 1.0); 
	gl_Position = ubo.projection * eyePos;
	vec4 pos = vec4(iPos, 1.0);
	vec3 lPos = ubo.lightPos.xyz;
	outLightVec = lPos - pos.xyz;
	outViewVec = -pos.xyz;		
}