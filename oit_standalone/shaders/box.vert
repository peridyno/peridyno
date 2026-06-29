#version 430

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModel;

out vec3 vNormalView;

void main()
{
	gl_Position = uMVP * vec4(aPos, 1.0);
	// view-space-ish normal (uModel already includes view here)
	vNormalView = mat3(uModel) * aNormal;
}
