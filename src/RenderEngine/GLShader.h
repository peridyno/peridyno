#pragma once

#include "GLObject.h"
#include <string>
#include <glm/glm.hpp>

class GLShader : public GLObject
{
public:
	GLShader() {}
	bool createFromFile(unsigned int type, const std::string& path);
	bool createFromSource(unsigned int type, const std::string& src);
	void release();

protected:
	void create() {};
	
};

class GLShaderProgram : public GLObject
{
public:
	void create();
	void release();

	void attachShader(const GLShader& shader);
	bool link();

	void use();

	//
	void setFloat(const char* name, float v); 
	void setInt(const char* name, int v);
	void setVec4(const char* name, glm::vec4 v);
	void setVec3(const char* name, glm::vec3 v);
	void setVec2(const char* name, glm::vec2 v);
};

// public helpe function...
GLShaderProgram CreateShaderProgram(const char* vs,	const char* fs,	const char* gs = 0);