#include "GLShader.h"

#include <glad/glad.h>
#include "Utility.h"

#include <fstream>


bool GLShader::createFromSource(unsigned int type, const std::string& src)
{
	const char* source = src.c_str();
	// create shader
	GLuint shader = glCreateShader(type);
	glShaderSource(shader, 1, &source, 0);
	glCompileShader(shader);

	// check error
	GLint success;
	GLchar infoLog[2048];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(shader, 2048, NULL, infoLog);
		//SPDLOG_ERROR("Shader Compiling Error:\n{}",std::string(infoLog));
		return false;
	}

	id = shader;
	glCheckError();

	return true;
}

bool GLShader::createFromFile(unsigned int type, const std::string& file)
{
	// read file content
	std::ifstream ifs(file);
	std::string content((std::istreambuf_iterator<char>(ifs)),
		(std::istreambuf_iterator<char>()));
	const char* source = content.c_str();

	return createFromSource(type, content);
	return true;
}

void GLShader::release()
{
	if (0xFFFFFFFF == id)
		return;
	glDeleteShader(id);
	glCheckError();
}

void GLShaderProgram::create()
{
	id = glCreateProgram();
}

void GLShaderProgram::release()
{
	glDeleteProgram(id);
}

void GLShaderProgram::attachShader(const GLShader& shader)
{
	glAttachShader(id, shader.id);
}

bool GLShaderProgram::link()
{
	glLinkProgram(id);

	GLint success;
	GLchar infoLog[2048];
	glGetProgramiv(id, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(id, 2048, NULL, infoLog);
		//SPDLOG_ERROR("Shader Program Linking Error:\n", std::string(infoLog));
		return false;
	}

	return true;
}

void GLShaderProgram::use()
{
	glUseProgram(id);
}

void GLShaderProgram::setFloat(const char* name, float v)
{
	GLuint location = glGetUniformLocation(id, name);
	glUniform1f(location, v);
}

void GLShaderProgram::setInt(const char* name, int v)
{
	GLuint location = glGetUniformLocation(id, name);
	glUniform1i(location, v);
}

void GLShaderProgram::setVec2(const char* name, glm::vec2 v)
{
	GLuint location = glGetUniformLocation(id, name);
	glUniform2f(location, v.x, v.y);
}

void GLShaderProgram::setVec3(const char* name, glm::vec3 v)
{
	GLuint location = glGetUniformLocation(id, name);
	glUniform3f(location, v.x, v.y, v.z);
}

void GLShaderProgram::setVec4(const char* name, glm::vec4 v)
{
	GLuint location = glGetUniformLocation(id, name);
	glUniform4f(location, v.x, v.y, v.z, v.w);
}


#include "shader_header.h"

GLShaderProgram CreateShaderProgram(const char* vs, const char* fs, const char* gs)
{
	GLShaderProgram program;
	program.create();

	GLShader vshader;
	GLShader fshader;
	GLShader gshader;

	if (vs != 0)
	{
		const std::string& src = ShaderSource.at(vs);
		if (vshader.createFromSource(GL_VERTEX_SHADER, src))
			program.attachShader(vshader);
	}

	if (fs != 0)
	{
		const std::string& src = ShaderSource.at(fs);
		if (fshader.createFromSource(GL_FRAGMENT_SHADER, src))
			program.attachShader(fshader);
	}

	if (gs != 0)
	{
		const std::string& src = ShaderSource.at(gs);
		if (gshader.createFromSource(GL_GEOMETRY_SHADER, src))
			program.attachShader(gshader);
	}

	program.link();

	vshader.release();
	fshader.release();
	gshader.release();
	return program;
}
