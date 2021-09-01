#include "Program.h"

#include <glad/glad.h>

namespace gl {


	void Program::create()
	{
		id = glCreateProgram();
	}

	void Program::release()
	{
		glDeleteProgram(id);
	}

	void Program::attachShader(const Shader& shader)
	{
		glAttachShader(id, shader.id);
	}

	bool Program::link()
	{
		glLinkProgram(id);

		GLint success;
		GLchar infoLog[2048];
		glGetProgramiv(id, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(id, 2048, NULL, infoLog);
			printf("Shader Program Linking Error:\n%s\n", infoLog);
			return false;
		}

		return true;
	}

	void Program::use()
	{
		glUseProgram(id);
	}

	void Program::setFloat(const char* name, float v)
	{
		GLuint location = glGetUniformLocation(id, name);
		glUniform1f(location, v);
	}

	void Program::setInt(const char* name, int v)
	{
		GLuint location = glGetUniformLocation(id, name);
		glUniform1i(location, v);
	}

	void Program::setVec2(const char* name, dyno::Vec2f v)
	{
		GLuint location = glGetUniformLocation(id, name);
		glUniform2f(location, v[0], v[1]);
	}

	void Program::setVec3(const char* name, dyno::Vec3f v)
	{
		GLuint location = glGetUniformLocation(id, name);
		glUniform3f(location, v[0], v[1], v[2]);
	}

	void Program::setVec4(const char* name, dyno::Vec4f v)
	{
		GLuint location = glGetUniformLocation(id, name);
		glUniform4f(location, v[0], v[1], v[2], v[3]);
	}


#include "shader_header.h"

	Program CreateShaderProgram(const char* vs, const char* fs, const char* gs)
	{
		Program program;
		program.create();

		Shader vshader;
		Shader fshader;
		Shader gshader;

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
}