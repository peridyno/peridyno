#include "Shader.h"

#include <glad/glad.h>
#include <fstream>

namespace gl {

	bool Shader::createFromSource(unsigned int type, const std::string& src)
	{
		const static char* path = "/";

		const char* source = src.c_str();
		// create shader
		GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &source, 0);
		glCompileShaderIncludeARB(shader, 1, &path, 0);

		// check error
		GLint success;
		GLchar infoLog[2048];
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 2048, NULL, infoLog);
			printf("Shader Compiling Error:\n%s\n", infoLog);
			return false;
		}

		id = shader;
		glCheckError();

		return true;
	}

	bool Shader::createFromSPIRV(unsigned int type, const void* src, const size_t len)
	{
		// create shader
		GLuint shader = glCreateShader(type);

		glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V, src, len);
		glSpecializeShader(shader, "main", 0, 0, 0);

		// check error
		GLint success;
		GLchar infoLog[2048];
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 2048, NULL, infoLog);
			printf("Shader Compiling Error:\n%s\n", infoLog);
			return false;
		}

		id = shader;
		glCheckError();

		return true;
	}

	bool Shader::createFromFile(unsigned int type, const std::string& file)
	{
		// read file content
		std::ifstream ifs(file);
		std::string content((std::istreambuf_iterator<char>(ifs)),
			(std::istreambuf_iterator<char>()));
		const char* source = content.c_str();

		if (createFromSource(type, content))
			return true;

		printf("Failed to compile shader from file: %s\n", file.c_str());
		return false;
	}

	void Shader::release()
	{
		if (0xFFFFFFFF == id)
			return;
		glDeleteShader(id);
		glCheckError();

		// reset object id
		id = GL_INVALID_INDEX;
	}


	void Program::create()
	{
		id = glCreateProgram();
	}

	void Program::release()
	{
		glDeleteProgram(id);

		// reset object id
		id = GL_INVALID_INDEX;
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


	Program* Program::createProgram(const char* vs, const char* fs, const char* gs)
	{
		Program* program = new Program;
		program->create();

		Shader vshader;
		Shader fshader;
		Shader gshader;

		if (vs != 0)
		{
			if (vshader.createFromSource(GL_VERTEX_SHADER, vs))
				program->attachShader(vshader);
			else
				printf("Failed to compile shader: %s\n", vs);
		}

		if (fs != 0)
		{
			if (fshader.createFromSource(GL_FRAGMENT_SHADER, fs))
				program->attachShader(fshader);
			else
				printf("Failed to compile shader: %s\n", fs);
		}

		if (gs != 0)
		{
			if (gshader.createFromSource(GL_GEOMETRY_SHADER, gs))
				program->attachShader(gshader);
			else
				printf("Failed to compile shader: %s\n", gs);
		}

		if (!program->link())
		{
			printf("Failed to link shader program: %s\n", fs);
		}

		vshader.release();
		fshader.release();
		gshader.release();

		return program;
	}

	Program* Program::createProgramSPIRV(
		const void* vs, size_t vs_len,
		const void* fs, size_t fs_len,
		const void* gs, size_t gs_len)
	{
		Program* program = new Program;
		program->create();

		Shader vshader;
		Shader fshader;
		Shader gshader;

		if (vs != 0)
		{
			if (vshader.createFromSPIRV(GL_VERTEX_SHADER, vs, vs_len))
				program->attachShader(vshader);
			else
				printf("Failed to compile shader: %s\n", vs);
		}

		if (fs != 0)
		{
			if (fshader.createFromSPIRV(GL_FRAGMENT_SHADER, fs, fs_len))
				program->attachShader(fshader);
			else
				printf("Failed to compile shader: %s\n", fs);
		}

		if (gs != 0)
		{
			if (gshader.createFromSPIRV(GL_GEOMETRY_SHADER, gs, gs_len))
				program->attachShader(gshader);
			else
				printf("Failed to compile shader: %s\n", gs);
		}

		if (!program->link())
		{
			printf("Failed to link shader program: %s\n", fs);
		}

		vshader.release();
		fshader.release();
		gshader.release();

		return program;
	}
}

