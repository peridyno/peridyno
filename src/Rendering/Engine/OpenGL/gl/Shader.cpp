#include "Shader.h"

#include <glad/glad.h>
#include <fstream>

#include "shader_header.h"

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

	bool Shader::createFromFile(unsigned int type, const std::string& file)
	{
		// read file content
		std::ifstream ifs(file);
		std::string content((std::istreambuf_iterator<char>(ifs)),
			(std::istreambuf_iterator<char>()));
		const char* source = content.c_str();

		return createFromSource(type, content);
		return true;
	}

	void Shader::release()
	{
		if (0xFFFFFFFF == id)
			return;
		glDeleteShader(id);
		glCheckError();
	}


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

	bool ShaderFactory::initialize()
	{
		static bool initialized = false;
		if (initialized)
			return true;

		printf("Initialize ShaderFactory...\n");

		// pre load shader code snippets
		for (const auto& pair : ShaderSource) {
			std::string str = "/" + pair.first;
			const char* suffix = ".glsl";
			if (std::string::npos != str.rfind(suffix, str.length() - 5, 5))
			{
				printf("%s\n", pair.first.c_str());
				glNamedStringARB(GL_SHADER_INCLUDE_ARB, str.length(), str.c_str(), pair.second.length(), pair.second.c_str());
			}
		}

		initialized = true;
		return initialized;
	}


	Program ShaderFactory::createShaderProgram(const char* vs, const char* fs, const char* gs)
	{
		ShaderFactory::initialize();

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

