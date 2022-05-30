#include "Shader.h"

#include <glad/glad.h>
#include <fstream>

namespace gl {

	bool Shader::createFromSource(unsigned int type, const std::string& src)
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

}

