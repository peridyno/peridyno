#pragma once
#include <glad/gl.h>
#include <assert.h>
#include <stdio.h>

namespace dyno {

/*
* glAssert, assert function extracted from Flex
*/
inline void glAssert(const char* msg, long line, const char* file)
{
	struct glError
	{
		GLenum code;
		const char* name;
	};

	static const glError errors[] = {
		{ GL_NO_ERROR, "No Error" },
		{ GL_INVALID_ENUM, "Invalid Enum" },
		{ GL_INVALID_VALUE, "Invalid Value" },
		{ GL_INVALID_OPERATION, "Invalid Operation" }
	};

	GLenum e = glGetError();

	if (e == GL_NO_ERROR)
	{
		return;
	}
	else
	{
		const char* errorName = "Unknown error";

		// find error message
		for (unsigned int i = 0; i < sizeof(errors) / sizeof(glError); i++)
		{
			if (errors[i].code == e)
			{
				errorName = errors[i].name;
			}
		}

		printf("OpenGL: %s - error %s in %s at line %d\n", msg, errorName, file, int(line));
		assert(0);
	}
}

#if defined(NDEBUG)
#define glVerify(x) x
#else
#define glVerify(x) x//{x; glAssert(#x, __LINE__, __FILE__);}
#endif

}