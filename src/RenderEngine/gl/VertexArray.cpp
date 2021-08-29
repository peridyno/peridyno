#include "VertexArray.h"

#include <glad/glad.h>
#include <vector>

namespace gl
{
	void VertexArray::create()
	{
		glGenVertexArrays(1, &id);
	}

	void VertexArray::release()
	{
		glDeleteVertexArrays(1, &id);
	}

	void VertexArray::bind()
	{
		glBindVertexArray(id);
	}

	void VertexArray::unbind()
	{
		glBindVertexArray(0);
	}

	void VertexArray::bindIndexBuffer(Buffer* buffer)
	{
		this->bind();
		if (buffer == 0)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
		}
		else
		{
			buffer->bind();
		}
		this->unbind();
	}


	void VertexArray::bindVertexBuffer(Buffer* buffer, int index,
		int size, int type, int stride, int offset, int divisor)
	{
		this->bind();
		if (buffer == 0)
		{
			glDisableVertexAttribArray(index);
		}
		else
		{
			buffer->bind();
			glEnableVertexAttribArray(index);
			glVertexAttribPointer(index, size, type, GL_FALSE, stride, (void*)offset);
			glVertexAttribDivisor(index, divisor);
		}
		this->unbind();
	}

}