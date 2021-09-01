#include "Buffer.h"

#include <glad/glad.h>

namespace gl
{

	void Buffer::create()
	{
		glGenBuffers(1, &id);
	}

	void Buffer::create(int target, int usage)
	{
		glGenBuffers(1, &id);
		this->target = target;
		this->usage = usage;
		glCheckError();
	}

	void Buffer::release()
	{
		glDeleteBuffers(1, &id);
		glCheckError();
	}

	void Buffer::bind()
	{
		glBindBuffer(target, id);
		glCheckError();
	}

	void Buffer::unbind()
	{
		glBindBuffer(target, 0);
		glCheckError();
	}

	void Buffer::allocate(int size)
	{
		if (size == this->size)
			return;

		this->size = size;
		glBindBuffer(target, id);
		glBufferData(target, size, 0, usage);
		glBindBuffer(target, 0);
		glCheckError();
	}

	void Buffer::load(void* data, int size, int offset)
	{
		if ((size + offset) > this->size)
			allocate(size + offset);

		glBindBuffer(target, id);
		glBufferSubData(target, offset, size, data);
		glBindBuffer(target, 0);
		glCheckError();
	}

	void Buffer::bindBufferBase(int idx)
	{
		glBindBufferBase(this->target, idx, id);
		glCheckError();
	}
	
}