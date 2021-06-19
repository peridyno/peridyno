#include "GLBuffer.h"

#include <glad/glad.h>
#include "Utility.h"

#include <cuda_gl_interop.h>

void GLBuffer::create()
{
	glGenBuffers(1, &id);
}

void GLBuffer::create(int target, int usage)
{
	glGenBuffers(1, &id);
	this->target = target;
	this->usage = usage;
	glCheckError();
}

void GLBuffer::release()
{
	glDeleteBuffers(1, &id);
	glCheckError();
}

void GLBuffer::bind()
{
	glBindBuffer(target, id);
	glCheckError();
}

void GLBuffer::unbind()
{
	glBindBuffer(target, 0);
	glCheckError();
}

void GLBuffer::allocate(int size)
{
	if (size == this->size)
		return;

	this->size = size;
	glBindBuffer(target, id); 
	glBufferData(target, size, 0, usage);
	glBindBuffer(target, 0);
	glCheckError();
}

void GLBuffer::load(void* data, int size, int offset)
{
	if ((size + offset) > this->size)
		allocate(size + offset);

	glBindBuffer(target, id);
	glBufferSubData(target, offset, size, data);
	glBindBuffer(target, 0);
	glCheckError();	
}

void GLBuffer::bindBufferBase(int idx)
{
	glBindBufferBase(this->target, idx, id);
	glCheckError();
}


/// <summary>
/// for cuda
/// </summary>

void GLCudaBuffer::allocate(int size)
{
	if (size == this->size)
		return;

	GLBuffer::allocate(size);
	
	// register the cuda resource after resize...
	if (resource != 0)
	{
		cudaGraphicsUnregisterResource(resource);
	}
	cudaGraphicsGLRegisterBuffer(&resource, id, cudaGraphicsRegisterFlagsWriteDiscard);
}

//void GLCudaBuffer::load(void* data, int size, int offset)
//{
//	if (size != this->size)
//		allocate(size);
//
//	GLBuffer::load(data, size, offset);	
//}

void GLCudaBuffer::release()
{
	GLBuffer::release();
	
	if (resource != 0)
	{
		cudaGraphicsUnregisterResource(resource);
	}
}

void GLCudaBuffer::loadCuda(void* devicePtr, int size)
{
	if (devicePtr == 0)
		return;

	if (size > this->size)
	{
		allocate(size);
	}

	void* cudaPtr = this->mapCuda();
	cudaMemcpy(cudaPtr, devicePtr, size, cudaMemcpyDeviceToDevice);
	this->unmapCuda();
}

void* GLCudaBuffer::mapCuda()
{
	size_t size0;
	void* cudaPtr = 0;
	cudaGraphicsMapResources(1, &resource);
	cudaGraphicsResourceGetMappedPointer(&cudaPtr, &size0, resource);
	return cudaPtr;
}

void GLCudaBuffer::unmapCuda()
{
	cudaGraphicsUnmapResources(1, &resource);
}
