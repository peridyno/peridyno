#include "GLCudaBuffer.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace dyno
{
	/// <summary>
	/// for cuda
	/// </summary>

	void GLCudaBuffer::allocate(int size)
	{
		if (size == this->size)
			return;

		Buffer::allocate(size);

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
		Buffer::release();

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
}