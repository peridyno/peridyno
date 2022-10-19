#include "CudaBuffer.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

namespace gl
{
	/// <summary>
	/// for cuda
	/// </summary>

	void CudaBuffer::allocate(int size)
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

		// map cuda resource
		size_t size0;
		cudaGraphicsMapResources(1, &resource);
		cudaGraphicsResourceGetMappedPointer(&devicePtr, &size0, resource);

	}


	void CudaBuffer::release()
	{
		Buffer::release();

		if (resource != 0)
		{
			cudaGraphicsUnmapResources(1, &resource);
			cudaGraphicsUnregisterResource(resource);
		}
	}

	void CudaBuffer::loadCuda(void* src, int size)
	{
		if (src == 0)
			return;

		if (size > this->size)
		{
			allocate(size);
		}

		cudaMemcpy(this->devicePtr, src, size, cudaMemcpyDeviceToDevice);
	}

}
