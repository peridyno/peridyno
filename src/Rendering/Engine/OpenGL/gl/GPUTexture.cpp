#include "GPUTexture.h"

#include <Vector.h>

#include <glad/glad.h>
#ifdef CUDA_BACKEND
#include <cuda_gl_interop.h>
#endif

using namespace gl;

void XTexture2D<dyno::Vec4f>::updateGL()
{
	int size = sizeof(float) * buffer.nx() * buffer.ny() * 4;

#ifdef CUDA_BACKEND
	// prepare dst texture
	if (this->id == GL_INVALID_INDEX)
	{
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;
		this->create();
	}

	// resize texture
	this->resize(buffer.nx(), buffer.ny());

	cudaGraphicsResource* resource = 0;
	cuSafeCall(cudaGraphicsGLRegisterImage(&resource, this->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	
	// Map buffer objects to get CUDA device pointers
	cudaArray* texture_ptr;
	cuSafeCall(cudaGraphicsMapResources(1, &resource));
	cuSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, resource, 0, 0));
	cuSafeCall(cudaMemcpyToArray(texture_ptr, 0, 0, buffer.begin(), size, cudaMemcpyDeviceToDevice));
	cuSafeCall(cudaGraphicsUnmapResources(1, &resource));

	cuSafeCall(cudaGraphicsUnregisterResource(resource));
#endif // CUDA_BACKEND


#ifdef VK_BACKEND
	// TODO
#endif // VK_BACKEND
}



