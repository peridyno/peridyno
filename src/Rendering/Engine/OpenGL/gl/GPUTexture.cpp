#include "GPUTexture.h"

#include <Vector.h>

#include <glad/glad.h>
#ifdef CUDA_BACKEND
#include <cuda_gl_interop.h>
#endif

using namespace gl;

template<typename T>
void gl::XTexture2D<T>::create()
{
	if (typeid(T) == typeid(dyno::Vec4f)) {
		this->format = GL_RGBA;
		this->internalFormat = GL_RGBA32F;
		this->type = GL_FLOAT;
	}
	else if (typeid(T) == typeid(dyno::Vec3f)) {
		this->format = GL_RGB;
		this->internalFormat = GL_RGB32F;
		this->type = GL_FLOAT;
	}
	else if (typeid(T) == typeid(dyno::Vec3u)) {
		this->format = GL_RGB;
		this->internalFormat = GL_RGB8;
		this->type = GL_UNSIGNED_BYTE;
	}

	Texture2D::create();
}

template<typename T>
void gl::XTexture2D<T>::load(dyno::DArray2D<T> data)
{
#ifdef VK_BACKEND

#endif // VK_BACKEND

#ifdef CUDA_BACKEND
	buffer.assign(data);
#endif // CUDA_BACKEND
}

template<typename T>
void XTexture2D<T>::updateGL()
{
	int size = buffer.nx() * buffer.ny() * sizeof(T);

#ifdef CUDA_BACKEND

	// prepare dst texture
	if (this->id == GL_INVALID_INDEX)
	{
		this->create();
	}

	if (width != buffer.nx() || height != buffer.ny()) {
		// resize texture
		this->resize(buffer.nx(), buffer.ny());
		width = buffer.nx();
		height = buffer.ny();

		// re-register resource when size changed...
		if(resource)
			cuSafeCall(cudaGraphicsUnregisterResource(resource));
		cuSafeCall(cudaGraphicsGLRegisterImage(&resource, this->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}
	
	// Map buffer objects to get CUDA device pointers
	cudaArray* texture_ptr;
	cuSafeCall(cudaGraphicsMapResources(1, &resource));
	cuSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, resource, 0, 0));
	cuSafeCall(cudaMemcpyToArray(texture_ptr, 0, 0, buffer.begin(), size, cudaMemcpyDeviceToDevice));
	cuSafeCall(cudaGraphicsUnmapResources(1, &resource));

#endif // CUDA_BACKEND


#ifdef VK_BACKEND
	// TODO
#endif // VK_BACKEND
}



