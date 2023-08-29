#include "GPUBuffer.h"

#include <glad/glad.h>
#include <iostream>

#ifdef CUDA_BACKEND
#include <cuda_gl_interop.h>
#endif

#ifdef VK_BACKEND
#include <VkSystem.h>
#include <VkContext.h>

#ifdef WIN32
#include <handleapi.h>
#else
#include <unistd.h>
#endif // WIN32

#endif // VK_BACKEND


namespace gl
{
	template<typename T>
	void XBuffer<T>::load(dyno::DArray<T> data)
	{
#ifdef VK_BACKEND
		this->loadVulkan(data.buffer(), data.bufferSize());
#endif // VK_BACKEND

#ifdef CUDA_BACKEND
		buffer.assign(data);
#endif // CUDA_BACKEND
	}

#ifdef VK_BACKEND
	void XBuffer::loadVulkan(VkBuffer src, int size) {

		if (src == nullptr || size <= 0) return;
		// simple strategy to reduce frequently memory allocation
		if (size > this->size || size < (this->size / 4)) {
			this->allocate(size * 2);
		}

		dyno::VkContext* vkCtx = dyno::VkSystem::instance()->currentContext();

		if (copyCmd == VK_NULL_HANDLE) {
			copyCmd = vkCtx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
		}

		// begin
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &beginInfo));

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0; // Optional
		copyRegion.dstOffset = 0; // Optional
		copyRegion.size = size;
		vkCmdCopyBuffer(copyCmd, src, buffer, 1, &copyRegion);

		// end and flush
		vkCtx->flushCommandBuffer(copyCmd, vkCtx->transferQueue, false);
	}
#endif // VK_BACKEND


	template<typename T>
	void XBuffer<T>::updateGL()
	{
		int size = buffer.size() * sizeof(T);
		if (size == 0)
			return;

#ifdef CUDA_BACKEND

		int newSize = this->size;

		// shrink
		if (size < (this->size / 2))
			newSize = size;
		// expand
		if (size > this->size)
			newSize = size * 1.5;

		// resized
		if(newSize != this->size) {
			printf("allocate XBuffer: %d -> %d\n", this->size, newSize);
			allocate(newSize);
			// need re-register resource
			if(resource != 0)
				cuSafeCall(cudaGraphicsUnregisterResource(resource));
			cuSafeCall(cudaGraphicsGLRegisterBuffer(&resource, id, cudaGraphicsRegisterFlagsWriteDiscard));
		}

		size_t size0;
		void* devicePtr = 0;
		cuSafeCall(cudaGraphicsMapResources(1, &resource));
		cuSafeCall(cudaGraphicsResourceGetMappedPointer(&devicePtr, &size0, resource));
		cuSafeCall(cudaMemcpy(devicePtr, buffer.begin(), size, cudaMemcpyDeviceToDevice));
		cuSafeCall(cudaGraphicsUnmapResources(1, &resource));

#endif // CUDA_BACKEND


#ifdef VK_BACKEND

		if (!resized)
			return;
		resized = false;

		// it seems that we need to re-create buffer and memory object...
		glDeleteBuffers(1, &id);
		glGenBuffers(1, &id);

		if(memoryObject)
			glDeleteMemoryObjectsEXT(1, &memoryObject);
		glCreateMemoryObjectsEXT(1, &memoryObject);

#ifdef WIN32
		glImportMemoryWin32HandleEXT(memoryObject, size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
		glCheckError();
#else
		//glImportMemoryFdEXT(bufGl.memoryObject, size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, bufGl.fd);
		// fd got consumed
		//bufGl.fd = -1;
#endif
		glNamedBufferStorageMemEXT(id, size, memoryObject, 0);
		glCheckError();

#endif // VK_BACKEND
	}

	template<typename T>
	int XBuffer<T>::count() const
	{
		return buffer.size();
	}

}
