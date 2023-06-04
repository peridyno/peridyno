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
	void gl::XBuffer::release()
	{
#ifdef VK_BACKEND
#ifdef WIN32
		CloseHandle(handle);
#else
		if (fd != -1)
		{
			close(fd);
			fd = -1;
		}
#endif
		if(memoryObject)
			glDeleteMemoryObjectsEXT(1, &memoryObject);
		// TODO: release command buffer?
#endif

#ifdef CUDA_BACKEND
		if (buffer)
			cudaFree(buffer);
		if (resource)
			cudaGraphicsUnregisterResource(resource);
#endif // CUDA_BACKEND

		// finally call buffer release
		Buffer::release();
	}

	void XBuffer::allocate(int size) 
	{
		std::cout << "allocate buffer: " << this->size << " -> " << size << " bytes" << std::endl;
		this->resized = true;

#ifdef CUDA_BACKEND
		if (buffer)
			cudaFree(buffer);
		cudaMalloc(&buffer, size); 
		cudaStreamSynchronize(0);
		this->size = size;
#endif // CUDA_BACKEND

#ifdef VK_BACKEND
		dyno::VkContext* ctx = dyno::VkSystem::instance()->currentContext();
		VkDevice device = ctx->deviceHandle();

		// create new vulkan buffer
		{
			// free current buffer
			vkDestroyBuffer(device, buffer, 0);
			vkFreeMemory(device, memory, 0);

			// OS platforms
#ifdef WIN32
			VkExternalMemoryHandleTypeFlags type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
			VkExternalMemoryHandleTypeFlags type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = size;
			bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			VkExternalMemoryBufferCreateInfo externalInfo{};
			externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
			externalInfo.handleTypes = type;
			bufferInfo.pNext = &externalInfo;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to create buffer!");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = ctx->getMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			// enable export memory
			VkExportMemoryAllocateInfo memoryHandleEx{};
			memoryHandleEx.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			memoryHandleEx.handleTypes = type;
			allocInfo.pNext = &memoryHandleEx;  // <-- Enabling Export

			if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate buffer memory!");
			}

			vkBindBufferMemory(device, buffer, memory, 0);
		}

		// get the real allocated size of the buffer
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);
		this->size = req.size;

		// get memory handle for import
#ifdef WIN32
		VkMemoryGetWin32HandleInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		info.memory = memory;
		info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

		auto vkGetMemoryWin32HandleKHR =
			PFN_vkGetMemoryWin32HandleKHR(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));

		vkGetMemoryWin32HandleKHR(device, &info, &handle);
#else
		// TODO: for linux and other OS
#endif  

#endif
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


#ifdef CUDA_BACKEND
	void XBuffer::loadCuda(void* src, int size)
	{
		if (src == nullptr || size <= 0) return;
		// simple strategy to reduce frequently memory allocation
		if (size > this->size || size < (this->size / 4)) {
			this->allocate(size * 2);
		}

		cudaMemcpy(buffer, src, size, cudaMemcpyDeviceToDevice);
		//cudaStreamSynchronize(0);
	}
#endif


	void XBuffer::mapGL()
	{

#ifdef CUDA_BACKEND
		
		if (resized)
		{
			resized = false;
			// resize buffer...		
			glBindBuffer(target, id);
			glBufferData(target, size, 0, usage);
			glBindBuffer(target, 0);

			// register the cuda resource after resize...
			if (resource != 0) {
				cudaGraphicsUnregisterResource(resource);
			}
			cudaGraphicsGLRegisterBuffer(&resource, id, cudaGraphicsRegisterFlagsWriteDiscard);
		}

		size_t size0;
		void* devicePtr = 0;
		cudaGraphicsMapResources(1, &resource);
		cudaGraphicsResourceGetMappedPointer(&devicePtr, &size0, resource);
		cudaMemcpy(devicePtr, buffer, size, cudaMemcpyDeviceToDevice);
		cudaGraphicsUnmapResources(1, &resource);
		//(cudaStreamSynchronize(0);

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

}
