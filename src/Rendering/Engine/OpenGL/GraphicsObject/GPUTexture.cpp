#include "GPUTexture.h"

#include <Vector.h>

#include <glad/glad.h>

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



namespace dyno
{
template<typename T>
XTexture2D<T>::XTexture2D() {}
template<typename T>
XTexture2D<T>::~XTexture2D() {

#ifdef CUDA_BACKEND
#ifdef _WIN32
	if (resource) {
		cuSafeCall(cudaGraphicsUnregisterResource(resource));
	}
#endif
	buffer.clear();
#elif defined(VK_BACKEND)
		dyno::VkContext* vkCtx = dyno::VkSystem::instance()->currentContext();
		if(copyCmd)
			vkFreeCommandBuffers(vkCtx->deviceHandle(), vkCtx->commandPool(), 1, &copyCmd);
		if(buffer != VK_NULL_HANDLE)
			vkDestroyBuffer(vkCtx->deviceHandle(), buffer, nullptr);
		if(memory != VK_NULL_HANDLE)
			vkFreeMemory(vkCtx->deviceHandle(), memory, nullptr);
		this->closeHandle();
#endif
}

template<typename T>
	void XTexture2D<T>::create()
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
	bool XTexture2D<T>::isValid() const
{
	return width > 0 && height > 0;
}


template<typename T>
	void XTexture2D<T>::load(dyno::DArray2D<T> data)
{
#ifdef CUDA_BACKEND
	buffer.assign(data);
#endif // CUDA_BACKEND

#ifdef VK_BACKEND

		temp.assign(data);

		VkBuffer src = data.buffer();
		int      size = data.size() * sizeof(T);
		auto ctx = dyno::VkSystem::instance()->currentContext();
		auto device = ctx->deviceHandle();

		if (this->width != data.nx() ||
			this->height != data.ny()) {

			this->width = data.nx();
			this->height = data.ny();
			this->resized = true;

			// allocate buffer
			// free current buffer
			vkDestroyBuffer(device, buffer, nullptr);
			vkFreeMemory(device, memory, nullptr);
			buffer = VK_NULL_HANDLE;
			memory = VK_NULL_HANDLE;

			this->closeHandle();

			VkExternalMemoryHandleTypeFlags type;
			// OS platforms
#ifdef WIN32
			type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
			type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
			// create vulkan buffer
			{
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
			}

			// create memory
			{
				VkMemoryRequirements memRequirements;
				vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

				VkMemoryAllocateInfo allocInfo{};
				allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
				allocInfo.allocationSize = memRequirements.size;
				allocInfo.memoryTypeIndex = ctx->getMemoryType(memRequirements.memoryTypeBits,
					VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

				// enable export memory
				VkExportMemoryAllocateInfo memoryHandleEx{};
				memoryHandleEx.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
				memoryHandleEx.handleTypes = type;
				allocInfo.pNext = &memoryHandleEx;  // <-- Enabling Export

				if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
					throw std::runtime_error("failed to allocate buffer memory!");
				}
			}

		 	VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));

		}

		// get the real allocated size of the buffer
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);

		// copy data
		{
			if (copyCmd == VK_NULL_HANDLE) {
				copyCmd = ctx->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);
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
			ctx->flushCommandBuffer(copyCmd, ctx->transferQueueHandle(), false);
		}

		{
			//test copy back
			dyno::DArray2D<T> wtf;
			wtf.resize(width, height);

			// begin
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
			VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &beginInfo));

			VkBufferCopy copyRegion{};
			copyRegion.srcOffset = 0; // Optional
			copyRegion.dstOffset = 0; // Optional
			copyRegion.size = size;
			vkCmdCopyBuffer(copyCmd, buffer, wtf.buffer(), 1, &copyRegion);

			// end and flush
			ctx->flushCommandBuffer(copyCmd, ctx->transferQueueHandle(), false);

			temp.assign(wtf);
		}

		// get memory handle for importing
#ifdef WIN32
		VkMemoryGetWin32HandleInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
		info.memory = memory;
		info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

		auto vkGetMemoryWin32HandleKHR =
			PFN_vkGetMemoryWin32HandleKHR(vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR"));
		VK_CHECK_RESULT(vkGetMemoryWin32HandleKHR(device, &info, &handle));
#else
		VkMemoryGetFdInfoKHR info{};
		info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
		info.memory = memory;
		info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

		auto vkGetMemoryFdKHR =
			PFN_vkGetMemoryFdKHR(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
		VK_CHECK_RESULT(vkGetMemoryFdKHR(device, &info, &fd));
#endif  

#endif
	}

template<typename T>
void XTexture2D<T>::updateGL()
	{
#ifdef CUDA_BACKEND

		if (buffer.size() <= 0) {
			return;
		}

		if (width != buffer.nx() || height != buffer.ny()) {
			// resize texture
			this->release();
			this->create();
			this->resize(buffer.nx(), buffer.ny());

			width = buffer.nx();
			height = buffer.ny();

			// re-register resource when size changed...
			if (resource) {
				cuSafeCall(cudaGraphicsUnregisterResource(resource));
			}
			cuSafeCall(cudaGraphicsGLRegisterImage(&resource, this->id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		}
		if (resource == nullptr) return;
		
		// Map buffer objects to get CUDA device pointers
		cudaArray* texture_ptr;
		cuSafeCall(cudaGraphicsMapResources(1, &resource));
		cuSafeCall(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, resource, 0, 0));

			// copy data with pitch
			cuSafeCall(cudaMemcpy2DToArray(texture_ptr, 0, 0,
				buffer.begin(), buffer.pitch(), buffer.nx() * sizeof(T), buffer.ny(),
				cudaMemcpyDeviceToDevice));

		cuSafeCall(cudaGraphicsUnmapResources(1, &resource));

#endif // CUDA_BACKEND


#ifdef VK_BACKEND

		if (width <= 0 || height <= 0)
			return;

		if (this->resized)
		{
			this->resized = false;

			// re-import memory object
			if (memoryObject)
				glDeleteMemoryObjectsEXT(1, &memoryObject);
			glCreateMemoryObjectsEXT(1, &memoryObject);

#ifdef WIN32
			glImportMemoryWin32HandleEXT(memoryObject,
				width * height * sizeof(T) * 2,
				GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
			glImportMemoryFdEXT(memoryObject, width*height*sizeof(T)*2, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
#endif
			glCheckError();
			// named buffer
			if (this->id != GL_INVALID_INDEX)
				glDeleteTextures(1, &this->id);

			glCreateTextures(GL_TEXTURE_2D, 1, &this->id);
			glBindTexture(GL_TEXTURE_2D, this->id);
			//this->create();

			glTextureParameteri(this->id, GL_TEXTURE_TILING_EXT, GL_LINEAR_TILING_EXT);

			glCheckError();

			//glTexStorageMem2DEXT(GL_TEXTURE_2D, 
			//	1, GL_RGBA32F, width, height, memoryObject, 0);

			glTextureStorageMem2DEXT(this->id,
				1, GL_RGBA32F, width, height, memoryObject, 0);

			glCheckError();

			//Texture2D::load(temp.nx(), temp.ny(), temp.handle()->data());

			glCheckError();
		}


#endif // VK_BACKEND
	}

#ifdef VK_BACKEND
template<typename T>
void XTexture2D<T>::closeHandle() {
#ifdef WIN32
	if (handle) {
		CloseHandle(handle);
		handle = nullptr;
	}
#else
	if (fd >= 0) {
		close(fd);
		fd = -1;
	}
#endif
}
#endif

}


