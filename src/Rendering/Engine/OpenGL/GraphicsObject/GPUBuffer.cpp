#include "GPUBuffer.h"
#include "Shader.h"

#include <glad/glad.h>
#include <iostream>

#ifdef CUDA_GL_INTEROPER
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
	class BufferCopy {
	public:
		static BufferCopy* instance() {
			static BufferCopy inst;
			return &inst;
		}

		void proc(GLuint src, GLuint dst, 
			int src_pitch, 
			int dst_pitch, 
			int count) 
		{
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, src);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, dst);

			int pitch = src_pitch < dst_pitch ? src_pitch : dst_pitch;
			program.use();
			program.setInt("uSrcPitch", src_pitch);
			program.setInt("uDstPitch", dst_pitch);
			glCheckError();
			glDispatchCompute(count, pitch, 1);
			glCheckError();
		}

	private:
		BufferCopy() {
			const char* src = R"===(
#version 430
layout(local_size_x=1,local_size_y=1) in;
layout(binding=1,std430) buffer BufferSrc { uint vSrc[]; };
layout(binding=2,std430) buffer BufferDst { uint vDst[]; };
uniform int uSrcPitch = 1;
uniform int uDstPitch = 1;
void main() { vDst[uDstPitch * gl_GlobalInvocationID.x + gl_GlobalInvocationID.y] 
			= vSrc[uSrcPitch * gl_GlobalInvocationID.x + gl_GlobalInvocationID.y]; }
)===";
			Shader shader;
			shader.createFromSource(GL_COMPUTE_SHADER, src);
			program.create();
			program.attachShader(shader);
			program.link();
			shader.release();
		}
		~BufferCopy() {
			// ignore
			program.id = GL_INVALID_INDEX;
		}

		Program program;
	};

#ifdef VK_BACKEND
	template<typename T>
	void XBuffer<T>::allocateVkBuffer(int size) {

		auto ctx = dyno::VkSystem::instance()->currentContext();
		auto device = ctx->deviceHandle();

		// free current buffer
		vkDestroyBuffer(device, buffer, nullptr);
		vkFreeMemory(device, memory, nullptr);
		buffer = VK_NULL_HANDLE;
		memory = VK_NULL_HANDLE;

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
			VkMemoryDedicatedRequirements memDedecatedReq{ VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
			VkMemoryRequirements2 memRequirements{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
			memRequirements.pNext = &memDedecatedReq;
			VkBufferMemoryRequirementsInfo2 bufferReqs{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 };
			bufferReqs.buffer = buffer;
			vkGetBufferMemoryRequirements2(device, &bufferReqs, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.memoryRequirements.size;
			allocInfo.memoryTypeIndex = ctx->getMemoryType(memRequirements.memoryRequirements.memoryTypeBits, 
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

			// enable export memory
			VkExportMemoryAllocateInfo memoryHandleEx{};
			memoryHandleEx.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
			memoryHandleEx.handleTypes = type;
			allocInfo.pNext = &memoryHandleEx;  // <-- Enabling Export

			if (memDedecatedReq.requiresDedicatedAllocation) {
				VkMemoryDedicatedAllocateInfo delicated{};
				delicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
				delicated.buffer = buffer;
				memoryHandleEx.pNext = &delicated;
			}

			if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate buffer memory!");
			}
		}

		VK_CHECK_RESULT(vkBindBufferMemory(device, buffer, memory, 0));

		// get the real allocated size of the buffer
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);
		this->allocatedSize = req.size;

		// get memory handle for import
		closeHandle();
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
		auto vkGetMemoryWin32HandleKHR =
			PFN_vkGetMemoryFdKHR(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
		VK_CHECK_RESULT(vkGetMemoryWin32HandleKHR(device, &info, &fd));
#endif  
		// memory handle changed
		resized = true;
		printf("Buffer allocated %d bytes\n", allocatedSize);
	}

	template<typename T>
	void XBuffer<T>::closeHandle() {
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

	template<typename T>
	void XBuffer<T>::loadVkBuffer(VkBuffer src, int size) {
		assert(size >= 0);
		srcBufferSize = size;
		if (src == nullptr || size == 0) return;

		// simple strategy to reduce frequently memory allocation
		 if (size > this->allocatedSize || size < (this->allocatedSize / 4)) {
		//if (size != allocatedSize) {
			this->allocateVkBuffer(size * 2);
		}

		// copy data
		{
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
			vkCtx->flushCommandBuffer(copyCmd, vkCtx->transferQueueHandle(), false);
		}
	}

#endif // VK_BACKEND

	template<typename T>
	XBuffer<T>::XBuffer() {}
	template<typename T>
	XBuffer<T>::~XBuffer() {
#if defined(CUDA_BACKEND)
#ifdef _WIN32
		if (resource != 0) {
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
		closeHandle();
#endif
	}

	template<typename T>
	void XBuffer<T>::release() {
		Buffer::release();
#if defined(VK_BACKEND)
		if (tempBuffer != GL_INVALID_INDEX) {
			glDeleteBuffers(1, &tempBuffer);
			tempBuffer = GL_INVALID_INDEX;
		}
		if (memoryObject != GL_INVALID_INDEX) {
			glDeleteMemoryObjectsEXT(1, &memoryObject);
			memoryObject = GL_INVALID_INDEX;
		}
#endif
	}

	template<typename T>
	void XBuffer<T>::updateGL()
	{
#ifdef CUDA_BACKEND
		int size = buffer.size() * sizeof(T);
		if (size == 0)
			return;

		int newSize = this->size;

		// shrink
		if (size < (this->size / 2))
			newSize = size;
		// expand
		if (size > this->size)
			newSize = size * 1.5;

		// resized
		if(newSize != this->size) {
			allocate(newSize);
			// need re-register resource
#if defined(CUDA_GL_INTEROPER)
			if (resource != 0) {
				cuSafeCall(cudaGraphicsUnregisterResource(resource));
			}
			cuSafeCall(cudaGraphicsGLRegisterBuffer(&resource, id, cudaGraphicsRegisterFlagsWriteDiscard));
#endif
		}

#if defined(CUDA_GL_INTEROPER) 
		if (resource == nullptr) return;

		size_t size0;
		void* devicePtr = 0;
		cuSafeCall(cudaGraphicsMapResources(1, &resource));
		cuSafeCall(cudaGraphicsResourceGetMappedPointer(&devicePtr, &size0, resource));
		cuSafeCall(cudaMemcpy(devicePtr, buffer.begin(), size, cudaMemcpyDeviceToDevice));
		cuSafeCall(cudaGraphicsUnmapResources(1, &resource));
		//(cudaStreamSynchronize(0);
#else
		if(size >= 0) {
			void* data {NULL};
			glBindBuffer(target, id);
			glMapBuffer(target, GL_WRITE_ONLY);
			glGetBufferPointerv(target, GL_BUFFER_MAP_POINTER, &data);
			cudaMemcpy(data, buffer.begin(), size, cudaMemcpyDeviceToHost);
			glUnmapBuffer(target);
			glBindBuffer(target, 0);

			glCheckError();
		}
#endif


#endif // CUDA_BACKEND

#ifdef VK_BACKEND
		assert(allocatedSize != -1);
		// we need to re-create buffer and memory object when buffer is resized...
		if (resized)
		{
			resized = false;
			this->allocate(allocatedSize);
			// re-import memory object
			if (memoryObject) {
				glDeleteMemoryObjectsEXT(1, &memoryObject);
			}
			glCreateMemoryObjectsEXT(1, &memoryObject);
#ifdef WIN32
			glImportMemoryWin32HandleEXT(memoryObject, allocatedSize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
			glImportMemoryFdEXT(memoryObject, allocatedSize, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd);
#endif
			glCheckError();
			// named buffer
			if (tempBuffer != GL_INVALID_INDEX) {
				glDeleteBuffers(1, &tempBuffer);
			}
			// require glCreateBuffers instead of glGenBuffers
			glCreateBuffers(1, &tempBuffer);
			glCheckError();
			glNamedBufferStorageMemEXT(tempBuffer, allocatedSize, memoryObject, 0);
			glCheckError();
		}

		if(tempBuffer != GL_INVALID_INDEX) {
			// copy data with stride...
			BufferCopy* copy = BufferCopy::instance();
			int src_pitch = std::max<int>(1, sizeof(T) / sizeof(int));
			int dst_pitch = std::max<int>(1, sizeof(T) / sizeof(int));
			if (typeid(T) == typeid(dyno::Vec3f) || typeid(T) == typeid(dyno::Vec3i)) {
				dst_pitch = 3;
			}
			copy->proc(tempBuffer, this->id, src_pitch, dst_pitch, count() * sizeof(T) / sizeof(int));
		}
#endif // VK_BACKEND
	}

	template<typename T>
	int XBuffer<T>::count() const
	{
#ifdef VK_BACKEND
		return srcBufferSize / sizeof(T);
#endif

#ifdef CUDA_BACKEND
		return buffer.size();
#endif
	}

}
