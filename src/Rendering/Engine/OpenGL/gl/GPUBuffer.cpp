#include "GPUBuffer.h"
#include "Shader.h"

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
layout(binding=1,std430) buffer BufferSrc { int vSrc[]; };
layout(binding=2,std430) buffer BufferDst { int vDst[]; };
uniform int uSrcPitch = 1;
uniform int uDstPitch = 1;
void main() { vDst[uDstPitch * gl_GlobalInvocationID.x + gl_GlobalInvocationID.y] 
			= vSrc[uSrcPitch * gl_GlobalInvocationID.x + gl_GlobalInvocationID.y]; }
)===";
			gl::Shader shader;
			shader.createFromSource(GL_COMPUTE_SHADER, src);
			program.create();
			program.attachShader(shader);
			program.link();
			shader.release();
		}

		gl::Program program;
	};

#ifdef VK_BACKEND
	template<typename T>
	void XBuffer<T>::allocateVkBuffer(int size) {

		auto ctx = dyno::VkSystem::instance()->currentContext();
		auto device = ctx->deviceHandle();

		// free current buffer
		vkDestroyBuffer(device, buffer, nullptr);
		vkFreeMemory(device, memory, nullptr);

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

		vkBindBufferMemory(device, buffer, memory, 0);

		// get the real allocated size of the buffer
		VkMemoryRequirements req;
		vkGetBufferMemoryRequirements(device, buffer, &req);
		this->allocatedSize = size;

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
		// memory handle changed
		resized = true;
		printf("Buffer allocated %d bytes\n", allocatedSize);
	}

	template<typename T>
	void XBuffer<T>::loadVkBuffer(VkBuffer src, int size) {

		srcBufferSize = size;
		if (src == nullptr || size <= 0) return;

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
			vkCtx->flushCommandBuffer(copyCmd, vkCtx->transferQueue, false);
		}
	}

#endif // VK_BACKEND


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
		// we need to re-create buffer and memory object when buffer is resized...
		if (resized)
		{
			resized = false;
			// re-import memory object
			if (memoryObject)
				glDeleteMemoryObjectsEXT(1, &memoryObject);
			glCreateMemoryObjectsEXT(1, &memoryObject);
#ifdef WIN32
			glImportMemoryWin32HandleEXT(memoryObject, allocatedSize, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
			//glImportMemoryFdEXT(bufGl.memoryObject, size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, bufGl.fd);
			// fd got consumed
			//bufGl.fd = -1;
#endif
			// named buffer
			if (tempBuffer != GL_INVALID_INDEX)
				glDeleteBuffers(1, &tempBuffer);
			glGenBuffers(1, &tempBuffer);
			glNamedBufferStorageMemEXT(tempBuffer, allocatedSize, memoryObject, 0);
			glCheckError();

			// allocate target buffer size
			this->allocate(allocatedSize);
		}

		// copy data with stride...
		BufferCopy* copy = BufferCopy::instance();
		int src_pitch = sizeof(T) / sizeof(int);
		int dst_pitch = sizeof(T) / sizeof(int);
		if (typeid(T) == typeid(dyno::Vec3f) || typeid(T) == typeid(dyno::Vec3i)) {
			dst_pitch = 3;
		}
		copy->proc(tempBuffer, this->id, src_pitch, dst_pitch, count());
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
