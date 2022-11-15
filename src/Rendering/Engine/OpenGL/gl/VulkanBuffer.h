#pragma once
#include "Buffer.h"

#include <VkDeviceArray.h>

namespace gl
{
	class VulkanBuffer : public gl::Buffer
	{
	public:

		void create(int target, int usage) override;
		void release() override;

		void allocate(int size) override;
		void load(VkBuffer src, int size);

	private:
		VkBuffer		buffer = VK_NULL_HANDLE;
		VkDeviceMemory	memory = VK_NULL_HANDLE;
		VkCommandBuffer copyCmd = VK_NULL_HANDLE;

	private:

#ifdef WIN32
		HANDLE handle = nullptr;  // The Win32 handle
#else
		int fd = -1;
#endif
		unsigned int memoryObject = 0;  // OpenGL memory object

	};
}