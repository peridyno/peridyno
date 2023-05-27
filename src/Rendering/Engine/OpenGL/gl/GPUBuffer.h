/**
 * Copyright 2017-2021 Jian SHI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "Platform.h"

#include "Buffer.h"

#ifdef CUDA_BACKEND
	struct cudaGraphicsResource;
#endif

#ifdef VK_BACKEND
#include <VkDeviceArray.h>
#endif

namespace gl
{
#ifdef CUDA_BACKEND
	class CudaBuffer : public gl::Buffer
	{
		GL_OBJECT(CudaBuffer)
	public:
		virtual void release() override;

		virtual void allocate(int size);

		void  loadCuda(void* src, int size);

	private:
		cudaGraphicsResource* resource = 0;
	};
#endif // CUDA_BACKEND

#ifdef VK_BACKEND
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
#endif	//VK_BACKEND
}