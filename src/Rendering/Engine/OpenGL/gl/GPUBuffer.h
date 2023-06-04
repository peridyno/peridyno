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
#include <Array/Array.h>

#ifdef CUDA_BACKEND
struct cudaGraphicsResource;
#endif

#ifdef VK_BACKEND
#include <VkDeviceArray.h>
#endif

namespace gl
{

	// buffer for exchange data from simulation to rendering
	// please note that we use additional buffer for r/w consistency between simulation and rendering loop
	class XBuffer : public gl::Buffer
	{
	public:
		void release() override;
		void allocate(int size) override;

		template<typename T>
		void load(dyno::Array<T, DeviceType::GPU> data)
		{		

#ifdef VK_BACKEND
			this->loadVulkan(data.buffer(), data.bufferSize());
#endif // VK_BACKEND

#ifdef CUDA_BACKEND
			this->loadCuda(data.begin(), data.size() * sizeof(T));
#endif // CUDA_BACKEND
		}

		void mapGL();

	private:

#ifdef VK_BACKEND
		VkBuffer		buffer = VK_NULL_HANDLE;
		VkDeviceMemory	memory = VK_NULL_HANDLE;

		VkCommandBuffer copyCmd = VK_NULL_HANDLE;

#ifdef WIN32
		HANDLE handle = nullptr;  // The Win32 handle
#else
		int fd = -1;
#endif
		unsigned int memoryObject = 0;  // OpenGL memory object

		void loadVulkan(VkBuffer src, int size);
#endif	//VK_BACKEND


#ifdef CUDA_BACKEND
		cudaGraphicsResource*	resource = 0;
		void*					buffer = 0;		// local cuda buffer

		void loadCuda(void* src, int size);
#endif

		// resize flag
		bool resized = false;
	};
}