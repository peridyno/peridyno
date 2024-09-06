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

#include "Buffer.h"
#include <Array/Array.h>

#include <Vector.h>
#include <Matrix/Transform3x3.h>
#include <Module/TopologyModule.h>

#ifdef CUDA_BACKEND
struct cudaGraphicsResource;
#endif

#ifdef VK_BACKEND
#include <VkDeviceArray.h>
#endif

namespace dyno
{

	// buffer for exchange data from simulation to rendering
	// please note that we use additional buffer for r/w consistency between simulation and rendering loop
	template<typename T>
	class XBuffer : public Buffer
	{
	public:
		// update OpenGL buffer within GL context
		void updateGL();
		// return number of elements
		int  count() const;

		// load data to into an intermediate buffer
		template<typename T1>
		void load(dyno::DArray<T1> data)
		{
#ifdef VK_BACKEND
			this->loadVkBuffer(data.buffer(), data.bufferSize());
#endif // VK_BACKEND

#ifdef CUDA_BACKEND
			buffer.assign(data);
#endif // CUDA_BACKEND
		}

	private:

#ifdef VK_BACKEND
		VkBuffer		buffer = VK_NULL_HANDLE;
		VkDeviceMemory	memory = VK_NULL_HANDLE;
		int srcBufferSize		= -1;	// real size of the data
		int allocatedSize	= -1;	// allocated buffer size
#ifdef WIN32
		HANDLE handle = nullptr;  // The Win32 handle
#else
		int fd = -1;
#endif
		// command for copy buffer
		VkCommandBuffer copyCmd = VK_NULL_HANDLE;

		unsigned int	memoryObject = 0;			// OpenGL memory object
		unsigned int	tempBuffer = 0xffffffff;	// temp buffer
		bool resized = true;

		void loadVkBuffer(VkBuffer src, int size);
		void allocateVkBuffer(int size);

#endif	//VK_BACKEND


#ifdef CUDA_BACKEND
		dyno::DArray<T>	buffer;
		cudaGraphicsResource* resource = 0;
#endif
	};

	template class XBuffer<dyno::Vec2f>;
	template class XBuffer<dyno::Vec3f>;
	template class XBuffer<dyno::Transform3f>;
	template class XBuffer<dyno::TopologyModule::Edge>;
	template class XBuffer<dyno::TopologyModule::Triangle>;
}