/**
 * Copyright 2023-2023 Jian SHI
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

#include "Texture.h"
#include <Vector.h>
#include <Array/Array2D.h>

#ifdef CUDA_BACKEND
struct cudaGraphicsResource;
#endif

#ifdef VK_BACKEND
#include <VkDeviceArray.h>
#endif

namespace gl {

	// texture for loading data from cuda/vulkan api
	template<typename T>
	class XTexture2D : public Texture2D
	{
	public:
		XTexture2D() {}
		~XTexture2D() {}

		virtual void create() override;
		bool isValid() const;

		// update OpenGL texture within GL context
		void updateGL();

		// load data to into an intermediate buffer
		void load(dyno::DArray2D<T> data);

	private:
		int width  = -1;
		int height = -1;

#ifdef CUDA_BACKEND
		dyno::DArray2D<T>		buffer;
		cudaGraphicsResource*	resource = 0;
#endif

#ifdef VK_BACKEND
		dyno::CArray2D<T> temp;

		VkBuffer		buffer = VK_NULL_HANDLE;
		VkDeviceMemory	memory = VK_NULL_HANDLE;
		bool resized		= false;
#ifdef WIN32
		HANDLE handle = nullptr;  // The Win32 handle
#else
		int fd = -1;
#endif
		// command for copy buffer
		VkCommandBuffer copyCmd = VK_NULL_HANDLE;
		// OpenGL memory object
		unsigned int	memoryObject = 0;
#endif
	};

	template class XTexture2D<dyno::Vec4f>;
	template class XTexture2D<dyno::Vec3f>;
	template class XTexture2D<dyno::Vec3u>;
}
