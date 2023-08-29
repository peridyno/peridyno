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

#include <Array/Array2D.h>
#include "Texture.h"

namespace gl {

	// texture for loading data from cuda/vulkan api
	template<typename T>
	class XTexture2D : public Texture2D
	{
	public:
		XTexture2D() {}
		~XTexture2D() {}

	public:
		template<DeviceType device>
		void load(dyno::Array2D<T, device> data) {
#ifdef VK_BACKEND

#endif // VK_BACKEND

#ifdef CUDA_BACKEND
			buffer.assign(data);
#endif // CUDA_BACKEND
		}

		void updateGL();

	private:

#ifdef CUDA_BACKEND
		dyno::Array2D<T, DeviceType::GPU> buffer;
#endif
	};
}
