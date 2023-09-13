/**
 * Copyright 2023 Xiaowei He
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

#include "Object.h"
#include "Platform.h"

#include "Array/Array2D.h"

#include "gl/GPUTexture.h"

namespace gl
{
	class Material : public Object
	{
	public:

		Material();
		~Material() override;
		void create() override;
		void release() override;
		void update();
		void updateGL();

	public:
		dyno::Vec3f ambient = { 0.0f, 0.0f, 0.0f };
		dyno::Vec3f diffuse = { 0.8f, 0.8f, 0.8f };
		dyno::Vec3f specular = { 0.0f, 0.0f, 0.0f };
		float roughness = 0.0f;
		float alpha = 1.0f;
		dyno::DArray2D<dyno::Vec4f> texColor;
		dyno::DArray2D<dyno::Vec4f> texBump;

#ifdef CUDA_BACKEND
		// color texture
		gl::XTexture2D<dyno::Vec4f> mColorTexture;
		gl::XTexture2D<dyno::Vec4f> mBumpTexture;
#endif

		bool mInitialized = false;
	};
};