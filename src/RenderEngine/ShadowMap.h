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

#include "GLFramebuffer.h"
#include "GLTexture.h"
#include "RenderParams.h"
#include "module/GLVisualModule.h"
#include <vector>

namespace dyno
{
	class ShadowMap
	{
	public:
		ShadowMap(int w = 1024, int h = 1024);
		~ShadowMap();

		void initialize();

		void update(const std::vector<dyno::GLVisualModule*>& modules, const dyno::RenderParams& rparams);

	private:
		// framebuffers
		GLFramebuffer	mFramebuffer;
		GLTexture2D		mShadowDepth;

	public:
		int				width;
		int				height;
	};
}