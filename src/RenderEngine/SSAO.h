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

#include "gl/Framebuffer.h"
#include "gl/Texture.h"
#include "gl/Buffer.h"
#include "gl/Program.h"

namespace dyno 
{
	class SSAO
	{
	public:
		SSAO();
		~SSAO();

		void initialize();
		void resize(unsigned int w, unsigned int h);

	private:

		// SSAO
		gl::Buffer		mSSAOKernelUBO;
		gl::Texture2D		mSSAONoiseTex;
		gl::Program mSSAOProgram;

		gl::Framebuffer	mDepthFramebuffer;
		gl::Texture2D		mDepthTex;

		gl::Framebuffer	mSSAOFramebuffer;
		gl::Texture2D		mSSAOTex;

		gl::Framebuffer	mSSAOFilterFramebuffer;
		gl::Texture2D		mSSAOFilterTex;

		unsigned int	mWidth;
		unsigned int	mHeight;
	};
}

