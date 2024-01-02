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

#include "GraphicsObject/Framebuffer.h"
#include "GraphicsObject/Texture.h"
#include "GraphicsObject/Buffer.h"
#include "GraphicsObject/Shader.h"

namespace dyno 
{
	class SSAO
	{
	public:
		SSAO();
		~SSAO();

		void resize(unsigned int w, unsigned int h);

	private:

		Program*		mSSAOProgram;

		Buffer			mSSAOKernelUBO;
		Texture2D		mSSAONoiseTex;

		Framebuffer		mDepthFramebuffer;
		Texture2D		mDepthTex;

		Framebuffer		mSSAOFramebuffer;
		Texture2D		mSSAOTex;

		Framebuffer		mSSAOFilterFramebuffer;
		Texture2D		mSSAOFilterTex;

		unsigned int	mWidth;
		unsigned int	mHeight;
	};
}

