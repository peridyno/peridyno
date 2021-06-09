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
#include "GLBuffer.h"

namespace dyno
{
	class RenderTarget
	{
	public:
		RenderTarget();
		~RenderTarget();

		void initialize();
		void resize(int w, int h);
		void blitTo(unsigned int fbo);

	private:
		void drawColorTex();
		void bindColorTex();

		void drawGBufferTex();
		void bindGBufferTex();

		void drawSSAOTex();
		void bindSSAOTex();

		// OIT
		void drawOITLinkedList();


	private:
		GLFramebuffer mFramebuffer;

		// frame color
		GLTexture2D	  mColorTex;
		// frame depth
		GLTexture2D	  mDepthTex;

		// G-buffers
		GLTexture2D	  mAlbedoTex;
		GLTexture2D	  mNormalTex;
		GLTexture2D	  mPositionTex;

		// screen-space ambient occlusion
		GLTexture2D	  mSSAOTex;


		// Linked-List
		GLBuffer		mFreeNodeIdx;
		GLTexture2D		mHeadIndexTex;
		GLBuffer		mLinkedListBuffer;

		int width;
		int height;

		friend class RenderEngine;
	};
}