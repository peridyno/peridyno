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

		// fluid layer
		void drawFluidTex(int idx);
		void bindFluidTex(int target, int idx);

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

		// for fluid rendering
		GLTexture2D	  mFluidTex[2];

		// Linked-List
		GLBuffer		mFreeNodeIdx;
		GLTexture2D		mHeadIndexTex;
		GLBuffer		mLinkedListBuffer;

		int width;
		int height;

		friend class RenderEngine;
	};
}