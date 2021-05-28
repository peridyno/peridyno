#include "RenderTarget.h"
#include <glad/glad.h>
#include "Utility.h"

namespace dyno
{
	RenderTarget::RenderTarget()
	{
		width = height = 0; // uninitialized status
	}

	RenderTarget::~RenderTarget()
	{
		//TODO: cleanup 
	}

	void RenderTarget::initialize()
	{
		// create textures
		mColorTex.create();

		mDepthTex.internalFormat = GL_DEPTH_COMPONENT32;
		mDepthTex.format = GL_DEPTH_COMPONENT;
		mDepthTex.create();

		mAlbedoTex.create();
		mNormalTex.create();
		mPositionTex.create();

		mSSAOTex.create();

		mFluidTex[0].create();
		mFluidTex[1].create();

		// transparency
		mHeadIndexTex.internalFormat = GL_R32UI;
		mHeadIndexTex.format = GL_RED_INTEGER;
		mHeadIndexTex.type = GL_UNSIGNED_INT;
		mHeadIndexTex.wrapS = GL_CLAMP_TO_EDGE;
		mHeadIndexTex.wrapT = GL_CLAMP_TO_EDGE;
		mHeadIndexTex.create();

		// use resize to initialize texture...
		resize(1, 1);

		// create framebuffer
		mFramebuffer.create();

		// bind framebuffer texture

		// color and depth
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mDepthTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mColorTex.id);

		// G-buffer
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT1, mAlbedoTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT2, mNormalTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT3, mPositionTex.id);

		// screen-space ambient occlusion
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT4, mSSAOTex.id);

		// screen-space fluid rendering
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT5, mFluidTex[0].id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT6, mFluidTex[1].id);

		mFramebuffer.checkStatus();

		// OIT
		mFreeNodeIdx.create(GL_ATOMIC_COUNTER_BUFFER, GL_DYNAMIC_DRAW);
		mFreeNodeIdx.allocate(sizeof(int));

		mLinkedListBuffer.create(GL_SHADER_STORAGE_BUFFER, GL_DYNAMIC_DRAW);
		struct _Node
		{
			glm::vec4 color;
			float	  depth;
			unsigned int next;
			float	  _pad0;
			float     _pad1;
		};
		const int max_node = 1024 * 1024 * 8;
		mLinkedListBuffer.allocate(sizeof(_Node) * max_node);

		glCheckError();
	}


	void RenderTarget::resize(int w, int h)
	{
		if (w == width && h == height)
			return;
		width = w;
		height = h;

		mColorTex.resize(width, height);
		mDepthTex.resize(width, height);

		// g-buffer
		mAlbedoTex.resize(width, height);
		mNormalTex.resize(width, height);
		mPositionTex.resize(width, height);

		// ssao
		mSSAOTex.resize(width, height);

		// fluid
		mFluidTex[0].resize(width, height);
		mFluidTex[1].resize(width, height);

		// transparency
		mHeadIndexTex.resize(width, height);
	}


	void RenderTarget::blitTo(unsigned int fbo)
	{
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer.id);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glBlitFramebuffer(
			0, 0, width, height,
			0, 0, width, height,
			GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
	}

	void RenderTarget::drawGBufferTex()
	{
		const GLenum buffers[] = {
			GL_COLOR_ATTACHMENT1,
			GL_COLOR_ATTACHMENT2,
			GL_COLOR_ATTACHMENT3
		};
		mFramebuffer.bind();
		mFramebuffer.drawBuffers(3, buffers);
		mFramebuffer.clearColor();
		mFramebuffer.clearDepth();
		// enable depth write
		glDepthMask(true);
		glViewport(0, 0, width, height);
	}

	void RenderTarget::drawSSAOTex()
	{
		const GLenum buffers[] = { GL_COLOR_ATTACHMENT4 };
		mFramebuffer.bind();
		mFramebuffer.drawBuffers(1, buffers);
		mFramebuffer.clearColor(1, 1, 1, 1);
		glViewport(0, 0, width, height);
	}

	void RenderTarget::drawColorTex()
	{
		const GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
		mFramebuffer.bind();
		mFramebuffer.drawBuffers(1, buffers);
		glViewport(0, 0, width, height);
	}


	void RenderTarget::drawOITLinkedList()
	{
		mFramebuffer.bind();
		mFramebuffer.drawBuffers(0, 0);
		glViewport(0, 0, width, height);

		// reset free node index
		const int zero = 0;
		mFreeNodeIdx.load((void*)&zero, sizeof(int));
		// reset head index
		const int clear = 0xFFFFFFFF;
		mHeadIndexTex.clear((void*)&clear);

		glBindImageTexture(0, mHeadIndexTex.id, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
		mFreeNodeIdx.bindBufferBase(0);
		mLinkedListBuffer.bindBufferBase(0);
	}

	void RenderTarget::bindGBufferTex()
	{
		mDepthTex.bind(GL_TEXTURE0);
		mAlbedoTex.bind(GL_TEXTURE1);
		mNormalTex.bind(GL_TEXTURE2);
		mPositionTex.bind(GL_TEXTURE3);
	}

	void RenderTarget::bindSSAOTex()
	{
		mSSAOTex.bind(GL_TEXTURE4);
	}

	void RenderTarget::bindColorTex()
	{
		mColorTex.bind(GL_TEXTURE1);
	}


	void RenderTarget::drawFluidTex(int idx)
	{
		const GLenum buffers[] = { GL_COLOR_ATTACHMENT5 + idx };
		mFramebuffer.bind();
		mFramebuffer.drawBuffers(1, buffers);
		glViewport(0, 0, width, height);
	}

	void RenderTarget::bindFluidTex(int target, int idx)
	{
		mFluidTex[idx].bind(target);
	}
}