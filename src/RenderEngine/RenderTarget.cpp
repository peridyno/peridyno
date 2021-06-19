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


		// transparency
		mNodeIDTex.internalFormat = GL_R32I;
		mNodeIDTex.format = GL_RED_INTEGER;
		mNodeIDTex.type = GL_INT;
		mNodeIDTex.wrapS = GL_CLAMP_TO_EDGE;
		mNodeIDTex.wrapT = GL_CLAMP_TO_EDGE;
		mNodeIDTex.create();

		// use resize to initialize texture...
		resize(1, 1);

		// create framebuffer
		mFramebuffer.create();

		mFramebuffer.bind();

		// bind framebuffer texture
		mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mDepthTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT0, mColorTex.id);
		mFramebuffer.setTexture2D(GL_COLOR_ATTACHMENT1, mNodeIDTex.id);

		const GLenum buffers[] = {
			GL_COLOR_ATTACHMENT0,
			GL_COLOR_ATTACHMENT1
		};
		mFramebuffer.drawBuffers(2, buffers);

		mFramebuffer.checkStatus();
		mFramebuffer.unbind();
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
		mNodeIDTex.resize(width, height);
	}

	void RenderTarget::blit(unsigned int attachment)
	{
		glBindFramebuffer(GL_READ_FRAMEBUFFER, mFramebuffer.id);
		glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment);
		glBlitFramebuffer(
			0, 0, width, height,
			0, 0, width, height,
			GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT, GL_NEAREST);
		glCheckError();
	}

	void RenderTarget::bind()
	{
		mFramebuffer.bind();
		glViewport(0, 0, width, height);
	}
	   
}