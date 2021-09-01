#include "Framebuffer.h"
#include <glad/glad.h>
#include <iostream>

namespace gl
{
	void Framebuffer::create()
	{
		glGenFramebuffers(1, &id);
	}

	void Framebuffer::release()
	{
		glDeleteFramebuffers(1, &id);
	}

	int Framebuffer::checkStatus()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
		int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE)
		{
			// TODO: error!
			exit(0);
		}
		return status;
	}

	unsigned int Framebuffer::current()
	{
		GLint fbo;
		glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
		return fbo;
	}

	void Framebuffer::clearColor(float r, float g, float b, float a)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
		glClearColor(r, g, b, a);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void Framebuffer::clearDepth(float depth)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
		glClearDepth(depth);
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	void Framebuffer::bind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
	}

	void Framebuffer::unbind()
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	void Framebuffer::setTexture2D(unsigned int attachment, unsigned int tex, int level)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
		glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, tex, level);
		glCheckError();
	}

	void Framebuffer::drawBuffers(int count, const unsigned int* buffers)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, id);
		glDrawBuffers(count, buffers);
	}
}