#include "GLFramebuffer.h"
#include <glad/glad.h>
#include "Utility.h"

void GLFramebuffer::create()
{
	glGenFramebuffers(1, &id);
}

void GLFramebuffer::release()
{
	glDeleteFramebuffers(1, &id);
}

int GLFramebuffer::checkStatus()
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

void GLFramebuffer::clearColor(float r, float g, float b, float a)
{
	glBindFramebuffer(GL_FRAMEBUFFER, id);
	glClearColor(r, g, b, a);
	glClear(GL_COLOR_BUFFER_BIT);
}

void GLFramebuffer::clearDepth(float depth)
{
	glBindFramebuffer(GL_FRAMEBUFFER, id);
	glClearDepth(depth);
	glClear(GL_DEPTH_BUFFER_BIT);
}

void GLFramebuffer::bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, id);
}

void GLFramebuffer::unbind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void GLFramebuffer::setTexture2D(unsigned int attachment, unsigned int tex, int level)
{
	glBindFramebuffer(GL_FRAMEBUFFER, id);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, tex, level);
	glCheckError();
}

void GLFramebuffer::drawBuffers(int count, const unsigned int* buffers)
{
	glBindFramebuffer(GL_FRAMEBUFFER, id);
	glDrawBuffers(count, buffers);
}