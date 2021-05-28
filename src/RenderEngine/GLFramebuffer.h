#pragma once

#include "GLObject.h"

class GLFramebuffer : public GLObject
{
public:
	void create() override;
	void release() override;

	void bind();
	void unbind();

	void clearColor(float r = 0.f, float g = 0.f, float b = 0.f, float a = 1.f);
	void clearDepth(float depth = 1.f);

	void setTexture2D(unsigned int attachment, unsigned int tex, int level = 0);

	void drawBuffers(int count, const unsigned int* buffers);


	int checkStatus();
};
