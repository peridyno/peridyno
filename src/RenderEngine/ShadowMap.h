#pragma once

#include "GLFramebuffer.h"
#include "GLTexture.h"

class ShadowMap
{
public:
	ShadowMap(int w = 1024, int h = 1024);
	~ShadowMap();

	void initialize();
	// bind framebuffer for write
	void bind();
	// bind shadowmap texture for read
	void bindShadowTex();

private:
	// framebuffers
	GLFramebuffer	mFramebuffer;
	GLTexture2D		mShadowDepth;
	GLTexture2D		mShadowColor;

	int				width;
	int				height;
};