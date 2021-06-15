#include "ShadowMap.h"
#include <glad/glad.h>

ShadowMap::ShadowMap(int w, int h)
{
	width = w;
	height = h;
}

ShadowMap::~ShadowMap()
{

}

void ShadowMap::initialize()
{
	mShadowDepth.internalFormat = GL_DEPTH_COMPONENT32;
	mShadowDepth.format = GL_DEPTH_COMPONENT;	
	mShadowDepth.create();
	mShadowDepth.resize(width, height);		

	mFramebuffer.create();
	mFramebuffer.setTexture2D(GL_DEPTH_ATTACHMENT, mShadowDepth.id);
	mFramebuffer.bind();
	glDrawBuffer(GL_NONE);

	mFramebuffer.checkStatus();
	mFramebuffer.unbind();
}

void ShadowMap::bind()
{
	mFramebuffer.bind();
	mFramebuffer.clearDepth(1.0);
	glViewport(0, 0, width, height);
}

void ShadowMap::bindShadowTex()
{
	mShadowDepth.bind(GL_TEXTURE5);
}